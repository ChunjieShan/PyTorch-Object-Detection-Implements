import torch
import math
from typing import List, Tuple
from torch import Tensor


class BalancedPositiveNegativeSampler:
    def __init__(self, batch_size_per_image, positive_fraction):
        """

        :param batch_size_per_image: number of elements to be selected per image;
        :param positive_fraction: percentage of positive samples per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs: List[Tensor]):
        """

        :param matched_idxs: list of tensors, which contains -1, 0, or positive values;
        :return:
        """
        pos_idx = []
        neg_idx = []

        for matched_idxs_per_image in matched_idxs:
            # positive sample if matched_idxs_per_image >= 1
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            # negative sample if matched_idxs_per_iamge = 0
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            # amount of positive image samples
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # if it's not enough, just use all the positive samples
            num_pos = min(positive.numel(), num_pos)
            # amount of negative image samples
            num_neg = self.batch_size_per_image - num_pos
            # if it's not enough, just use all the negative samples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative samples
            # returns a random permutation of integers from 0 to n - 1
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            # get positive and negative samples indices
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # initialize the mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            # select keep which images as positive samples
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


@torch.jit.script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    """

    :param reference_boxes: reference boxes(gt)
    :param proposals: boxes to be encoded(anchors)
    :param weights:
    :return:
    """
    wx, wy, ww, wh = weights[0], weights[1], weights[2], weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # parse width and height
    ex_width = proposals_x2 - proposals_x1
    ex_height = proposals_y2 - proposals_y1

    # parse coordinates of center points
    ex_ctr_x = proposals_x1 + 0.5 * ex_width
    ex_ctr_y = proposals_y1 + 0.5 * ex_height

    # the same as above
    gt_width = reference_boxes_x2 - reference_boxes_x1
    gt_height = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_width
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_height

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_width
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_height
    targets_dw = ww * torch.log(gt_width / ex_width)
    targets_dh = wh * torch.log(gt_height / ex_height)

    targets = torch.cat([targets_dx, targets_dy, targets_dw, targets_dh], dim=1)
    return targets


class BoxCoder:
    """
    This class encodes and decodes a set of boxes into label format.
    """
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """

        :param weights: 4-element tuple;
        :param bbox_xform_clip: float;
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self,
               reference_boxes: List[Tensor],
               proposals: List[Tensor]):
        """
        calculating regression label through anchor boxes;
        :param reference_boxes: gt boxes
        :param proposals: anchors/proposals
        :return:
        """
        # get the anchors amount for splitting
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        # dx, dy, dw, dh
        targets = self.encode_single(reference_boxes, proposals)
        # split the targets
        return targets.split(boxes_per_image, 0)

    def encode_single(self,
                      reference_boxes: Tensor,
                      proposals: Tensor):
        """
        Encode a set of proposals with respect to some reference boxes;
        :param reference_boxes:
        :param proposals: boxes to be encoded
        :return:
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self,
               rel_codes: Tensor,
               boxes: List[Tensor]):
        """
        Decode a set of boxes
        :param rel_codes: bbox regression parameters
        :param boxes: anchors/proposals
        :return:
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, Tensor)
        boxes_per_image = [len(b) for b in boxes]
        concat_boxes = torch.cat(boxes, 0)

        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        # decode bboxes

    def decode_single(self,
                      rel_codes: Tensor,
                      boxes: Tensor):
        """
        Decode a single boxes；
        :param rel_codes: encoded boxes;
        :param boxes: reference boxes;
        :return:
        """
        boxes = boxes.to(rel_codes.device)
        # x1, y1, x2, y2
        # reference boxes: x, y, w, h
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + width * 0.5
        ctr_y = boxes[:, 1] + height * 0.5

        wx, wy, ww, wh = self.weights
        # anchor boxes: x, y, w, h
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # limit max values
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * width[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * height[:, None] + ctr_y[:, None]

        pred_w = torch.exp(dw) * width[:, None]
        pred_h = torch.exp(dh) * height[:, None]

        # pred_boxes: x1, y1, x2, y2
        pred_boxes_x1 = pred_ctr_x - Tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_ctr_x) * pred_w
        pred_boxes_y1 = pred_ctr_y - Tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_ctr_y) * pred_h
        pred_boxes_x2 = pred_ctr_x + Tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_ctr_x) * pred_w
        pred_boxes_y2 = pred_ctr_y + Tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_ctr_y) * pred_h

        pred_boxes = torch.cat([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2).flatten(1)
        return pred_boxes


class Matcher:
    def __init__(self, high_thresh, low_thresh, allow_low_quality_match=False):
        """

        :param high_thresh: quality threshold higher than or equal to this value are candidates;
        :param low_thresh:
        :param allow_low_quality_match:
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_thresh < high_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.allow_low_quality_match = allow_low_quality_match

    def __call__(self, match_quality_matrix: Tensor):
        """
        Calculating the iou between gt and anchors and recording indices;
        :param match_quality_matrix: [M, N];
        :return:
        """
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No gt boxes during training!")
            else:
                raise ValueError("No proposal boxes during training!")

        # match_quality_matrix is M(gt boxes) x N(predicted)
        # each column represent the iou values between an anchor and gt.
        # matched_vals: max iou values
        # matches: indices
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_match:
            all_matches = matches.clone()
        else:
            all_matches = None

        below_low_threshold = matched_vals < self.low_thresh
        between_thresholds = (matched_vals >= self.low_thresh) and (matched_vals < self.high_thresh)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS
        if self.allow_low_quality_match:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    @staticmethod
    def set_low_quality_matches_(matches, all_matches, match_quality_matrix):
        """
        Produce matches if there are only low quality matches;
        :param matches:
        :param all_matches:
        :param match_quality_matrix:
        :return:
        """
        # for each gt, find the prediction with which it has the highest quality
        # highest_quality_for_each_gt: the maximum iou value
        highest_quality_for_each_gt, _ = match_quality_matrix.max(dim=1)

        # find highest iou quality between a single gt and anchors
        # gt_pred_pairs_of_highest_quality: [gt index, prediction_index]
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_for_each_gt[:, None])
        )
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


def smooth_l1_loss(inputs: Tensor,
                   targets: Tensor,
                   beta: float = 1. / 9,
                   size_average: bool = True):
    """

    :param inputs:
    :param targets:
    :param beta:
    :param size_average:
    :return:
    """
    n = torch.abs(targets - inputs)
    # targets - inputs < beta
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()

    return loss.sum()