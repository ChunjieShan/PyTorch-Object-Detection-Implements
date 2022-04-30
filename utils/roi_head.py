from typing import Optional, List, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

import detect
import box as box_ops
# from . import detect
# from . import box as box_ops


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster-RCNN;
    :param class_logits: prediction label, [num_anchors, num_classes];
    :param box_regression: bbox regression;
    :param labels:  class labels;
    :param regression_targets: bbox labels;
    :return:
    """
    labels = torch.cat(labels, 0)
    regression_targets = torch.cat(regression_targets, 0)

    classification_loss = F.cross_entropy(class_logits, labels)  # classification loss

    # return the indices where labels > 0
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]
    labels_pos = labels[sampled_pos_inds_subset]

    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    # calculating bbox loss
    box_loss = detect.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False
    ) / labels.numel()

    return classification_loss, box_loss


class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': detect.BoxCoder,
        'proposal_matcher': detect.Matcher,
        'fg_bg_sampler': detect.BalancedPositiveNegativeSampler
    }

    def __init__(self,
                 box_roi_pool,  # Multi-scale RoIAlign pooling
                 box_head,  # Two MLP heads
                 box_predictor,  # Fast-RCNN Predictor
                 # Faster-RCNN training
                 fg_iou_thresh,
                 bg_iou_thresh,
                 batch_size_per_image,
                 positive_fraction,
                 bbox_reg_weights,
                 # Faster-RCNN inference
                 score_thresh,
                 nms_thresh,
                 detection_per_img):

        super(RoIHeads, self).__init__()

        # calculating similarity through IoU
        self.box_similarity = box_ops.box_iou
        # assign gt boxes for each proposal
        self.proposal_matcher = detect.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            False
        )

        self.fg_bg_sampler = detect.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)

        self.box_coder = detect.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        """
        Assigning each proposal to gt_boxes
        :param proposals:
        :param gt_boxes:
        :param gt_labels:
        :return:
        """
        matched_idxs = []
        labels = []

        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:  # no gt boxes, bg
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )

            else:
                # calculating iou between proposal and gt
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                # if iou < low_thresh: return -1,
                # low_thresh <= iou < high_thresh: return -2
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                # minimum value limitation, in case label oversize
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                # get matched label
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0  # background

                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # abandoned

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)

        return matched_idxs, labels