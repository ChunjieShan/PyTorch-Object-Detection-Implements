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
                 detection_per_img # how many prediction results per image
                 ):

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

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []

        # get every positive and negative indices
        for idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # get every indices
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)
            sampled_inds.append(img_sampled_inds)

        return sampled_inds

    def add_gt_proposal(self, proposals, gt_boxes):
        """
        Concatenate gts and proposals
        :param proposals:
        :param gt_boxes:
        :return:
        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def check_targets(self, targets):
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(self,
                                proposals: List[Tensor],
                                targets):
        """
        Select positive and negative samples, get labels and bboxes info.
        :param proposals:
        :param targets:
        :return:
        """
        # check target if it's empty
        self.check_targets(targets)
        # add this line or torch.jit.script() won't pass
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device

        # get labeled bboxes and classification infos
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append gt boxes to proposal
        proposals = self.add_gt_proposal(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of pos-neg samples
        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []
        num_images = len(proposals)

        # traversing each image
        for img_id in range(num_images):
            # get pos-neg indices
            img_sampled_inds = sampled_inds[img_id]
            # get pos-neg proposals
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # get pos-neg class labels
            labels[img_id] = labels[img_id][img_sampled_inds]
            # get gt indices
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            # get this image's gt boxes
            gt_boxes_in_img = gt_boxes[img_id]
            # if it's a background image
            if gt_boxes_in_img.numel() == 0:
                gt_boxes_in_img = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_img)

        # get regression labels according to gt and proposals
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets

    def postprocess_detections(self,
                               class_logits: Tensor,
                               box_regression: Tensor,
                               proposals: List[Tensor],
                               image_shapes: List[Tuple[int, int]]):
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        # bbox amount of each image
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # get final bbox coordinates according to proposals
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # get classification result
        pred_scores = F.softmax(class_logits, -1)
        # split scores and bboxes per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # traversing prediction infos
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # crop pred boxes
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove background(index=0)
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low score boxes
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove small targets
            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # nms
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep top k
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,
                proposals,
                image_shapes,
                targets=None):
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t['boxes'].dtype in floating_point_types, "Targets boxes must be float."
                assert t['labels'].dtype == torch.int64, "Labels must be int64."

        if self.training:
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        # pass features and proposal to RoIAlign Pooling
        box_features = self.box_roi_pool(features, proposals, image_shapes)

        # predicting labels and bbox
        class_logits, box_regression = self.box_predictor(box_features)
        # to hint the jit compiler the type
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}

        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            # get classification and regression losses
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }

        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(image_shapes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i]
                    }
                )

        return result, losses
