from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import torchvision

from . import detect
from . import box as box_ops
from .image_list import ImageList


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))

    return num_anchors, pre_nms_top_n


class AnchorGenerator(nn.Module):
    """
    Anchors generator.
    """
    __annotations__ = {
        "cell_anchors": Optional[List[Tensor]],
        "_cache": Dict[str, List[Tensor]]
    }

    def __init__(self,
                 sizes=(128, 256, 512),
                 aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)

        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        """
        computing anchors sizes;
        :param scales: sqrt(anchor_area)
        :param aspect_ratios: h/w ratios
        :param dtype: float32
        :param device: cpu/gpu
        :return:
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # left-top and right-bottom coordinates relative to anchor center(0, 0)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device == device:
                return

        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios, in zip(self.sizes, self.aspect_ratios)
        ]

        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        # computing amount of objects on every feature layers
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        """
        anchors position in grid coordinate axis map into origin image;
        :param grid_sizes:
        :param strides:
        :return:
        """

        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # [grid_width], correspond to column of the original.
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # [grid_height], correspond to row of the original.
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # [grid_height], [grid_width] -> [grid_height, grid_width]
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # computing the anchors' coordinates offsets on the original
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            # add the base_anchor and offsets then we get the original coordinates
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)

            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        # cache anchors
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]

        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list: ImageList, feature_maps):
        # prediction feature maps shape
        grid_sizes = list([feature_map[-2:] for feature_map in feature_maps])

        # input images shape
        image_size = image_list.tensors.shape[-2:]
        # dtype and device
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # one step in feature map equals to n pixel stride in original image
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        # generate anchors template

        self.set_cell_anchors(dtype, device)

        # computing all anchors coordinates
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])

        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_img = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_img.append(anchors_per_feature_map)
            anchors.append(anchors_in_img)

        anchors = [torch.cat(anchors_per_img) for anchors_per_img in anchors]
        self._cache.clear()

        return anchors


class RPNHead(nn.Module):
    """
    Region Proposal Network head.
    Calculating objectness and boxes.
    """
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1))

        # objectness
        self.cls_logits = nn.Conv2d(in_channels,
                                    num_anchors,
                                    kernel_size=(1, 1),
                                    stride=(1, 1))

        # boxes regression
        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_anchors * 4,
                                   kernel_size=(1, 1),
                                   stride=(1, 1))

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))

        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    """
    permute and reshape tensors;
    :param layer:  objectness or coordinates;
    :param N: batch size;
    :param A: anchors_num_per_position;
    :param C: classes_num or 4
    :param H: height;
    :param W: width;
    :return:
    """
    # view is equal to reshape, but it only works on contiguous tensor
    # we need to use reshape either.
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)  # (N, H, W, -1 C)
    layer = layer.reshape(N, -1, C)

    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    """
    box_cls, box_regression -> [N, -1, C]
    :param box_cls: classification probability;
    :param box_regression: bbox coordinates;
    :return:
    """
    box_cls_flattened = []
    box_regression_flatten = []

    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # classes_num = 1
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]

        A = Ax4 // 4  # anchors
        C = AxC // A  # num_classes

        # [N, -1, C]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N, -1, C]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flatten.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flatten, dim=1).reshape(-1, 4)

    return box_cls, box_regression


class RegionProposalNetwork(nn.Module):
    """
    RPN Implementation;
    """

    __annotations__ = {
        'box_coder': detect.BoxCoder,
        'proposal_matcher': detect.Matcher,
        'fg_bg_sampler': detect.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int]
    }

    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n,
                 nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = detect.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # training
        # compute similarity through IoU
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = detect.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_match=True
        )

        self.fg_bg_sampler = detect.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        # testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        """
        match anchors to gt, sample positive, negative samples
        :param anchors:
        :param targets:
        :return:
        """
        labels = []
        matched_gt_boxes = []

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # IoU between gt and anchors
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                # get the biggest IoU indices
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding gt for each proposal
                # set negative values to 0 to avoid out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                # get positive samples
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # background samples
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between the thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)

        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        """
        get top n indices anchors;
        :param objectness: object probability;
        :param num_anchors_per_level:
        :return:
        """
        r = []  # indices
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)

            # Returns the k largest elements of the given input tensor along a given dimension
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)

            offset += num_anchors

        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        """
        filter small bbox and nms, get post_nms_top_n targets;
        :param proposals: bbox coordinates;
        :param objectness: objectness confidence;
        :param image_shapes: image size;
        :param num_anchors_per_level: anchors num;
        :return:
        """
        num_images = proposals.shape[0]
        device = proposals.device

        # avoid backprop when training
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # Return a tensor of size filled with fill_value
        # record indices of anchors on different prediction feature layers
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, dim=0)

        # expand levels to the same size as objectness
        levels = levels.reshape(1, -1).expand_as(objectness)

        # get top pre_nms_top_n anchors indices
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        # get top-n objectness
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []

        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # clip boxes to image
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low score boxes
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # nms
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only top-k scoring predictions
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)

        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_target):
        # positive and negative samples indices
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        # get non-zero indices
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # concat positive indices with negative indices
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_target = torch.cat(regression_target, dim=0)

        box_loss = detect.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_target[sampled_pos_inds],
            beta=1/9,
            size_average=False
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self,
                images,
                features,
                targets=None):
        features = list(features.values())
        # get objectness and bbox coordinates
        objectness, pred_bbox_deltas = self.head(features)

        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)

        # num_anchors_per_level
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # adjust tensors' format and shapes
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # detach deltas to avoid backprop
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # get rid of small bbox, compute nms and get post nms top n targets
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            # compute the most matchable gt
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )

            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }

        return boxes, losses


