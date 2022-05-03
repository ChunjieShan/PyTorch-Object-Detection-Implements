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
