import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
import torchvision

from .image_list import ImageList


@torch.jit.unused
def _resize_image_onnx(image, self_min_size, self_max_size):
    from torch.onnx import operators
    im_shape = operators.shape_as_tensor(image)[-2:]
    min_size = torch.min(im_shape).to(dtype=torch.float32)  # short side
    max_size = torch.max(im_shape).to(dtype=torch.float32)  # long side
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)
    
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False
    )[0]

    return image


def _resize_image(image, self_min_size, self_max_size):
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))
    max_size = float(torch.max(im_shape))
    scale_factor = self_min_size / self_max_size

    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size

    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False
    )[0]

    return image


def resize_boxes(boxes, original_size, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]

    ratios_height, ratios_width = ratios
    # removes a tensor dimension
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmin = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean  # mean of normalization
        self.image_std = image_std    # std of normalization

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)

        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        """
        `random_choice` via torch ops
        :param k:
        :return:
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self, image, target):
        h, w = image.shape[-2:]

        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])

        if torchvision._is_tracing():
            image = _resize_image_onnx(image, size, float(self.max_size))

        else:
            image = _resize_image(image, size, float(self.max_size))

        if target is not None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        max_size = []

