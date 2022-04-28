import torch
import torchvision

from typing import Tuple
from torch import Tensor


def nms(boxes, scores, iou_threshold):
    """

    :param boxes: Tensor[N, 4], in (x1, y1, x2, y2) format
    :param scores: Tensor[N]
    :param iou_threshold: float
    :return:
    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes: Tensor,
                scores: Tensor,
                idxs: Tensor,
                iou_threshold):
    """
    Performs nms in batched forms;
    :param boxes: Tensor[N, 4], in (x1, y1, x2, y2) format;
    :param scores: Tensor[N];
    :param idxs: Tensor[N], indices of the categories for each one of the boxes;
    :param iou_threshold:
    :return:
    """
    # if boxes is empty, return torch.empty()
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # get the maximum (x1, y1, x2, y2) coordinates in boxes
    max_coordinate = boxes.max()

    # generate a huge offsets for every class
    # note that idxs.to() method just make sure `idxs` and `boxes` have the same dtype and device
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # when boxes add offsets, boxes from different classes will not overlap
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)

    return keep


def remove_small_boxes(boxes, min_size):
    """
    remove boxes which contains at least one side smaller than min_size;
    :param boxes: Tensor[N, 4], in (x1, y1, x2, y2) format;
    :param min_size:
    :return:
    """
    # calculate the width and height of the boxes
    w = boxes[:, 2] - boxes[:, 0]  # width
    h = boxes[:, 3] - boxes[:, 1]  # height

    keep = torch.logical_and(torch.ge(w, min_size), torch.ge(h, min_size))
    keep = torch.where(keep)[0]

    return keep


def clip_boxes_to_image(boxes: Tensor,
                        size):
    """
    Clip boxes so that they lie inside an image
    :param boxes: Tensor[N, 4], (x1, y1, x2, y2) format;
    :param size:
    :return:
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    h, w = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(w, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(h, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=w)  # make boxes_x into range 0 and w
        boxes_y = boxes_y.clamp(min=0, max=h)  # make boxes_y into range 0 and h

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def box_area(boxes: Tensor):
    """
    Compute the area of the boxes;
    :param boxes: Tensor[N, 4], in (x1, y1, x2, y2) format;
    :return:
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor,
            boxes2: Tensor):
    """

    :param boxes1: Tensor[N, 4];
    :param boxes2: Tensor[M, 4];
    :return:
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (x1, None, y1)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (x2, None, y2)

    wh = (rb - lt).clamp(min=0)  # (w, None, h)

    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1 + area2 - inter)
    return iou


if __name__ == '__main__':
    _boxes1 = torch.Tensor([[300, 300, 350, 350], [200, 300, 500, 400]])
    _boxes2 = torch.Tensor([[200, 310, 500, 610], [200, 100, 400, 900]])
    _iou = box_iou(_boxes1, _boxes2)
    # clipped = clip_boxes_to_image(_boxes, (400, 400))
    # print(clipped)
    # _keep = remove_small_boxes(_boxes, 60)
    # print(_keep)
