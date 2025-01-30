import torch


def get_intersection(pred: torch.Tensor, gt: torch.Tensor):
    """
    print("pred shape", pred.shape)
    print("gt shape", gt.shape)
    print("pred", pred)
    print("gt", gt)
    """
    assert len(pred.shape) == len(gt.shape) and gt.shape[-1] == 4
    
    gt = gt.view(1, -1, 4)
    pred = pred.view(-1, 1, 4)
    
    xmin = torch.maximum(gt[..., 0], pred[..., 0])
    ymin = torch.maximum(gt[..., 1], pred[..., 1])
    xmax = torch.minimum(gt[..., 2], pred[..., 2])
    ymax = torch.minimum(gt[..., 3], pred[..., 3])
    
    area = torch.relu(xmax - xmin) * torch.relu(ymax - ymin)
    return area


def get_union(pred: torch.Tensor, gt: torch.Tensor, intersection: torch.Tensor=None):
    assert len(pred.shape) == len(gt.shape) and gt.shape[-1] == 4

    gt = gt.view(-1, 4)
    pred = pred.view(-1, 4)

    if intersection is None:
        intersection = get_intersection(pred, gt)

    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])

    gt_area = gt_area.unsqueeze(0)
    pred_area = pred_area.unsqueeze(1)
    return gt_area + pred_area - intersection


def intersection_over_first(boxes1: torch.Tensor, boxes2: torch.Tensor):
    intersection = get_intersection(boxes1, boxes2)
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes1_area = boxes1_area.unsqueeze(1)
    return intersection / boxes1_area


def intersection_over_union(pred: torch.Tensor, gt: torch.Tensor):
    assert len(pred.shape) == len(gt.shape) and gt.shape[-1] == 4 and len(gt.shape) == 2
    intersection = get_intersection(pred, gt)
    union = get_union(pred, gt)
    iou = intersection / union
    return iou