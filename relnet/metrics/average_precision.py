import torch
import numpy as np

from mmdet.evaluation.functional import eval_map

class SingleClassAP:
    """
    Note: 
        - Currently only supports no-class specific IoU.
        - Requires the inputs to be in [x1, y1, x2, y2] format.
    """

    def __init__(self, iou_thold: float=0.5):
        super().__init__()
        """
            Args:
                - reduction: can take "mean", "sum", or None
        """
        self.iou_thold = iou_thold
        self.detections = []
        self.annotations = []


    def add_data(self, pred: torch.Tensor, gt: torch.Tensor):
        if len(pred.shape) == 2: # add batch dimension
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
        
        for b in range(pred.shape[0]):
            p = pred[b, :].cpu().numpy()
            self.detections.append([p])
            g = gt[b, :].cpu().numpy()
            self.annotations.append({
                "bboxes": np.asarray(g),
                "labels": np.zeros([g.shape[0]])
            })

    def calculate(self):
        mAP, vals = eval_map(
            self.detections, self.annotations, iou_thr=self.iou_thold)
        self.detections = []
        self.annotations = []
        return mAP
