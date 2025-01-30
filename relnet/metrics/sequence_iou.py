import torch
import numpy as np
from tqdm import tqdm


class SequenceIoU:
    
    def __init__(self, filter_out_qd: bool=False):
        self.pred = []
        self.gt = []
        self.gt_class = []
        self.filter_out_qd = filter_out_qd
        self.non_qd_classes = {
            284, 419, 378, 356, 55, 397, 277, 53, 371, 4, 
            359, 267, 468, 135, 430, 448, 179, 157, 162, 
            262, 128, 331, 116, 265, 129, 42, 184, 156, 88, 
            304, 431, 389, 341, 26, 428, 167, 296, 344, 39, 
            259, 328, 140, 207, 155, 225, 362, 2, 20, 293, 
            466, 168, 241, 444, 457, 202, 423, 250, 163, 43, 
            282, 107, 424, 62, 363, 24, 58, 461, 322, 283, 
            435, 445, 158, 405, 415, 63, 388, 330, 350, 191, 
            348, 164, 351, 278, 337, 185, 354, 460, 432, 
            392, 151, 247, 286, 36, 75, 68, 299, 44, 398, 
            463, 452, 416, 476, 78, 219, 99, 417, 159, 120, 
            69, 395, 141, 214, 465, 84, 108, 369, 471, 294, 
            303, 208, 443, 98, 217, 37, 66, 180, 384, 276, 
            473, 321, 352, 209, 373, 132, 82, 261, 333, 307, 
            270, 244, 470, 31, 479}


    def add(self, pred: torch.Tensor, gt: torch.Tensor, gt_class: torch.Tensor=None):
        if len(gt.shape) == 1:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
            gt_class = gt_class.unsqueeze(0)
        
        for b in range(gt.shape[0]):
            self.pred.append(pred[b, :].cpu().numpy().astype(int).tolist())
            self.gt.append(gt[b, :].cpu().numpy().astype(int).tolist())
            if gt_class is not None:
                self.gt_class.append(gt_class[b, :].cpu().numpy().astype(int).tolist())
    
    def calculate(self):
        total_score, total_num_cnt = 0, 0
        for b in tqdm(range(len(self.gt))):
            score, num_cnt = 0, 0
            for i in range(len(self.gt[b]) - 1):
                if self.filter_out_qd and self.gt_class[b][i-1] not in self.non_qd_classes:
                    continue
                else:
                    num_cnt += 1
                max_iou_for_tuple = 0
                for j in range(len(self.pred[b]) - 1):
                    intersection = min(self.gt[b][i+1], self.pred[b][j+1]) - max(self.gt[b][i], self.pred[b][j])
                    if intersection >= 0.0:
                        gt_area = self.gt[b][i+1] - self.gt[b][i]
                        pred_area = self.pred[b][j+1] - self.pred[b][j]
                        union = gt_area + pred_area - intersection
                        iou = intersection / union
                        if max_iou_for_tuple < iou:
                            max_iou_for_tuple = iou
                score += max_iou_for_tuple
            total_score += score / max(1, num_cnt)
            if num_cnt > 0:
                total_num_cnt += 1
        
        total_score = total_score / max(1, total_num_cnt)
        if total_num_cnt < 1:
            print("No class is found to calculate the AoN.")
        self.pred, self.gt, self.gt_class = [], [], []
        return total_score
        