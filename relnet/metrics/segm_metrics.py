import numpy as np


class SegmentationMetrics:
    
    def __init__(self, n_classes: int, ignore_bg: bool=True):
        self.n_classes = n_classes
        self.ignore_bg = ignore_bg
        self.hist = np.zeros((self.n_classes, self.n_classes))
    
    
    def reset(self):
        self.hist = np.zeros((self.n_classes, self.n_classes))
    
    
    def add(self, preds: np.ndarray, gts: np.ndarray):
        """
        preds: should be prediction matrix with unnormalized logits
        gts: ground truth indices
        """
        # Evaluation Part
        pred_label = np.array(preds, dtype=int)  # shape = [H, W]
        pred_label = pred_label[np.newaxis, ...]  # shape = [1, H, W]
        
        gt_label = np.array(gts, dtype=int)  # shape = [H, W]
        gt_label = gt_label[np.newaxis, ...]  # shape = [1, H, W]
        
        gt_flat = gt_label.flatten()
        pred_flat = pred_label.flatten()
        ints = np.where(gt_flat > 0)[0]
        self.hist += self.fast_hist(gt_flat, pred_flat, self.n_classes)     
    
    
    def calculate(self):
        if self.ignore_bg > 0:
            hist = self.hist[1:, 1:]
        else:
            hist = self.hist

        # overall accuracy
        ova_acc = np.diag(hist).sum() / hist.sum()
        
        # mAcc
        acc = np.diag(hist) / hist.sum(1)
        acc = np.nan_to_num(acc)
        mean_acc = np.nanmean(acc)
    
        # mIoU
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        iou = np.nan_to_num(iou)
        mean_iou = np.nanmean(iou)
        
        # FWIoU
        freq = hist.sum(1) / hist.sum()
        fw_iou = (freq[freq > 0] * iou[freq > 0]).sum()
        
        return ova_acc, mean_acc, mean_iou, fw_iou
        
    
    def fast_hist(self, a, b, n):
        """
        :param a: gt
        :param b: pred
        """
        k = (a >= 0) & (a < n)
        return np.bincount(
            n * a[k].astype(int) + b[k], minlength=n ** 2
        ).reshape(n, n)