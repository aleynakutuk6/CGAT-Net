import numpy as np


class Accuracy:
    
    def __init__(self, pass_idx: int=-1, top_k: int=1, valid_ids: list=None):
        self.pass_idx = pass_idx
        self.top_k = top_k
        self.valid_ids = None if valid_ids is None else set(valid_ids)
        self.preds = []
        self.gts = []
    
    
    def reset(self):
        self.preds = []
        self.gts = []
    
    
    def add(self, preds: np.ndarray, gts: np.ndarray):
        """
        preds: should be prediction matrix with unnormalized logits
        gts: ground truth indices
        """
        emb_dim = preds.shape[-1]
        flat_preds = preds.reshape(-1, emb_dim)
        flat_gts = gts.reshape(-1)
        
        for b in range(flat_gts.shape[0]):
            if flat_gts[b] == self.pass_idx:
                continue
            self.gts.append(flat_gts[b])
            
            sels = []
            for idx in np.argsort(flat_preds[b, :])[::-1]:
                if self.valid_ids is None or idx in self.valid_ids:
                    sels.append(idx)
                if len(sels) == self.top_k:
                    break
            
            self.preds.append(sels)
            
    def add_as_idxs(self, preds: list, gts: np.ndarray):
        """
        preds: should be prediction list of list with indices at cells
        gts: ground truth indices
        """
        flat_gts = gts.reshape(-1)
        
        for b in range(flat_gts.shape[0]):
            if flat_gts[b] == self.pass_idx:
                continue
            self.gts.append(flat_gts[b])
            
            sels = []
            for idx in preds[b]:
                if self.valid_ids is None or idx in self.valid_ids:
                    sels.append(idx)
                if len(sels) == self.top_k:
                    break
            
            self.preds.append(sels)
        
    
    def calculate(self):
        acc, total = 0, 0
        for ps, gt in zip(self.preds, self.gts):
            if gt in ps:
                acc += 1
            total += 1
        
        return round(acc / max(1, total), 4)
            