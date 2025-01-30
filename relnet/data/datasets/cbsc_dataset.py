import os
import json
import torch
import random
import numpy as np

from collections import defaultdict
from .base_dataset import BaseSceneDataset

class CBSCDataset(BaseSceneDataset):

    def __init__(self, split, cfg, save_dir: str=None, preprocessor=None):
        super().__init__(split, cfg, save_dir, preprocessor)
        self.data_dir = cfg[split]["data_dir"]
        self.img_ids = sorted(self.getImgIds())

    def __len__(self):
        return len(self.img_ids)
    
    def getImgIds(self):
        
        file_list = []
        for file_ in os.listdir(self.data_dir):
            if ".json" in file_:
                file_list.append(os.path.join(self.data_dir, file_))
                
        return file_list   
    

    def __getitem__(self, index):
        
        with open(self.img_ids[index], "rb") as f:
            data = json.load(f)
        
        # img_id = self.img_ids[index].split("/")[-1].replace(".json", "")
        w, h = data["img_w"], data["img_h"]
        scene_strokes = np.asarray(data["scene_strokes"])
        object_divisions = data["object_divisions"]
        gt_classes = data["gt_class_names"]
        
        sketch_vectors, gt_labels, divisions = self.filter_objects_from_scene(
            gt_classes, scene_strokes, object_divisions)

        if len(gt_labels) > 0:
            sketch_vectors, gt_labels, divisions, padding_mask = self.pad_items(
                sketch_vectors, gt_labels, divisions)
        else:
            max_obj_cnt = max(self.max_obj_cnt, 1)
            sketch_vectors = torch.full((max_obj_cnt, 3), -1, dtype=int)
            gt_labels = torch.full((max_obj_cnt, ), -1, dtype=int)
            divisions = torch.full((max_obj_cnt+1, ), -1, dtype=int)
            padding_mask = torch.ones(max_obj_cnt, 1, dtype=int)

        img_size = torch.LongTensor([w, h])
  
        return_dict = {
            "vectors": sketch_vectors,
            "labels": gt_labels,
            "divisions": divisions,
            "padding_mask": padding_mask,
            "img_size": img_size,
            "image_id": index,
        }
        
        return self.run_preprocessor(return_dict)