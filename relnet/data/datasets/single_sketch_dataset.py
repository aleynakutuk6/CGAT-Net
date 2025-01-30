import json
import numpy as np
import os
import torch
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from relnet.utils.sketch_utils import *
from relnet.utils.visualize_utils import *


class SingleSketchDataset(BaseDataset):
    
    def __init__(self, split: str, cfg, save_dir: str=None, labels_info_dir: str=None, preprocessor=None):
        super().__init__(split, cfg, save_dir, preprocessor)
        self.data_dir = cfg[split]["data_dir"]
        self.train_set_list = cfg[split]["train_set_list"]
        self.labels_info, self.img_ids = self.read_labels_info_and_sketch_paths()    
        self.num_categories = len(self.labels_info["idx_to_label"])
        if self.split == "train":
            self.save_labels_info(save_dir)
        
        
    def __len__(self):
        return len(self.img_ids)
    
    
    def read_labels_info_and_sketch_paths(self):
        classes_set, files = set(), []
        
        if "qd" in self.train_set_list:
            qd_paths = os.path.join(self.data_dir, "qd", self.split, "sketches")
            for folder in tqdm(os.listdir(qd_paths), desc="Reading QD directory"):
                if "." in folder: continue
                if not os.path.isdir(os.path.join(qd_paths, folder)): continue
                if (self.labels_info is not None and 
                    folder not in self.labels_info["label_to_idx"]):
                    continue
                
                classes_set.add(folder)
                files_list = sorted(os.listdir(os.path.join(qd_paths, folder)))
                
                if self.split != "train": 
                    files_list = files_list[:min(10, len(files_list))]
                else: 
                    files_list = files_list[:min(500, len(files_list))]
                
                for file in files_list:
                    if ".npy" not in file: continue
                    files.append([folder, os.path.join(qd_paths, folder, file)])
                
        if "friss" in self.train_set_list:
            friss_paths = os.path.join(self.data_dir, "friss", self.split, "sketches")
            for folder in tqdm(os.listdir(friss_paths), desc="Reading FRISS directory"):
                if "." in folder: continue
                if not os.path.isdir(os.path.join(friss_paths, folder)): continue
                if (self.labels_info is not None and 
                    folder not in self.labels_info["label_to_idx"]):
                    continue
                
                classes_set.add(folder)
                files_list = sorted(os.listdir(os.path.join(friss_paths, folder)))
                
                if self.split != "train": 
                    files_list = files_list[:min(10, len(files_list))]
                else: 
                    files_list = files_list[:min(500, len(files_list))]
                
                for file in files_list:
                    if ".npy" not in file: continue
                    files.append([folder, os.path.join(friss_paths, folder, file)])
        
        if self.labels_info is None:
            labels_info = {"idx_to_label": {}, "label_to_idx": {}}
            for c_idx, cls in enumerate(sorted(list(classes_set))):
                labels_info["idx_to_label"][c_idx] = cls
                labels_info["label_to_idx"][cls] = c_idx
        else:
            labels_info = self.labels_info
        
        return labels_info, files
            
         
    def __getitem__(self, idx):
        cls_name, pth = self.img_ids[idx]
        cls_id = self.labels_info["label_to_idx"][cls_name]
        sketch = self.read_and_scale_sketch(pth)
        sketch = self.pad_vector_sketch(sketch)
        sketch = torch.from_numpy(sketch)
        
        return_dict = {
            "vectors": sketch,
            "labels": cls_id
        }
        
        return self.run_preprocessor(return_dict)  