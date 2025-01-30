import os
import json
import torch
import torchvision.transforms as T
import random
import numpy as np
import math

from PIL import Image
from relnet.utils.sketch_utils import *
from relnet.utils.visualize_utils import *
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    
    def __init__(self, split, cfg, save_dir: str=None, preprocessor=None):
        super().__init__()
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.dataset_name = cfg[split]["dataset_name"]
        self.max_point_cnt = cfg[split]["max_point_cnt"] if "max_point_cnt" in cfg[split] else 10000
        self.labels_info = self.read_labels_info(save_dir)
        self.preprocessor = preprocessor
  
    def __getitem__(self, idx):
        raise NotImplementedError
        
    
    def read_labels_info(self, save_dir: str=None):
        if self.split == "train":
            # in train, labels are generated from scratch
            labels_info = None
        else:
            # in validation and test, already existing labels info is read
            assert os.path.exists(os.path.join(save_dir, "labels_info.json"))
            with open(os.path.join(save_dir, "labels_info.json"), "r") as f:
                labels_info = json.load(f)
                
            self.num_categories = len(labels_info["idx_to_label"])

        return labels_info
    
    
    def save_labels_info(self, save_dir: str=None):
        assert self.split == "train"
        if save_dir is not None:
            with open(os.path.join(save_dir, "labels_info.json"), "w") as f:
                json.dump(self.labels_info, f)
                

    def read_and_scale_sketch(self, npy_path, bbox=None):
        
        sketch = read_npy(npy_path)

        # shift sketch to top-left
        abs_sketch = relative_to_absolute(sketch)
        xmin, ymin, xmax, ymax = get_absolute_bounds(abs_sketch)
        s_w = xmax - xmin
        s_h = ymax - ymin
        
        # shift sketch to bbox top-left
        abs_sketch[:, 0] -= xmin
        abs_sketch[:, 1] -= ymin
        
        if bbox is not None:
            # get obj bbox information
            o_xmin, o_ymin, o_xmax, o_ymax = bbox
            o_h = (o_ymax - o_ymin)
            o_w = (o_xmax - o_xmin)
            
            # re-scale sketch without losing orig object ratio
            r_h = o_h / max(1, s_h)
            r_w = o_w / max(1, s_w)
            
            # scale sketch coords according to bbox w & h
            abs_sketch[:, 0] *= r_w
            abs_sketch[:, 1] *= r_h
            
            # shift sketch coords according to bbox start
            abs_sketch[:, 0] += o_xmin
            abs_sketch[:, 1] += o_ymin
        
        return abs_sketch.astype(int).astype(float)
    
    
    def read_and_scale_sketch_alternative(self, npy_path, bbox=None):
        
        sketch = read_npy(npy_path)

        # shift sketch to top-left
        abs_sketch = relative_to_absolute(sketch)
        xmin, ymin, xmax, ymax = get_absolute_bounds(abs_sketch)
        s_w = xmax - xmin
        s_h = ymax - ymin
        
        # shift sketch to bbox top-left
        abs_sketch[:, 0] -= xmin
        abs_sketch[:, 1] -= ymin
        
        s_cx = (xmax - xmin) // 2
        s_cy = (ymax - ymin) // 2
        
        if bbox is not None:
            # get obj bbox information
            o_xmin, o_ymin, o_xmax, o_ymax = bbox
            o_h = (o_ymax - o_ymin)
            o_w = (o_xmax - o_xmin)
            o_cx = (o_xmax + o_xmin) // 2
            o_cy = (o_ymax + o_ymin) // 2
            
            sft_x = o_cx - s_cx
            sft_y = o_cy - s_cy
            
            if s_h / max(1, s_w) > o_h / max(1, o_w):
                r = o_h / max(1, s_h)
            else:
                r = o_w / max(1, s_w)
                
            # scale sketch coords according to bbox w & h
            abs_sketch[:, 0] *= r
            abs_sketch[:, 1] *= r
            
            # shift sketch coords according to bbox start
            abs_sketch[:, 0] += s_cx
            abs_sketch[:, 1] += s_cy
        
        return abs_sketch.astype(int).astype(float)
    
    
    def pad_vector_sketch(self, sketch: np.ndarray):
        sk_len = sketch.shape[0]
        
        if self.max_point_cnt > 0:
            if sk_len < self.max_point_cnt:
                diff = self.max_point_cnt - sk_len
                pad_pnts = np.full((diff, 3), -1)
                sketch = np.concatenate([sketch, pad_pnts], axis=0)
            else:
                sketch = sketch[:self.max_point_cnt, :]
                sketch[-1, -1] = 1
        
        return sketch
    
    def run_preprocessor(self, return_dict: dict):
        
        if self.preprocessor is not None:
            if self.preprocessor.task == "classification":
                return_dict["obj_visuals"] = self.preprocessor(return_dict["vectors"])
            else:
                raise ValueError
        
        return return_dict
        
        
        
class BaseSceneDataset(BaseDataset):
    
    def __init__(self, split, cfg, save_dir=None, preprocessor=None):
        super().__init__(split, cfg, save_dir=save_dir, preprocessor=preprocessor)
        self.max_obj_cnt = cfg[split]["max_obj_cnt"]
        
        self.mapping_dict, labels_set = self.read_mapping_dict(cfg[split]["mapping_file"])
        if self.labels_info is None:
            assert split == "train"
            # initialized in the BaseDataset
            self.labels_info = self.generate_labels_info(labels_set)   
            self.save_labels_info(save_dir)
            
        if "extra_filter_file" in cfg[split]:
            extra_filter_file = cfg[split]["extra_filter_file"]
        else:
            extra_filter_file = None
            
        self.extra_filter_classes = self.read_extra_filter_classes(extra_filter_file)
    
    
    def read_extra_filter_classes(self, extra_filter_file: str=None):
        if extra_filter_file is None:
            filter_classes = []
            for k in self.mapping_dict:
                if self.mapping_dict[k] is not None:
                    filter_classes.extend(self.mapping_dict[k])
            return set(filter_classes)
        
        with open(extra_filter_file, "r") as f:
            lines = f.readlines()
        classes = set([cls_name.replace("\n", "").strip() for cls_name in lines if len(cls_name) > 2])
        
        return classes
    
    
    def read_mapping_dict(self, mapping_file):
        if self.dataset_name == "coco":
            key = "coco_to_sketch"
        elif self.dataset_name == "vg":
            key = "vg_to_sketch"
        elif self.dataset_name == "cbsc":
            key = "cbsc_to_qd"
        elif self.dataset_name == "friss":
            key = "friss_to_qd"
        elif self.dataset_name == "fscoco":
            key = "fscoco_to_sketch"
            
        mapping_dict = json.load(open(mapping_file, 'r'))[key]
        labels_set = set()
        for class_name in mapping_dict:
            if mapping_dict[class_name] is not None:
                if type(mapping_dict[class_name]) == str:
                    mapping_dict[class_name] = [mapping_dict[class_name]]
                for mapped_cls in mapping_dict[class_name]:
                    labels_set.add(mapped_cls)
        
        return mapping_dict, labels_set
    
    
    def generate_labels_info(self, labels_set):
        labels_info = {"idx_to_label": {}, "label_to_idx": {}}
        for i, val in enumerate(sorted(list(labels_set))):
            labels_info["idx_to_label"][i] = val
            labels_info["label_to_idx"][val] = i            
        
        self.num_categories = len(labels_info["idx_to_label"])      
        return labels_info
    

    def pad_items(self, sketch_vectors: np.ndarray, gt_labels: list, divisions: list): 
        
        sketch_vectors = self.pad_vector_sketch(np.asarray(sketch_vectors))
        diff_size = max(0, self.max_obj_cnt - len(gt_labels))

        sketch_vectors = torch.Tensor(sketch_vectors)
        gt_labels = torch.LongTensor(gt_labels)
        divisions = torch.LongTensor(divisions)
        
        # generation of padding mask
        labels_length = gt_labels.shape[0]
        
        if self.max_obj_cnt > 0:
            if diff_size > 0:
                # padding for gt_labels
                gt_labels = torch.cat([gt_labels, torch.full((diff_size,), -1, dtype=int)], dim=0)
                # padding for divisions
                divisions = torch.cat([divisions, torch.full((diff_size,), -1, dtype=int)], dim=0)
                
            elif diff_size < 0:       
                # padding for gt_labels
                gt_labels = gt_labels[:self.max_obj_cnt]
                # padding for divisions
                divisions = divisions[:self.max_obj_cnt+1]
        
            padding_mask = torch.ones([self.max_obj_cnt, 1], dtype=int)
            if labels_length < self.max_obj_cnt:
                padding_mask[labels_length:, 0] = 0
        else:
            padding_mask = torch.ones([labels_length, 1], dtype=int)

        if sketch_vectors[-1, -1] != 1:
            sketch_vectors[-1, -1] = 1
        
        return sketch_vectors, gt_labels, divisions, padding_mask

    
    def filter_objects_from_scene(self, gt_classes, scene_strokes, object_divisions):
        
        stroke_start_points = [0] + (np.where(scene_strokes[..., -1] == 1)[0] + 1).tolist()
        abs_scene = relative_to_absolute(scene_strokes)
        
        sketch_vectors, gt_labels, gt_divisions = [], [], [0]
        for idx, cls_name in enumerate(gt_classes):
            start_id = object_divisions[idx]
            end_id = object_divisions[idx+1]
            stroke_cnt = end_id - start_id
            
            start_point = stroke_start_points[start_id]
            end_point = stroke_start_points[end_id]

            if cls_name in self.mapping_dict and self.mapping_dict[cls_name] is not None:
                if self.split != "train":
                    mapped_cls = self.mapping_dict[cls_name][0]
                else:
                    mapped_cls = random.choice(self.mapping_dict[cls_name])
            else:
                mapped_cls = None
                    
            # a subset of mapping_dict keys
            if (mapped_cls is not None and mapped_cls in self.extra_filter_classes): 
                obj_strokes = abs_scene[start_point:end_point].tolist()
                sketch_vectors.extend(obj_strokes)
                gt_labels.append(self.labels_info["label_to_idx"][mapped_cls]) 
                gt_divisions.append(gt_divisions[-1] + stroke_cnt)
            
            start_point = end_point
        
        return sketch_vectors, gt_labels, gt_divisions
    
    
    def run_preprocessor(self, return_dict: dict):
        
        if self.preprocessor is not None:
            if self.preprocessor.task == "classification":
                sketch_visuals, attns = self.preprocessor(
                    return_dict["vectors"], 
                    return_dict["divisions"], 
                    return_dict["img_size"])
                return_dict.update({
                    "obj_visuals": sketch_visuals,
                    "attns": attns
                })
            elif self.preprocessor.task == "segmentation":
                sketch_visuals, boxes, stroke_areas, stroke_area_inds, new_sizes = self.preprocessor(
                    return_dict["vectors"], 
                    return_dict["divisions"], 
                    return_dict["img_size"])
                return_dict.update({
                    "scene_visuals": sketch_visuals,
                    "boxes": boxes,
                    "stroke_areas": stroke_areas,
                    "stroke_area_inds": stroke_area_inds,
                    "segmentation_sizes": new_sizes
                })
            
            else:
                raise ValueError
        
        return return_dict
    

class SyntheticBaseSceneDataset(BaseSceneDataset):
    
    def __init__(self, split, cfg, save_dir=None, preprocessor=None):
        super().__init__(split, cfg, save_dir=save_dir, preprocessor=preprocessor)
        self.prioritize_sknet = cfg[split]["prioritize_sknet"]
        self.random_class_select = cfg[split]["random_class_select"] if "random_class_select" in cfg[split] else False
        self.qd_prob = cfg[split]["qd_prob"]
        self.sknet_prob = cfg[split]["sknet_prob"]
        self.friss_prob = cfg[split]["friss_prob"]
        self.ratios_path = cfg[split]["ratios_path"]
        self.max_obj_per_class = cfg[split]["max_obj_per_class"] if "max_obj_per_class" in cfg[split] else -1
        self.sketch_catalog = self.read_sketch_objects_catalog()
        
    def read_sketch_objects_catalog(self):
        
        sketch_catalog = {}
        for data_name in os.listdir(self.ratios_path):
            if "." in data_name or "__" in data_name:
                continue
                
            # if "sknet" in data_name: continue
            
            folder_path = os.path.join(self.ratios_path, data_name, self.split)
            if not os.path.exists(folder_path):
                continue
                
            for cls_name in os.listdir(os.path.join(folder_path, "sketches")):
                mapped_cls_name = cls_name.lower().replace("  ", " ")
                if mapped_cls_name not in sketch_catalog:
                    sketch_catalog[mapped_cls_name] = [cls_name, [data_name]]
                else:
                    sketch_catalog[mapped_cls_name][1].append(data_name) 
        
        return sketch_catalog
    
    
    def get_sketch_dataset(self, pos_datasets):
        
        if self.split == "train":
            if len(pos_datasets) == 1:
                sel_dataset = pos_datasets[0]
            else:
                if self.prioritize_sknet:
                    if "sknet" in pos_datasets:
                        sel_dataset = "sknet"
                    else:
                        sel_dataset = np.random.choice(
                            ["qd", "friss"], 
                            p=[self.qd_prob, self.friss_prob])
                        while sel_dataset not in pos_datasets:
                            sel_dataset = np.random.choice(
                                ["qd", "friss"], 
                                p=[self.qd_prob, self.friss_prob])
                else:
                    
                    if self.random_class_select:
                        sel_dataset = np.random.choice(
                            ["qd", "sknet", "friss"], 
                            p=[self.qd_prob, self.sknet_prob, self.friss_prob])
                        while sel_dataset not in pos_datasets:
                            sel_dataset = np.random.choice(
                                ["qd", "sknet", "friss"], 
                                p=[self.qd_prob, self.sknet_prob, self.friss_prob])
                    else:
                        
                        if "qd" in pos_datasets:
                            sel_dataset = "qd"
                        elif "friss" in pos_datasets:
                            sel_dataset = "friss"
                        else:
                            sel_dataset = "sknet"
                        
        else:
            if len(pos_datasets) == 1:
                sel_dataset = pos_datasets[0]
            else:
                if self.prioritize_sknet:
                    if "sknet" in pos_datasets:
                        sel_dataset = "sknet"
                    elif "qd" in pos_datasets:
                        sel_dataset = "qd"
                    else:
                        sel_dataset = "friss"
                else:
                    if "qd" in pos_datasets:
                        sel_dataset = "qd"
                    elif "friss" in pos_datasets:
                        sel_dataset = "friss"
                    else:
                        sel_dataset = "sknet"
                    
        return sel_dataset
    
    
    def get_closest_sketch_obj_path(self, bbox, sel_dataset, orig_cls_name, k=1):
        
        ratio_path = os.path.join(self.ratios_path, sel_dataset, self.split, "wh_ratios", orig_cls_name + ".json")
        ratio_info = json.load(open(ratio_path, 'r'))
        xmin, ymin, xmax, ymax = bbox
        obj_ratio = ((xmax - xmin) / (ymax - ymin)).item()
        diff = np.abs(np.asarray(list(ratio_info.values())) - obj_ratio)
        if self.split == "train":
            top_idxs = np.argsort(diff)[:k]
            rand_id = random.randint(0, len(top_idxs)-1)
            sel_id = int(top_idxs[rand_id])
        else:
            sel_id = diff.argmin()
            
        sel_filename = list(ratio_info.keys())[sel_id]
        sel_file = sel_filename + ".npy"

        return sel_file
    
    def get_sketch_mappings(self, gt_classes, gt_bboxes):
        
        sketches, gt_labels, divisions = [], [], [0]
        for idx, gt_cls in enumerate(gt_classes):
            cls_name = self.object_idx_to_name[gt_cls].lower().replace("  ", " ")
            if cls_name in self.mapping_dict:
                mapped_classes = self.mapping_dict[cls_name]
                # print("Mapped Classes for", cls_name, "-->", mapped_classes)
                
                if self.extra_filter_classes is not None and mapped_classes is not None:
                    mapped_classes = [cls for cls in mapped_classes if cls in self.extra_filter_classes]
                    if len(mapped_classes) == 0:
                        mapped_classes = None
                        
                # print("Final mapped classes for", cls_name, "-->", mapped_classes)

                if mapped_classes is not None:
                    sel_cls = random.choice(mapped_classes)

                    if sel_cls in self.sketch_catalog:
                        orig_cls_name, pos_datasets = self.sketch_catalog[sel_cls]
                        sel_dataset = self.get_sketch_dataset(pos_datasets)
                        bbox = gt_bboxes[idx]
                        sel_file = self.get_closest_sketch_obj_path(
                            bbox, sel_dataset, orig_cls_name)
                        npy_path = os.path.join(
                            self.ratios_path, sel_dataset, self.split, 
                            "sketches", orig_cls_name, sel_file)
                        sketch = self.read_and_scale_sketch(npy_path, bbox)
                        num_strokes = len(np.where(sketch[:, -1] == 1)[0])
                        
                        sketches.extend(sketch)
                        divisions.append(divisions[-1] + num_strokes)
                        
                        orig_cls_name = orig_cls_name.lower()
                        gt_labels.append(self.labels_info["label_to_idx"][orig_cls_name])
        
        return sketches, gt_labels, divisions