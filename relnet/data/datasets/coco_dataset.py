import os
import json
import torch
import random
import numpy as np

from collections import defaultdict
from .base_dataset import SyntheticBaseSceneDataset
from pycocotools.coco import COCO

class COCODataset(SyntheticBaseSceneDataset):

    def __init__(self, split, cfg, save_dir: str=None, preprocessor=None):
        super().__init__(split, cfg, save_dir, preprocessor)
        self.inst_ann_file = cfg[split]["instances_file"]
        self.stuff_ann_file = cfg[split]["stuff_file"]
        self.coco_stuff = COCO(self.stuff_ann_file)
        self.coco_inst = COCO(self.inst_ann_file)
        self.img_ids = self.coco_inst.getImgIds()
        self.image_id_to_size = self.get_img_info()
        self.inst_anns, self.stuff_anns = self.load_annotations()
        self.image_id_to_objects, self.object_idx_to_name, self.object_name_to_idx = self.load_objects_info()
        
        print(self.object_idx_to_name)

        
    def __len__(self):
        return len(self.img_ids)
    
    
    def get_img_info(self):
        image_id_to_size = {}
        if 'images' in self.coco_stuff.dataset:
            for image_data in self.coco_stuff.dataset['images']:
                image_id = image_data['id']
                filename = image_data['file_name']
                width = image_data['width']
                height = image_data['height']
                image_id_to_size[image_id] = (width, height)
        
        return image_id_to_size
    
    
    def load_objects_info(self):
        
        image_id_to_objects = defaultdict(list)
        object_idx_to_name, object_name_to_idx = {}, {}
        
        for inst_ann in self.inst_anns:
            category_id = inst_ann["category_id"]
            bbox = inst_ann["bbox"]
            image_id_to_objects[inst_ann['image_id']].append({"bbox": bbox, "category_id": category_id})
            category_name = self.coco_inst.loadCats([category_id])[0]["name"]
            object_idx_to_name[category_id] = category_name
            object_name_to_idx[category_name] = category_id
        
        for stuff_ann in self.stuff_anns:
            category_id = stuff_ann["category_id"]
            bbox = stuff_ann["bbox"]
            image_id_to_objects[stuff_ann['image_id']].append({"bbox": bbox, "category_id": category_id})
            category_name = self.coco_stuff.loadCats([category_id])[0]["name"]
            object_idx_to_name[category_id] = category_name
            object_name_to_idx[category_name] = category_id
        
        return image_id_to_objects, object_idx_to_name, object_name_to_idx
    
    
    def load_annotations(self):
        
        stuff_anns = self.coco_stuff.getAnnIds(imgIds=self.img_ids)
        stuff_anns = self.coco_stuff.loadAnns(stuff_anns)
        
        inst_anns = self.coco_inst.getAnnIds(imgIds=self.img_ids)
        inst_anns = self.coco_inst.loadAnns(inst_anns)
        
        return inst_anns, stuff_anns        
    
    
    def filter_largest_max_objs(self, img_id):
        # store all boxes of the same category along with their area info
        gt_class_data, gt_class_sizes = {}, {}
        for obj in self.image_id_to_objects[img_id]:
            obj_cat = obj['category_id']
            if obj_cat not in gt_class_data:
                gt_class_data[obj_cat] = []
                gt_class_sizes[obj_cat] = []
            
            xmin, ymin, obj_w, obj_h = obj['bbox']
            xmax = xmin + obj_w
            ymax = ymin + obj_h
            
            gt_class_sizes[obj_cat].append(obj_w * obj_h)
            gt_class_data[obj_cat].append([xmin, ymin, xmax, ymax])
            
        # sort each box index per category descending order w.r.t. their areas
        max_cat_len = -1
        for obj_cat in gt_class_sizes:
            gt_class_sizes[obj_cat] = np.argsort(gt_class_sizes[obj_cat])[::-1].tolist()
            if len(gt_class_sizes[obj_cat]) > self.max_obj_per_class:
                gt_class_sizes[obj_cat] = gt_class_sizes[obj_cat][:self.max_obj_per_class]
            max_cat_len = max(max_cat_len, len(gt_class_sizes[obj_cat]))
        max_cat_len += 1
        
        obj_cats_list, cat_idx = list(gt_class_data.keys()), 0
        order_idx, gt_classes, gt_bboxes = 0, [], []
        while order_idx < max_cat_len and len(gt_classes) < self.max_obj_cnt:
            # get the object category
            obj_cat = obj_cats_list[cat_idx % len(obj_cats_list)]
            
            if order_idx < len(gt_class_sizes[obj_cat]): 
                # get the box data index from the sorted index list of areas
                sel_data_idx = gt_class_sizes[obj_cat][order_idx]
                # get the box data from data dictionary
                box_data = gt_class_data[obj_cat][sel_data_idx]
                gt_bboxes.append(box_data)
                gt_classes.append(obj_cat)
            
            cat_idx += 1
            if cat_idx % len(obj_cats_list) == 0:
                order_idx += 1
                
        gt_classes = np.asarray(gt_classes)
        gt_bboxes = np.asarray(gt_bboxes)
        
        return gt_classes, gt_bboxes
            

    def __getitem__(self, index):
        
        img_id = self.img_ids[index]
        w, h = self.image_id_to_size[img_id]

        gt_classes, gt_bboxes = self.filter_largest_max_objs(img_id)
        sketch_vectors, gt_labels, divisions = self.get_sketch_mappings(gt_classes, gt_bboxes)
        
        if len(gt_labels) > 0:
            sketch_vectors, gt_labels, divisions, padding_mask = self.pad_items(
                sketch_vectors, gt_labels, divisions)
        else:
            
            if self.split == "train":
                return self.__getitem__(random.randint(0, self.__len__() - 1))
            else:
                print("[WARN] Inside an empty scene in coco synthetic generation...")
                max_obj_cnt = max(self.max_obj_cnt, 1)
                max_pnt_cnt = max(100, self.max_point_cnt)
                sketch_vectors = torch.full((max_pnt_cnt, 3), -1, dtype=int)
                gt_labels = torch.full((max_obj_cnt, ), -1, dtype=int)
                divisions = torch.full((max_obj_cnt+1, ), -1, dtype=int)
                padding_mask = torch.ones(max_obj_cnt, 1, dtype=int)

        return_dict = {
            "vectors": sketch_vectors,
            "labels": gt_labels,
            "divisions": divisions,
            "padding_mask": padding_mask,
            "img_size": torch.LongTensor([w, h]),
        }
        
        return self.run_preprocessor(return_dict)    
    