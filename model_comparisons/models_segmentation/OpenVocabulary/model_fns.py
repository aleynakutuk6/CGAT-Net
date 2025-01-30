import os
import copy
import json
import cv2
import torch

from collections import OrderedDict
from PIL import Image
from torchvision.transforms import (
    InterpolationMode, Compose, Resize, ToTensor, Normalize
)

from relnet.utils.visualize_utils import *

import sys
sys.path.append("models_segmentation/OpenVocabulary")

import models
from utils import setup, get_similarity_map, get_segmentation_map

OV_THRESHOLD = 0

class Arguments:
    def __init__(self):
        self.config_file = "models_segmentation/OpenVocabulary/vpt/configs/prompt/cub.yaml"
        self.train_type = ""
        self.output_path = "output"
        self.data_path = None
        self.checkpoint_path = "weights/openvocab/sketch_seg_best_miou.pth"
        self.threshold = OV_THRESHOLD

        
def load_model():
    args = Arguments()
    cfg = setup(args)
    device = "cuda"
    
    Ours, preprocess = models.load(
        "CS-ViT-B/16", device=device, cfg=cfg, zero_shot=False)
    state_dict = torch.load(args.checkpoint_path)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v 
    Ours.load_state_dict(new_state_dict)     
    Ours.eval().cuda()
    
    preprocess = Compose([
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC), 
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711))
    ])
    
    with open("models_segmentation/OpenVocabulary/cgatnet_to_fscoco_mappings.json", "r") as f:
        rev_fscoco_map = json.load(f)
   
    return {
        "model": Ours,
        "cfg": cfg,
        "preprocessor": preprocess,
        "labels_info": rev_fscoco_map
    }


def pass_from_model(info, scene_strokes, labels, data_labels_info, W, H, thickness, to_vis=False):
    Ours = info["model"]
    cfg = info["cfg"]
    preprocessor = info["preprocessor"]
    label_mapping = info["labels_info"]
    device = "cuda"
    
    mdl_label_names = []
    for idx in labels:
        cgatnet_label = data_labels_info["idx_to_label"][idx]
        if cgatnet_label in label_mapping:
            mdl_label_names.append(label_mapping[cgatnet_label])
        else:
            print(f"{cgatnet_label} not in mapping, taking as it is.")
            mdl_label_names.append(cgatnet_label)
            label_mapping[cgatnet_label] = cgatnet_label # adds to dict

    if not to_vis:
        scene_visuals, _ = draw_sketch(
            scene_strokes,
            canvas_size=[W, H],
            margin=0,
            white_bg=True,
            color_fg=False,
            shift=False,
            scale_to=-1,
            is_absolute=True,
            thickness=thickness
        )
    
    else:
        scene_visuals, _ = draw_sketch(
            scene_strokes,
            canvas_size=[800, 800],
            margin=50,
            white_bg=True,
            color_fg=False,
            shift=True,
            scale_to=800,
            is_absolute=True,
            thickness=thickness
        )
        W, H = 800, 800
    
    scene_visuals = scene_visuals.astype(np.uint8)
    new_scene_visuals = np.full((max(H, W), max(H, W), 3), 255, dtype=np.uint8)
    new_scene_visuals[:H, :W, :] = scene_visuals
    scene_visuals = new_scene_visuals
    binary_sketch = np.where(scene_visuals > 0, 255, scene_visuals)
    
    pil_img = Image.fromarray(scene_visuals).convert("RGB")
    sketch_tensor = preprocessor(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        text_features = models.encode_text_with_prompt_ensemble(
            Ours, mdl_label_names, device, no_module=True)
        redundant_features = models.encode_text_with_prompt_ensemble(
            Ours, [""], device, no_module=True) 
        
        sketch_features = Ours.encode_image(
            sketch_tensor,
            layers=12,
            text_features=text_features-redundant_features,
            mode="test")
        sketch_features = sketch_features / sketch_features.norm(dim=1, keepdim=True)
        
    similarity = sketch_features @ (text_features - redundant_features).t()
    patches_similarity = similarity[0, cfg.MODEL.PROMPT.NUM_TOKENS + 1:, :]
    pixel_similarity = get_similarity_map(patches_similarity.unsqueeze(0), (max(W, H), max(W, H)))
    pixel_similarity[pixel_similarity < OV_THRESHOLD] = 0
    pixel_similarity_array = pixel_similarity.cpu().numpy().transpose(2, 0, 1)
    
    pred_mtx = get_segmentation_map(
        pixel_similarity_array,
        binary_sketch,
        labels)[:H, :W]
    
    return pred_mtx
    
    
    