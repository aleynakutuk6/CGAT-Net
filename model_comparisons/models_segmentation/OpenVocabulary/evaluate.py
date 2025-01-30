import sys
sys.path.append("/kuacc/users/akutuk21/hpc_run/Sketch-Graph-Network")

from relnet.utils.cfg_utils import parse_configs
from relnet.utils.visualize_utils import *
from relnet.data import CBSCDataset
from relnet.data.preprocessors import CAVTPreprocessor
from relnet.metrics import SegmentationMetrics

# ------------------------------------------------------------------

import os
import copy
import json
import cv2

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import (
    InterpolationMode, Compose, Resize, ToTensor, Normalize
)

import models

from utils import setup, get_similarity_map, display_segmented_sketch, get_segmentation_map
from vpt.launch import default_argument_parser

# ------------------------------------------------------------------
# Initialize the configs
# ------------------------------------------------------------------

"""
python3 evaluate.py --config-file vpt/configs/prompt/cub.yaml --checkpoint-path checkpoint/sketch_seg_best_miou.pth
"""

args = default_argument_parser().parse_args()
cfg = setup(args)

THRESHOLD = 0
device = "cuda"

preprocess =  Compose([
    Resize((224, 224), interpolation=InterpolationMode.BICUBIC), 
    ToTensor(),
    Normalize(
        (0.48145466, 0.4578275, 0.40821073), 
        (0.26862954, 0.26130258, 0.27577711))
])

BASE_DIR = "/kuacc/users/akutuk21/hpc_run/Sketch-Graph-Network"
whole_cfg = f"{BASE_DIR}/run_results/segmentation/friss_segmentation/training_config.json"
val_data_cfg = f"{BASE_DIR}/configs/dataset/friss.json"
prep_cfg = f"{BASE_DIR}/configs/preprocessor/only_segmentation.json"

relnet_cfg = parse_configs(
    whole_config_path=whole_cfg,
    val_data_path=val_data_cfg,
    preprocessor_path=prep_cfg,
    override_params="color_fg=true",
)

dataset_name = val_data_cfg.replace(".json", "").split("/")[-1]
partition = "test"

if dataset_name == "friss":
    relnet_cfg["dataset"][partition]["data_dir"] = os.path.join(
    BASE_DIR, 
    relnet_cfg["dataset"][partition]["data_dir"])

relnet_cfg["dataset"][partition]["mapping_file"] = os.path.join(
    BASE_DIR, 
    relnet_cfg["dataset"][partition]["mapping_file"])

relnet_cfg["dataset"][partition]["extra_filter_file"] = os.path.join(
    BASE_DIR, 
    relnet_cfg["dataset"][partition]["extra_filter_file"])

preprocessor = CAVTPreprocessor(relnet_cfg["preprocessor"]["segmentation"])
dataset = CBSCDataset(
    partition, relnet_cfg["dataset"], 
    save_dir=f"{BASE_DIR}/model_comparisons/OpenVocabulary")
dataloader = DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=4)


valid_idxs = set()
for data in tqdm(dataloader, desc="Dummy run for valid class generation"):
    gt_labels = data["labels"][0, :].numpy().tolist()
    for gt_label in gt_labels:
        if gt_label >= 0:
            valid_idxs.add(gt_label)
            
valid_idx_dict, num_valid_cls = {}, len(valid_idxs) + 1
for v_idx, valid_idx in enumerate(valid_idxs):
    valid_idx_dict[valid_idx] = v_idx + 1

with open("labels_info.json", "r") as f:
    labels_info = json.load(f)
    
with open("../../datasets/mapping_files/fscoco_to_sketch_mappings.json", "r") as f:
    fscoco_mapping = json.load(f)

rev_fscoco_map = {}
for k in fscoco_mapping:
    v = fscoco_mapping[k]
    if v is None: 
        continue
    if v not in rev_fscoco_map:
        rev_fscoco_map[v] = []
    rev_fscoco_map[v].append(k)

direct_matches, chosen_matches = 0, 0
for k in rev_fscoco_map:
    if len(rev_fscoco_map[k]) == 1:
        rev_fscoco_map[k] = rev_fscoco_map[k][0]
        direct_matches += 1
    elif k in rev_fscoco_map[k]:
        rev_fscoco_map[k] = k
        direct_matches += 1
    else:
        print("Choosing for", k, "from", sorted(rev_fscoco_map[k]))
        rev_fscoco_map[k] = sorted(rev_fscoco_map[k])[0]
        chosen_matches += 1

if "yoga" in rev_fscoco_map:
    rev_fscoco_map["yoga"] = "person"
if "stage" in rev_fscoco_map:
    rev_fscoco_map["stage"] = "platform"
if "fire hydrant" in rev_fscoco_map:
    rev_fscoco_map["fire hydrant"] = "hydrant" 
if "frisbee" in rev_fscoco_map:
    rev_fscoco_map["frisbee"] = "disc" 
if "bicycle" in rev_fscoco_map:
    rev_fscoco_map["bicycle"] = "bike" 
if "bush" in rev_fscoco_map:
    rev_fscoco_map["bush"] = "bush" 
if "palm tree" in rev_fscoco_map:
    rev_fscoco_map["palm tree"] = "palm"
if "billboard" in rev_fscoco_map:
    rev_fscoco_map["billboard"] = "board" 
        
# print(direct_matches, "vs", chosen_matches)

# ------------------------------------------------------------------
# Initialize the Model
# ------------------------------------------------------------------

Ours, preprocess = models.load(
    "CS-ViT-B/16", device=device, cfg=cfg, zero_shot=False)
state_dict = torch.load(args.checkpoint_path)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v 
Ours.load_state_dict(new_state_dict)     
Ours.eval().cuda()
    
# ------------------------------------------------------------------
# Iterate the data
# ------------------------------------------------------------------

segm_fn = SegmentationMetrics(num_valid_cls, ignore_bg=True)

for idx, data in enumerate(tqdm(dataloader)):
    img_id = str(idx)
    
    if data is None or data["vectors"] is None: continue
    elif data["vectors"][0, 0, -1] < 0: continue
    
    scene_strokes = data["vectors"].squeeze(0).cpu().numpy().astype(float)
    W, H = data["img_size"].squeeze(0).cpu().numpy().tolist()

    scene_visuals, _ = draw_sketch(
        scene_strokes,
        canvas_size=[W, H],
        margin=0,
        white_bg=True,
        color_fg=False,
        shift=False,
        scale_to=-1,
        is_absolute=True,
    )

    # W, H = 512, 512
    # scene_visuals, _ = draw_sketch(
    #     scene_strokes,
    #     canvas_size=[512, 512],
    #     margin=50,
    #     white_bg=True,
    #     color_fg=False,
    #     shift=True,
    #     scale_to=412,
    #     is_absolute=True,
    # )
    
    scene_visuals = scene_visuals.astype(np.uint8)
    new_scene_visuals = np.full((max(H, W), max(H, W), 3), 255, dtype=np.uint8)
    new_scene_visuals[:H, :W, :] = scene_visuals
    scene_visuals = new_scene_visuals
    # cv2.imwrite("scene_visual.jpg", scene_visuals)
    binary_sketch = np.where(scene_visuals > 0, 255, scene_visuals)
    
    pil_img = Image.fromarray(scene_visuals).convert("RGB")
    sketch_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    
    labels, label_names = data["labels"][0].numpy().tolist(), []
    for lbl in labels:
        label_name = labels_info["idx_to_label"][str(lbl)]
        if label_name in rev_fscoco_map:
            label_name = rev_fscoco_map[label_name]
        label_names.append(label_name)  
    labels = [valid_idx_dict[cls] for cls in labels]
         
    with torch.no_grad():
        text_features = models.encode_text_with_prompt_ensemble(
            Ours, label_names, device, no_module=True)
        redundant_features = models.encode_text_with_prompt_ensemble(
            Ours, [""], device, no_module=True) 
        
        sketch_features = Ours.encode_image(
            sketch_tensor,
            layers=12,
            text_features=text_features-redundant_features,
            mode=partition)
        sketch_features = sketch_features / sketch_features.norm(dim=1, keepdim=True)
        
    similarity = sketch_features @ (text_features - redundant_features).t()
    patches_similarity = similarity[0, cfg.MODEL.PROMPT.NUM_TOKENS + 1:, :]
    pixel_similarity = get_similarity_map(patches_similarity.unsqueeze(0), (max(W, H), max(W, H)))
    pixel_similarity[pixel_similarity < THRESHOLD] = 0
    pixel_similarity_array = pixel_similarity.cpu().numpy().transpose(2, 0, 1)

    """                                                                         
    os.system(f"mkdir -p outputs")
    
    classes_dir = os.path.join("outputs", 'CLASS_PRED')
    if not os.path.isdir(classes_dir):
        os.mkdir(classes_dir)
    
    drawings_dir = os.path.join("outputs", 'DRAWING_PRED')
    if not os.path.isdir(drawings_dir):
        os.mkdir(drawings_dir)
        
    colors = plt.get_cmap("tab20").colors
    if len(colors) < len(label_names):
        colors = colors + colors
    label_names_colors = colors[:len(label_names)]
    
    display_segmented_sketch(
        pixel_similarity_array,
        binary_sketch,
        label_names,
        label_names_colors,
        labels,
        img_id,
        save_path="outputs")
    """
    
    pred_mtx = get_segmentation_map(
        pixel_similarity_array,
        binary_sketch,
        labels)[:H, :W]
    
    divisions = data["divisions"]
    gt_labels = data["labels"]
    
    str_starts = [0] + (np.where(scene_strokes[:, -1] == 1)[0] + 1).tolist()
    str_to_pnt_maps = {}
    for str_idx, str_start in enumerate(str_starts):
        str_to_pnt_maps[str_idx] = str_start
    str_to_pnt_maps[len(str_starts)] = scene_strokes.shape[-2]
    
    divisions = [str_to_pnt_maps[div] for div in divisions.squeeze(0).numpy().tolist()]

    gt_mtx = preprocessor.create_class_segmentation_mtx(
        scene_strokes,
        divisions,
        labels,
        W, H)
    
    segm_fn.add(pred_mtx, gt_mtx)

ova_acc, mean_acc, mean_iou, fw_iou = segm_fn.calculate()
        
print("#"*60)
print(f"--> OVA-acc  : {ova_acc}")
print(f"--> Mean-acc : {mean_acc}")
print(f"--> Mean-IoU : {mean_iou}")
print(f"--> FW-IoU   : {fw_iou}")
print(f"--> Dataset  : {dataset_name} ({partition} partition)")
print(f"--> Model    : Open Vocabulary")
print("#"*60)
