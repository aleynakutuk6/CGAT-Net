import os
import argparse
import sys
import torch
import json
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

cwd = os.getcwd()
sys.path.append(f"{cwd}/..")

from relnet.data import CBSCDataset
from relnet.data.preprocessors import CAVTPreprocessor
from relnet.metrics import SegmentationMetrics
from relnet.utils.cfg_utils import parse_configs

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument('-c', '--common-class-set', default="common", type=str, 
    help="Which set of common classes to choose.")
parser.add_argument('-t', '--thickness', default=None, type=int, 
    help="Thickness of the lines drawn for dual-model evaluation")
parser.add_argument('-ov', '--ov-style-preds', action="store_true")
args = parser.parse_args()

model_type = args.model
thickness = args.thickness
ov_style_preds = args.ov_style_preds
common_set = args.common_class_set
partition = "test"

"""
python3 segment_dual_with_skformer.py -t 2 -m cavt-no_T-no_CA-no_PP -d friss -c qd ; python3 segment_dual_with_skformer.py -t 2 -m cavt-no_T-no_CA-no_PP -d cbsc -c qd
"""


if "cavt" in model_type:
    from relnet.models import CAVT
    from relnet.data.preprocessors import CAVTPreprocessor
    segm_config = f"{cwd}/../run_results/{model_type}/training_config.json"
    cfg = parse_configs(whole_config_path=segm_config)
    cfg["preprocessor"]["segmentation"]["thickness"] = thickness
    cfg["model"]["segmentation"]["mmdet_cfg"] = f"{cwd}/../" + cfg["model"]["segmentation"]["mmdet_cfg"]
    cfg["model"]["segmentation"]["weight"] = f"{cwd}/../" + cfg["model"]["segmentation"]["weight"]
    model = CAVT(cfg["model"]["segmentation"], "cuda:0")
    model = model.eval().cuda()
    preprocessor = CAVTPreprocessor(cfg["preprocessor"]["segmentation"])
    
elif model_type == "skformer":
    segm_config = f"{cwd}/../run_results/cavt/training_config.json"
    from model_comparisons.models_classification.Sketchformer.model_fns import (
        load_model, pass_from_model)
    model = load_model()
    preprocessor = None
    

tr_data_cfg = f"{cwd}/../configs/dataset/coco.json"
val_data_cfg = f"{cwd}/../configs/dataset/{args.dataset}.json"

cfg_all = parse_configs(
    whole_config_path=segm_config,
    tr_data_path=tr_data_cfg, 
    val_data_path=val_data_cfg)

cfg = cfg_all["dataset"]

cfg[partition]["extra_filter_file"] = (f"{cwd}/../datasets/mapping_files/"
                                       f"common_classes/{common_set}.txt")
if "cbsc" == args.dataset:
    map_file = f"{cwd}/../datasets/mapping_files/cbsc_to_qd_mappings.json"
elif "friss" == args.dataset:
    map_file = f"{cwd}/../datasets/mapping_files/friss_segmentation_mappings.json"
elif "fscoco" == args.dataset:
    map_file = f"{cwd}/../datasets/mapping_files/fscoco_to_sketch_mappings.json"
else:
    raise ValueError
cfg[partition]["mapping_file"] = map_file

if cfg[partition]["data_dir"][0] != "/":
    cfg[partition]["data_dir"] = f"{cwd}/../" + cfg[partition]["data_dir"]
    
dataset = CBSCDataset(
    partition, cfg, 
    preprocessor=preprocessor, 
    save_dir="jsons/complete")

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

valid_idxs = set()
if model_type != "cavt":
    for data in tqdm(dataloader, desc="Dummy run for valid class generation"):
        gt_labels = data["labels"][0, :].numpy().tolist()
        for gt_label in gt_labels:
            if gt_label >= 0:
                valid_idxs.add(gt_label)
    
    valid_idx_dict, num_valid_cls = {}, len(valid_idxs) + 1
    for v_idx, valid_idx in enumerate(valid_idxs):
        valid_idx_dict[valid_idx] = v_idx + 1
                
    with open(f"segmentation_outs_{args.dataset}_{common_set}.json", "r") as f:
        segm_outs = json.load(f)
    
    segm_preprocessor = CAVTPreprocessor(cfg_all["preprocessor"]["segmentation"])
    metric_fn = SegmentationMetrics(num_valid_cls, ignore_bg=True)
    
else:
    segm_outs = {}


for i, data in enumerate(tqdm(dataloader, desc="Iterating the dataset...")):
    
    if "cavt" in model_type:
        scene_visuals = data["scene_visuals"].cuda()
        scene_strokes = data["vectors"].cuda()
        if scene_strokes[0, 0, -1] < 0: continue
        stroke_areas = data["stroke_areas"].cuda()
        stroke_area_inds = data["stroke_area_inds"].cuda()
        segmentation_sizes = data["segmentation_sizes"].cuda()
        image_index = data["image_id"].unsqueeze(0).item()
        
        with torch.no_grad():
            _, _, ranges = model(
                scene_visuals, 
                scene_strokes, 
                stroke_areas, 
                stroke_area_inds, 
                segmentation_sizes)
        
        segm_outs[str(image_index)] = ranges.squeeze(0).cpu().numpy().tolist()
        
    else:
        abs_stroke3 = data["vectors"].squeeze(0).numpy()
        # print(abs_stroke3)
        if abs_stroke3 is None or abs_stroke3[0, -1] == -1: continue
        divisions = data["divisions"].squeeze(0).numpy().tolist()
        gt_labels = data["labels"].squeeze(0).numpy().tolist()
        image_size = data["img_size"].squeeze(0).numpy().tolist()
        image_index = data["image_id"].unsqueeze(0).item()
        ranges = segm_outs[str(image_index)]
        # print(divisions, ranges)
        
        if common_set == "sketchy":
            changed = 134 in gt_labels or 43 in gt_labels or 34 in gt_labels
        elif common_set == "sky":
            changed = 43 in gt_labels
        else:
            changed = False

        pred_names = pass_from_model(model, abs_stroke3, ranges)
        pred_labels = []
        for pred_name_list in pred_names:
            for pred_name in pred_name_list:
                if pred_name not in dataset.labels_info["label_to_idx"]:
                    continue
                pred_idx = int(dataset.labels_info["label_to_idx"][pred_name])
                if changed and pred_idx == 133: pred_idx = 134
                if changed and pred_idx == 17: pred_idx = 43
                if changed and pred_idx == 123: pred_idx = 34
                if ov_style_preds:
                    if int(pred_idx) not in gt_labels: continue
                    mapped_idx = valid_idx_dict[pred_idx]
                    pred_labels.append(mapped_idx)  
                    break
                else:
                    if pred_idx not in valid_idx_dict: continue
                    mapped_idx = valid_idx_dict[pred_idx]
                    pred_labels.append(mapped_idx)  
                    break

        # generate CLASS MATRICES
        str_starts = [0] + (np.where(abs_stroke3[:, -1] == 1)[0] + 1).tolist()
        str_to_pnt_maps = {}
        for str_idx, str_start in enumerate(str_starts):
            str_to_pnt_maps[str_idx] = str_start
            
        gt_divs = [str_to_pnt_maps[div] for div in divisions]
        pred_divs = [str_to_pnt_maps[div] for div in ranges]
        gt_labels = [valid_idx_dict[cls] for cls in gt_labels]

        gt_mtx = segm_preprocessor.create_class_segmentation_mtx(
            abs_stroke3,
            gt_divs,
            gt_labels,
            *image_size)
            
        pred_mtx = segm_preprocessor.create_class_segmentation_mtx(
            abs_stroke3,
            pred_divs,
            pred_labels,
            *image_size)
        
        metric_fn.add(pred_mtx, gt_mtx)
        
if "cavt" in model_type:
    with open(f"segmentation_outs_{args.dataset}_{common_set}.json", "w") as f:
        json.dump(segm_outs, f)
        
else:
    ova_acc, mean_acc, mean_iou, fw_iou = metric_fn.calculate()
    print("#"*60)
    print(f"--> OVA-acc  : {ova_acc}")
    print(f"--> Mean-acc : {mean_acc}")
    print(f"--> Mean-IoU : {mean_iou}")
    print(f"--> FW-IoU   : {fw_iou}")
    print(f"--> Dataset  : {args.dataset} ({partition} partition)")
    print(f"--> Model    : CAVT + {model_type}")
    print("#"*60)
