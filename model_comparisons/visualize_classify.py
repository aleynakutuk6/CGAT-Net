import os
import argparse
import sys
import json

cwd = os.getcwd()

sys.path.append(f"{cwd}/..")

from tqdm import tqdm
from torch.utils.data import DataLoader

from relnet.data import CBSCDataset
from relnet.data.preprocessors import CGATNetPreprocessor
from relnet.utils.cfg_utils import parse_configs
from relnet.utils.visualize_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument("-c", "--common-class-type", type=str, default="common")
args = parser.parse_args()

model_type = args.model
common_set = args.common_class_type
assert common_set == "common"

if model_type == "sketchformer" or model_type == "skformer":
    from model_comparisons.models_classification.Sketchformer.model_fns import load_model, pass_from_model
elif model_type == "sketchr2cnn":
    from model_comparisons.models_classification.Sketch_R2CNN.model_fns import load_model, pass_from_model
elif model_type == "mgt":
    from model_comparisons.models_classification.MGT.model_fns import load_model, pass_from_model
else:
    raise ValueError
    
partition = "test"
tr_data_cfg = f"{cwd}/../configs/dataset/coco.json"
val_data_cfg = f"{cwd}/../configs/dataset/{args.dataset}.json"
common_file = f"{cwd}/../datasets/mapping_files/common_classes/{common_set}.txt"

if "cbsc" == args.dataset:
    map_file = f"{cwd}/../datasets/mapping_files/cbsc_to_qd_mappings.json"
elif "friss" == args.dataset:
    map_file = f"{cwd}/../datasets/mapping_files/friss_segmentation_mappings.json"
elif "fscoco" == args.dataset:
    map_file = f"{cwd}/../datasets/mapping_files/fscoco_to_sketch_mappings.json"
else:
    raise ValueError

cfg = parse_configs(
    tr_data_path=tr_data_cfg, 
    val_data_path=val_data_cfg,
)["dataset"]

if common_set == "sky":
    cmap_p = f"{cwd}/../datasets/mapping_files/sky_color_maps.json"
elif common_set == "sketchy":
    cmap_p = f"{cwd}/../datasets/mapping_files/sketchy_color_maps.json"
else:
    cmap_p = f"{cwd}/../datasets/mapping_files/complete_color_maps.json"

if cfg[partition]["data_dir"][0] != "/":
    cfg[partition]["data_dir"] = f"{cwd}/../" + cfg[partition]["data_dir"]
if cfg[partition]["mapping_file"][0] != "/":
    cfg[partition]["mapping_file"] = map_file
if cfg[partition]["extra_filter_file"][0] != "/":
    cfg[partition]["extra_filter_file"] = common_file
    
dataset = CBSCDataset(partition, cfg, save_dir="jsons")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
MODEL = load_model()

valid_idxs = set()
for data in tqdm(dataloader, desc="Dummy run for valid class generation"):
    gt_labels = data["labels"][0, :].numpy().tolist()
    for gt_label in gt_labels:
        if gt_label >= 0:
            valid_idxs.add(gt_label)

image_ids_filter_set = set()
if common_set == "sketchy" and args.dataset == "cbsc":
    image_ids_filter_set = {4, 41, 63, 76, 81, 98, 168, 173, 185, 202, 217, 184}
elif common_set == "common"  and args.dataset == "cbsc":
    image_ids_filter_set = {9, 12, 13, 33, 49, 58, 61, 87, 88, 103, 128, 116}
elif common_set == "sketchy" and args.dataset == "friss":
    image_ids_filter_set = {45, 70, 242, 284, 417, 420, 480, 433, 434, 441, 484, 469, 492}
elif common_set == "common" and args.dataset == "friss":
    image_ids_filter_set = {56, 66, 104, 134, 199, 211, 283, 287, 321, 374, 383, 410}

cmap_p = f"{cwd}/../datasets/colors/cls_"
cmap_p += args.dataset + "_colors.json"   

with open(cmap_p, "r") as f:
    color_data = json.load(f)

pbar = tqdm(dataloader)
for i, data in enumerate(pbar):
    img_id = data["image_id"].squeeze(0).item()
    pbar.set_description(f"ID: {img_id}")
    if len(image_ids_filter_set) > 0 and img_id not in image_ids_filter_set:
        continue
    
    abs_stroke3 = data["vectors"].squeeze(0).numpy()
    if abs_stroke3 is None or abs_stroke3[0, -1] == -1: continue
    
    W, H = data["img_size"].squeeze(0).numpy()
    divisions = data["divisions"].squeeze(0).numpy()
    labels = data["labels"].squeeze(0).numpy().tolist()
    if len(set(labels)) < 3: continue

    # pred_names are 2D matrix with names are sorted from most probable to least
    pred_names = pass_from_model(MODEL, abs_stroke3, divisions)
    pred_sel_names = []
    for pred_name_list in pred_names:
        for pred_name in pred_name_list:
            if pred_name in dataset.labels_info["label_to_idx"]:
                if pred_name == "yoga": pred_name = "person"
                pred_sel_names.append(pred_name)    
                break
                
    str_starts = [0] + (np.where(abs_stroke3[:, -1] == 1)[0] + 1).tolist()
    str_to_pnt_maps = {}
    for str_idx, str_start in enumerate(str_starts):
        str_to_pnt_maps[str_idx] = str_start
    divisions = np.array([str_to_pnt_maps[div] for div in divisions])
    
    curr_color_data = color_data[str(img_id)]
    with open("temp_skformer_colors.json", "w") as f:
        json.dump(curr_color_data, f)
    
    save_dir = f"{cwd}/../vis_outs/sketchformer/{args.dataset}"
    os.system(f"mkdir -p {save_dir}")
    visualize_results(
        abs_stroke3,
        divisions,
        pred_sel_names,
        [W, H],
        "temp_skformer_colors.json",
        is_absolute=True,
        save_path=os.path.join(save_dir, f"{img_id}.jpg")
    )
    
    os.system("rm -rf temp_skformer_colors.json")
