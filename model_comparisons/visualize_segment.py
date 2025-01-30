import os
import argparse
import sys
import json

cwd = os.getcwd()

sys.path.append(f"{cwd}/..")

from tqdm import tqdm
from torch.utils.data import DataLoader

from relnet.data import CBSCDataset
from relnet.data.preprocessors import CAVTPreprocessor
from relnet.metrics import SegmentationMetrics
from relnet.utils.cfg_utils import parse_configs
from relnet.utils.visualize_utils import *

# ---------------------------------------------------------------------------------
# PARSE ARGUMENTS
# ---------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument("-c", "--common-class-type", type=str, required=True)
parser.add_argument("-t", "--thickness", type=int, default=2)
args = parser.parse_args()

model_type = args.model
common_class_type = args.common_class_type
thickness = args.thickness

if model_type == "openvocab":
    from model_comparisons.models_segmentation.OpenVocabulary.model_fns import load_model
    from model_comparisons.models_segmentation.OpenVocabulary.model_fns import pass_from_model
elif model_type == "ldp":
    if "sky" in common_class_type:
        from model_comparisons.models_segmentation.LDP.model_fns import load_model_sky as load_model 
        from model_comparisons.models_segmentation.LDP.model_fns import pass_from_model
    elif "sketchy" in common_class_type:
        from model_comparisons.models_segmentation.LDP.model_fns import load_model_sketchy as load_model 
        from model_comparisons.models_segmentation.LDP.model_fns import pass_from_model
else:
    raise ValueError
    
tr_data_cfg = f"{cwd}/../configs/dataset/coco.json"
val_data_cfg = f"{cwd}/../configs/dataset/{args.dataset}.json"
prep_cfg = f"{cwd}/../configs/preprocessor/only_segmentation.json"
common_txt = f"{cwd}/../datasets/mapping_files/common_classes/{common_class_type}.txt"

cfg = parse_configs(
    tr_data_path=tr_data_cfg, 
    val_data_path=val_data_cfg,
    preprocessor_path=prep_cfg,
    override_params=f"color_fg=true; extra_filter_file={common_txt}")

partition = "test"
dataset_name = args.dataset
cfg["preprocessor"]["segmentation"]["thickness"] = thickness

# ---------------------------------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------------------------------

if cfg["dataset"][partition]["data_dir"][0] != "/":
    cfg["dataset"][partition]["data_dir"] = f"{cwd}/../" + cfg["dataset"][partition]["data_dir"]

if "cbsc" == dataset_name:
    map_file = f"{cwd}/../datasets/mapping_files/cbsc_to_qd_mappings.json"
elif "friss" == dataset_name:
    map_file = f"{cwd}/../datasets/mapping_files/friss_segmentation_mappings.json"
elif "fscoco" == dataset_name:
    map_file = f"{cwd}/../datasets/mapping_files/fscoco_to_sketch_mappings.json"
else:
    raise ValueError   
cfg["dataset"][partition]["mapping_file"] = map_file

preprocessor = CAVTPreprocessor(cfg["preprocessor"]["segmentation"])
dataset = CBSCDataset(partition, cfg["dataset"], save_dir="jsons/complete")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
MODEL = load_model()

# ---------------------------------------------------------------------------------
# SET VALID INDICES
# ---------------------------------------------------------------------------------

valid_idxs = set()
for data in tqdm(dataloader, desc="Dummy run for valid class generation"):
    gt_labels = data["labels"][0, :].numpy().tolist()
    for gt_label in gt_labels:
        if gt_label >= 0:
            valid_idxs.add(gt_label)

num_valid_cls = len(valid_idxs) + 1
valid_idx_dict = {"label_to_idx": {}, "idx_to_label": {}} 
for v_idx, valid_idx in enumerate(valid_idxs):
    valid_name = dataset.labels_info["idx_to_label"][str(valid_idx)]
    valid_idx_dict["label_to_idx"][valid_name] = v_idx + 1
    valid_idx_dict["idx_to_label"][v_idx + 1] = valid_name
    
if common_class_type == "sky":
    cmap_p = "../datasets/mapping_files/sky_color_maps.json"
elif common_class_type == "sketchy":
    cmap_p = "../datasets/mapping_files/sketchy_color_maps.json"
else:
    cmap_p = "../datasets/mapping_files/complete_color_maps.json"
    
cmap_p = "../datasets/colors/"
if common_class_type == "sketchy":
    cmap_p += "segm_"
else:
    cmap_p += "cls_"
cmap_p += dataset_name + "_colors.json"
        
# ---------------------------------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------------------------------

image_ids_filter_set = set()
if common_class_type == "sketchy" and args.dataset == "cbsc":
    image_ids_filter_set = {4, 41, 63, 76, 81, 98, 168, 173, 185, 202, 217, 184}
elif common_class_type == "common"  and args.dataset == "cbsc":
    image_ids_filter_set = {9, 12, 13, 33, 49, 58, 61, 87, 88, 103, 128, 116}
elif common_class_type == "sketchy" and args.dataset == "friss":
    image_ids_filter_set = {45, 70, 242, 284, 417, 420, 480, 433, 434, 441, 484, 469, 492}
elif common_class_type == "common" and args.dataset == "friss":
    image_ids_filter_set = {56, 66, 104, 134, 199, 211, 283, 287, 321, 373, 383, 410}

cmap_p = f"{cwd}/../datasets/colors/segm_"
cmap_p += args.dataset + "_colors.json"

with open(cmap_p, "r") as f:
    color_class_map_overall = json.load(f)

pbar = tqdm(dataloader)
for idx, data in enumerate(pbar):
    # if idx > 5: break
    img_id = data["image_id"].squeeze(0).item()
    pbar.set_description(f"ID: {img_id}")
    if len(image_ids_filter_set) > 0 and img_id not in image_ids_filter_set:
        continue
    W, H = data["img_size"].squeeze(0).cpu().numpy().tolist()
    scene_strokes = data["vectors"].squeeze(0).cpu().numpy().astype(float)
    if scene_strokes is None or scene_strokes[0, -1] < 0: 
        print("None for idx:", img_id)
        continue
    assert scene_strokes[-1, -1] == 1
    
    divisions = data["divisions"].squeeze(0).cpu().numpy().astype(int)
    orig_labels = data["labels"].squeeze(0).cpu().numpy().astype(int)
    if len(set(orig_labels)) < 3: continue
    
    labels = []
    for idx in orig_labels:
        label_name = dataset.labels_info["idx_to_label"][str(idx)]
        new_label = valid_idx_dict["label_to_idx"][label_name]
        labels.append(new_label)

    pred_mtx = pass_from_model(
        MODEL, 
        copy.deepcopy(scene_strokes), copy.deepcopy(labels), 
        valid_idx_dict, W, H, thickness, to_vis=True) 
    
    
    H, W = pred_mtx.shape[:2]
    vis_arr = np.full((H, W, 3), 255)
    valid_class_set = set()
    for h in range(H):
        for w in range(W):
            pred_val = pred_mtx[h, w]
            if pred_val != 0:
                pred_name = valid_idx_dict["idx_to_label"][pred_val]
                if pred_name == "yoga": pred_name = "person"
                valid_class_set.add(pred_name)
                color = color_class_map_overall[str(img_id)][pred_name]
                vis_arr[h, w, 0] = color[0]
                vis_arr[h, w, 1] = color[1]
                vis_arr[h, w, 2] = color[2]
    
    # print(img_id, valid_class_set)

    vis_arr = vis_arr.astype(np.uint8)
    text_size, _ = cv2.getTextSize("0", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    
    for c_idx, class_name in enumerate(valid_class_set):
        ow = 40
        oh = (H - 15) - 30 * c_idx
        vis_arr = cv2.circle(
            vis_arr, 
            (ow - 20, oh - text_size[1] // 2), 
            5, color_class_map_overall[str(img_id)][class_name], -1)
        vis_arr = cv2.putText(
            vis_arr, class_name, (ow, oh), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 0], 2, cv2.LINE_AA)
    
    save_dir = f"{cwd}/../vis_outs/{model_type}/{dataset_name}_{common_class_type}"
    os.system(f"mkdir -p {save_dir}")
    cv2.imwrite(os.path.join(save_dir, f"{img_id}.jpg"), vis_arr)

os.system("rm -rf CUB")
