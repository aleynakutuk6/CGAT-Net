import os
import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from relnet.data import CBSCDataset, COCODataset
from relnet.data.preprocessors import CGATNetPreprocessor, CAVTPreprocessor
from relnet.models import CGATNet, CAVT, SingleSketchModel
from relnet.metrics import Accuracy, AllOrNothing, SequenceIoU, SegmentationMetrics
from relnet.utils.cfg_utils import parse_configs
from relnet.utils.visualize_utils import visualize_results, draw_sketch

# --------------------------------
# Parsing arguments
# --------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-vdc', '--val-data-config', required=True, type=str, 
    help="Config file path for the dataset.")
parser.add_argument('-td', '--train-dir', default="model_comparisons/jsons", type=str, 
    help="Directory of the training experiment, to get the model data and config.")
parser.add_argument('-t', '--thickness', default=None, type=int, 
    help="Thickness of the lines drawn for dual-model evaluation")
parser.add_argument('-c', '--common-class-set', default="common", type=str, 
    help="Which set of common classes to choose.")
parser.add_argument('-ov', '--ov-style-preds', action="store_true")
parser.add_argument('-gt', '--vis-ground-truth', action="store_true")
parser.add_argument('-inst', '--instance-level', action="store_true")
args = parser.parse_args()

thickness = args.thickness
ov_style_preds = args.ov_style_preds
common_set = args.common_class_set
vis_gt = args.vis_ground_truth
vis_inst_level = args.instance_level

# --------------------------------
# Parsing the config file
# --------------------------------

print()
print("-"*60)
print()
print("* Reading config file..")

partition = "test"
common_file = f"datasets/mapping_files/common_classes/{common_set}.txt"

if "cbsc" in args.val_data_config:
    map_file = "datasets/mapping_files/cbsc_to_qd_mappings.json"
elif "friss" in args.val_data_config:
    map_file = "datasets/mapping_files/friss_segmentation_mappings.json"
elif "fscoco" in args.val_data_config:
    map_file = "datasets/mapping_files/fscoco_to_sketch_mappings.json"
else:
    raise ValueError
    
if vis_gt:
    whole_cfg = None
else:
    whole_cfg = os.path.join(args.train_dir, "training_config.json")

cfg = parse_configs(
    val_data_path=args.val_data_config,
    whole_config_path=whole_cfg,
    override_params=f"mapping_file={map_file} ; extra_filter_file={common_file}"
)

# --------------------------------
# Loading Preprocessors
# --------------------------------

print("* Loading preprocessors..")

if "classification" in cfg["preprocessor"]:
    cls_preprocessor = CGATNetPreprocessor(cfg["preprocessor"]["classification"])
else:
    cls_preprocessor = None

if "segmentation" in cfg["preprocessor"]:
    if thickness is not None:
        cfg["preprocessor"]["segmentation"]["thickness"] = thickness
    segm_preprocessor = CAVTPreprocessor(cfg["preprocessor"]["segmentation"])
else:
    segm_preprocessor = None

multi_preprocessors = cls_preprocessor is not None and segm_preprocessor is not None

if multi_preprocessors or segm_preprocessor is not None:
    pass_preprocessor = segm_preprocessor
elif cls_preprocessor is not None:
    pass_preprocessor = cls_preprocessor
else:
    pass_preprocessor = None
# --------------------------------
# Loading Dataset and Dataloaders
# --------------------------------
    
print("* Loading dataset and dataloader..")

dataset_name = cfg["dataset"][partition]["dataset_name"]

test_data = CBSCDataset(
    partition, cfg["dataset"], 
    save_dir=args.train_dir, 
    preprocessor=pass_preprocessor)

testloader = DataLoader(
    test_data, batch_size=1, shuffle=False, num_workers=4)

# --------------------------------
# Loading Models
# --------------------------------

print("* Loading model options..")

if "classification" in cfg["model"]:
    cfg["model"]["classification"]["model_path"] = None
    model = CGATNet(cfg["model"]["classification"])
    model.load_state_dict(
        torch.load(os.path.join(args.train_dir, f"best_model_{dataset_name}.pth"))["model"], 
        strict=True)
    model = model.cuda().eval()
    # Gets the valid indices for the evaluated dataset
    valid_idxs = set()
    for data in tqdm(testloader, desc="Dummy run for valid class generation"):
        gt_labels = data["labels"][0, :].numpy().tolist()
        for gt_label in gt_labels:
            if gt_label >= 0:
                valid_idxs.add(gt_label)
    num_valid_cls = len(valid_idxs) + 1

else:
    model = None
    valid_idxs = set()
    cls_model_paths = None
    num_valid_cls = 1
    

if "segmentation" in cfg["model"]:
    segm_model = CAVT(cfg["model"]["segmentation"], "cuda:0")
    segm_model = segm_model.eval().cuda()
else:
    segm_model = None  

# --------------------------------
# Testing Loop
# --------------------------------

if common_set == "sketchy" and dataset_name == "cbsc":
    image_ids_filter_set = {4, 41, 63, 76, 81, 98, 168, 173, 185, 202, 217, 184}
elif common_set == "common"  and dataset_name == "cbsc":
    image_ids_filter_set = {9, 12, 13, 33, 49, 58, 61, 87, 88, 103, 128, 116}
elif common_set == "sketchy" and dataset_name == "friss":
    image_ids_filter_set = {45, 70, 242, 284, 417, 420, 480, 433, 434, 441, 484, 469, 492}
elif common_set == "common" and dataset_name == "friss":
    image_ids_filter_set = {56, 66, 104, 134, 199, 211, 283, 287, 321, 374, 383, 410}
else:
    image_ids_filter_set = set()

cmap_p = "datasets/colors/"
if common_set == "sketchy":
    cmap_p += "segm_"
else:
    cmap_p += "cls_"
cmap_p += dataset_name + "_colors.json"
    
with open(cmap_p, "r") as f:
    color_data = json.load(f)
    
pbar = tqdm(testloader)
for data in pbar:
    image_id = data["image_id"].numpy().tolist()[0]
    pbar.set_description(f"ID: {image_id}")
    if len(image_ids_filter_set) > 0 and image_id not in image_ids_filter_set:
        continue
    
    gt_labels = data["labels"].squeeze(0).numpy().tolist()
    if len(set(gt_labels)) < 3: continue
    
    divisions = data["divisions"].cuda()
    img_size = data["img_size"]
    sketch_vector = data["vectors"].cuda()
    if sketch_vector[0, 0, -1] < 0: continue
    
    if not vis_gt: 
        if multi_preprocessors:
            scene_visuals = data["scene_visuals"].cuda()
            stroke_areas = data["stroke_areas"].cuda()
            stroke_area_inds = data["stroke_area_inds"].cuda()
            segmentation_sizes = data["segmentation_sizes"].cuda()
            
            save_folder = f"vis_outs/dual_network/{dataset_name}_{common_set}"
            if ov_style_preds: save_folder += "_withOV"
            
            with torch.no_grad():
                scores, boxes, ranges = segm_model(
                    scene_visuals, 
                    sketch_vector, 
                    stroke_areas, 
                    stroke_area_inds, 
                    segmentation_sizes)
                
                # pass from classification preprocessor
                sketch_images, attns = cls_preprocessor(
                    sketch_vector.cpu(), 
                    ranges.long().cpu(), 
                    img_size)
                sketch_images = sketch_images.cuda()
                attns = attns.cuda()
                padding_mask = torch.ones(
                    (1, sketch_images.shape[1], 1), 
                    dtype=torch.long).cuda()
                divisions = ranges 
        else:
            sketch_images = data["obj_visuals"].cuda()
            attns = data["attns"].cuda()
            padding_mask = data["padding_mask"].cuda()
            save_folder = f"vis_outs/cgatnet/{dataset_name}"
            
        with torch.no_grad():
                outputs = model(sketch_images, attns, padding_mask)
        
        pred_labels = outputs.squeeze(0)
        pred_labels = pred_labels.argsort(dim=1, descending=True)
        pred_labels = pred_labels.cpu().numpy()
        final_labels = []
        for s in range(pred_labels.shape[0]):
            for s2 in range(pred_labels.shape[1]):
                if ov_style_preds:
                    if pred_labels[s, s2] in gt_labels:
                        final_labels.append(pred_labels[s, s2])
                        break
                elif pred_labels[s, s2] in valid_idxs:
                    final_labels.append(pred_labels[s, s2])
                    break
    else:
        final_labels = gt_labels
        save_folder = f"vis_outs/ground_truth/{dataset_name}_{common_set}"
    
    W, H = img_size.squeeze(0).numpy().tolist()
    sketch_vector = sketch_vector.squeeze(0).cpu().numpy()
    
    pred_labels = np.asarray(final_labels)
    pred_label_names = []
    for idx in pred_labels:
        pred_name = test_data.labels_info["idx_to_label"][str(idx)]
        if pred_name == "yoga": pred_name = "person"
        pred_label_names.append(pred_name)

    str_starts = [0] + (np.where(sketch_vector[:, -1] == 1)[0] + 1).tolist()
    str_to_pnt_maps = {}
    for str_idx, str_start in enumerate(str_starts):
        str_to_pnt_maps[str_idx] = str_start
    
    divisions = divisions.squeeze(0).cpu().numpy()
    divisions = np.array([str_to_pnt_maps[div] for div in divisions])
    
    os.system(f"mkdir -p {save_folder}")
    
    if vis_inst_level:
        draw_sketch(
            sketch_vector,
            divisions,
            class_ids=None,
            canvas_size=[800, 800],
            margin=50,
            is_absolute=True,
            white_bg=True,
            color_fg=True,
            shift=True,
            scale_to=800,
            save_path=os.path.join(save_folder, f"{image_id}_instlevel.jpg"),
            class_names=pred_label_names,
            thickness=2)
        
    else:
        curr_color_data = color_data[str(image_id)]
        with open(f"temp_{common_set}_{dataset_name}_colors.json", "w") as f:
            json.dump(curr_color_data, f)  
        
        visualize_results(
            sketch_vector,
            divisions,
            pred_label_names,
            [W, H],
            f"temp_{common_set}_{dataset_name}_colors.json",
            is_absolute=True,
            save_path=os.path.join(save_folder, f"{image_id}.jpg"))
    
    
        os.system(f"rm -rf temp_{common_set}_{dataset_name}_colors.json")
