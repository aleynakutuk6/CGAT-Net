import os
import argparse
import sys

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
    elif "sketchy" in common_class_type:
        from model_comparisons.models_segmentation.LDP.model_fns import load_model_sketchy as load_model 
    from model_comparisons.models_segmentation.LDP.model_fns import pass_from_model
else:
    raise ValueError
    
tr_data_cfg = f"{cwd}/../configs/dataset/coco.json"
val_data_cfg = f"{cwd}/../configs/dataset/{args.dataset}.json"
prep_cfg = f"{cwd}/../configs/preprocessor/only_segmentation.json"

cfg = parse_configs(
    tr_data_path=tr_data_cfg, 
    val_data_path=val_data_cfg,
    preprocessor_path=prep_cfg,
    override_params=f"color_fg=true; extra_filter_file={cwd}/../datasets/mapping_files/common_classes/{common_class_type}.txt")

partition = "test"
dataset_name = args.dataset
cfg["preprocessor"]["segmentation"]["thickness"] = thickness

# ---------------------------------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------------------------------

if cfg["dataset"][partition]["data_dir"][0] != "/":
    cfg["dataset"][partition]["data_dir"] = f"{cwd}/../{cfg['dataset'][partition]['data_dir']}"
if cfg["dataset"][partition]["mapping_file"][0] != "/":
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

# ---------------------------------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------------------------------

metric_fn = SegmentationMetrics(num_valid_cls, ignore_bg=True)
for idx, data in enumerate(tqdm(dataloader)):
    # if idx > 5: break
    img_id = str(idx)
    
    W, H = data["img_size"].squeeze(0).cpu().numpy().tolist()
    scene_strokes = data["vectors"].squeeze(0).cpu().numpy().astype(float)
    if scene_strokes is None or scene_strokes[0, -1] < 0: continue
    assert scene_strokes[-1, -1] == 1
    
    divisions = data["divisions"].squeeze(0).cpu().numpy().astype(int)
    orig_labels = data["labels"].squeeze(0).cpu().numpy().astype(int)
    
    labels = []
    for idx in orig_labels:
        label_name = dataset.labels_info["idx_to_label"][str(idx)]
        new_label = valid_idx_dict["label_to_idx"][label_name]
        labels.append(new_label)

    pred_mtx = pass_from_model(
        MODEL, 
        copy.deepcopy(scene_strokes), copy.deepcopy(labels), 
        valid_idx_dict, W, H, thickness)
    
    str_starts = [0] + (np.where(scene_strokes[:, -1] == 1)[0] + 1).tolist()
    str_to_pnt_maps = {}
    for str_idx, str_start in enumerate(str_starts):
        str_to_pnt_maps[str_idx] = str_start
    # str_to_pnt_maps[len(str_starts)] = scene_strokes.shape[-2]
    pnt_divisions = [str_to_pnt_maps[div] for div in divisions]

    gt_mtx = preprocessor.create_class_segmentation_mtx(
        scene_strokes,
        pnt_divisions,
        labels,
        W, H)
    
    metric_fn.add(pred_mtx, gt_mtx)
    
# ---------------------------------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------------------------------

os.system("rm -rf CUB")
    
ova_acc, mean_acc, mean_iou, fw_iou = metric_fn.calculate()

print("#"*60)
print(f"--> OVA-acc  : {ova_acc}")
print(f"--> Mean-acc : {mean_acc}")
print(f"--> Mean-IoU : {mean_iou}")
print(f"--> FW-IoU   : {fw_iou}")
print(f"--> Dataset  : {dataset_name} ({partition} partition)")
print(f"--> Model    : Open Vocabulary")
print("#"*60)