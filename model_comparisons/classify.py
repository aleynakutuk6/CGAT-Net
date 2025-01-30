import os
import argparse
import sys

cwd = os.getcwd()

sys.path.append(f"{cwd}/..")

from tqdm import tqdm
from torch.utils.data import DataLoader

from relnet.data import CBSCDataset
from relnet.data.preprocessors import CGATNetPreprocessor
from relnet.metrics import Accuracy
from relnet.utils.cfg_utils import parse_configs
from relnet.utils.visualize_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument("-k", "--topk", type=int, default=1)
args = parser.parse_args()

model_type = args.model

if model_type == "sketchformer":
    from model_comparisons.models_classification.Sketchformer.model_fns import load_model, pass_from_model
elif model_type == "sketchr2cnn":
    from model_comparisons.models_classification.Sketch_R2CNN.model_fns import load_model, pass_from_model
elif model_type == "mgt":
    from model_comparisons.models_classification.MGT.model_fns import load_model, pass_from_model
else:
    raise ValueError
    
tr_data_cfg = f"{cwd}/../configs/dataset/coco.json"
val_data_cfg = f"{cwd}/../configs/dataset/{args.dataset}.json"

cfg = parse_configs(tr_data_path=tr_data_cfg, val_data_path=val_data_cfg)["dataset"]
partition = "test"

if cfg[partition]["data_dir"][0] != "/":
    cfg[partition]["data_dir"] = f"{cwd}/../" + cfg[partition]["data_dir"]
if cfg[partition]["mapping_file"][0] != "/":
    cfg[partition]["mapping_file"] = f"{cwd}/../" + cfg[partition]["mapping_file"]
if cfg[partition]["extra_filter_file"][0] != "/":
    cfg[partition]["extra_filter_file"] = f"{cwd}/../" + cfg[partition]["extra_filter_file"]
    
dataset = CBSCDataset(partition, cfg, save_dir="jsons")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
MODEL = load_model()

valid_idxs = set()
for data in tqdm(dataloader, desc="Dummy run for valid class generation"):
    gt_labels = data["labels"][0, :].numpy().tolist()
    for gt_label in gt_labels:
        if gt_label >= 0:
            valid_idxs.add(gt_label)

acc_fn = Accuracy(top_k=args.topk, valid_ids=valid_idxs)

for i, data in enumerate(tqdm(dataloader, desc="Iterating the dataset...")):
    # if i == 10: break
    abs_stroke3 = data["vectors"].squeeze(0).numpy()
    divisions = data["divisions"].squeeze(0).numpy()
    labels = data["labels"].squeeze(0).numpy()
    
    if abs_stroke3 is None or abs_stroke3[0, -1] == -1: continue
    
    # labels_name = [dataset.labels_info["idx_to_label"][str(lbl)] for lbl in labels]
    
    # pred_names are 2D matrix with names are sorted from most probable to least
    pred_names = pass_from_model(MODEL, abs_stroke3, divisions)
    pred_idxs = []
    for pred_name_list in pred_names:
        pred_row = []
        for pred_name in pred_name_list:
            if pred_name in dataset.labels_info["label_to_idx"]:
                pred_row.append(int(dataset.labels_info["label_to_idx"][pred_name]))      
        pred_idxs.append(pred_row)
    
    acc_fn.add_as_idxs(pred_idxs, labels)

acc = acc_fn.calculate()
print("#"*60)
print(f"--> SCORE   : {acc}")
print(f"--> Dataset : {args.dataset} ({partition} partition)")
print(f"--> Model   : {model_type}")
print("#"*60)