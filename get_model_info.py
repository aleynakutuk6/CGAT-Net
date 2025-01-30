from fvcore.nn import flop_count, FlopCountAnalysis, parameter_count_table, parameter_count

import os
import copy
import torch
import time
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader

from relnet.data import CBSCDataset, COCODataset
from relnet.data.preprocessors import CGATNetPreprocessor, CAVTPreprocessor
from relnet.models import CGATNet, CAVT, SingleSketchModel
from relnet.metrics import Accuracy, AllOrNothing, SequenceIoU, SegmentationMetrics
from relnet.utils.cfg_utils import parse_configs

# --------------------------------
# Parsing arguments
# --------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-vdc', '--val-data-config', default="configs/dataset/friss.json", 
                    type=str, help="Config file path for the dataset.")
parser.add_argument('-td', '--train-dir', default="run_results/cgatnet", type=str, 
    help="Directory of the training experiment, to get the model data and config.")
parser.add_argument('-t', '--thickness', default=2, type=int, 
    help="Thickness of the lines drawn for dual-model evaluation")
parser.add_argument('-k', '--topk', default=1, type=int)
parser.add_argument('-c', '--common-class-set', default="common", type=str, 
    help="Which set of common classes to choose.")
parser.add_argument('-ov', '--ov-style-preds', action="store_true")
parser.add_argument('-foq', '--filter-out-qd', action="store_true")
args = parser.parse_args()

thickness = args.thickness
ov_style_preds = args.ov_style_preds
filter_out_qd = args.filter_out_qd

# --------------------------------
# Parsing the config file
# --------------------------------

print()
print("-"*60)
print()
print("* Reading config file..")

map_file = "datasets/mapping_files/friss_segmentation_mappings.json"

cfg = parse_configs(
    val_data_path=args.val_data_config,
    whole_config_path=os.path.join(args.train_dir, "training_config.json"),
    override_params=(f"mapping_file={map_file} ; "
                     f"extra_filter_file=datasets/mapping_files/"
                     f"common_classes/{args.common_class_set}.txt")
)

exp_name = args.val_data_config.split("/")[-1].replace(".json", "")

# --------------------------------
# Loading Preprocessors
# --------------------------------

print("* Loading preprocessors..")

partition = "test"
cls_preprocessor = CGATNetPreprocessor(cfg["preprocessor"]["classification"])

# --------------------------------
# Loading Dataset and Dataloaders
# --------------------------------
    
print("* Loading dataset and dataloader..")

DATASET_REGISTRY = {
    "cbsc": CBSCDataset,
    "friss": CBSCDataset,
}

dataset_name = cfg["dataset"][partition]["dataset_name"]

test_data = DATASET_REGISTRY[dataset_name](
    partition, cfg["dataset"], 
    save_dir=args.train_dir, 
    preprocessor=cls_preprocessor)

testloader = DataLoader(
    test_data, batch_size=1, shuffle=False, num_workers=4)

# --------------------------------
# Loading Models
# --------------------------------

print("* Loading model options..")
cls_model_path = os.path.join(args.train_dir, "best_model_friss.pth")
    
# --------------------------------
# Testing Loop
# --------------------------------

model = CGATNet(cfg["model"]["classification"])
model.load_state_dict(torch.load(cls_model_path)["model"], strict=True)
model = model.cuda().eval()


performed_model_analysis, total_iters = False, 0

start = 0
for i, data in enumerate(tqdm(testloader)):
    
    if i == 1:
        start = time.time()
    
    sketch_images = data["obj_visuals"].cuda()
    attns = data["attns"].cuda()
    padding_mask = data["padding_mask"].cuda()
    
    if not performed_model_analysis:
        print(sketch_images.shape, attns.shape)
        # flops = flop_count(model, (sketch_images, attns, padding_mask))
        flops = FlopCountAnalysis(model, (sketch_images, attns, padding_mask)).total() / 1e9
        print(f"Total FLOPs: {flops} GFLOPs")
        param_nums_dicts = parameter_count(model)
        total_params_count = 0
        for k, v in param_nums_dicts.items():
            print(k, "-->", v)
            total_params_count += v
        print(f"Total param count:", total_params_count)
        performed_model_analysis = True
        
    with torch.no_grad():
        outputs = model(sketch_images, attns, padding_mask)
    
    if i > 0:
        total_iters += 1
    
end = time.time()

print("Total processing time:", (end - start) / total_iters, "seconds")