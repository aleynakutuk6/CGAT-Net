import os
import copy
import torch
import argparse
import json
import torch.optim as optim

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from relnet.data import COCODataset, CBSCDataset, VGDataset, SingleSketchDataset
from relnet.data.preprocessors import CGATNetPreprocessor, SingleSketchPreprocessor
from relnet.metrics.accuracy import Accuracy
from relnet.models import CGATNet, SingleSketchModel
from relnet.utils.cfg_utils import parse_configs, set_nclasses_in_config

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--experiment-name', required=True, type=str, 
                    help="Default: None. If given, overwrites the experiment name initialization.")
parser.add_argument('-tdc', '--train-data-config', required=True, type=str, 
                    help="Config file path for training dataset.")
parser.add_argument('-vdc', '--val-data-config', default=None, type=str, 
                    help="Config file path for validation dataset. If none, read from training dataset config.")
parser.add_argument('-mc', '--model-config', required=True, type=str, 
                    help="Config file path for model.")
parser.add_argument('-pc', '--pipeline-config', default="configs/pipeline/default.json", type=str, 
                    help="Config file path for training pipeline.")
parser.add_argument('-prc', '--preprocessor-config', default="configs/preprocessor/only_classification.json", type=str, 
                    help="Config file path for training preprocessor.")
parser.add_argument('-op', '--override-params', default=None, type=str, 
                    help='Override params. Example: "attn_maps=0,1,2,3,4 ; learning_rate=1e-4"')
parser.add_argument('--use-lr-scheduler', action="store_true")              
args = parser.parse_args()

print("-"*60)
print()
print("* Reading config files..")

exp_name = args.experiment_name

if os.path.exists(f"run_results/{exp_name}"):
    whole_config = f"run_results/{exp_name}/training_config.json"
    cfg = parse_configs(whole_config_path=whole_config)
    weight_files = [
        os.path.join("run_results", exp_name, file) for file in os.listdir(f"run_results/{exp_name}") if ".pth" in file]
else:
    weight_files = []
    
if len(weight_files) > 0:
    
    max_iter_file, max_iter_val = None, 0
    for wfile in weight_files:
        WEIGHTS = torch.load(wfile)
        iter_val = WEIGHTS["global_step"]
        if iter_val > max_iter_val:
            max_iter_val = iter_val
            max_iter_file = wfile
    
    cfg["model"]["classification"]["model_path"] = max_iter_file

else:
    cfg = parse_configs(
        tr_data_path=args.train_data_config, 
        model_path=args.model_config, 
        pipeline_path=args.pipeline_config,
        preprocessor_path=args.preprocessor_config,
        val_data_path=args.val_data_config,
        override_params=args.override_params)

os.system(f"mkdir -p run_results/{exp_name}")
txt_name = f"run_results/{exp_name}/rel_log.txt"
model_save_name = f"run_results/{exp_name}/best_model.pth"

# Tr & Val Dataloaders
print("* Loading preprocessors, datasets, and dataloaders..")
assert "classification" in cfg["preprocessor"]

PREPROCESSOR_REGISTRY = {
    "coco": CGATNetPreprocessor,
    "vg": CGATNetPreprocessor,
    "cbsc": CGATNetPreprocessor,
    "friss": CGATNetPreprocessor,
    "single_sketch": SingleSketchPreprocessor,
}

DATASET_REGISTRY = {
    "coco": COCODataset,
    "vg": VGDataset,
    "cbsc": CBSCDataset,
    "friss": CBSCDataset,
    "single_sketch": SingleSketchDataset,
}

tr_dataname = cfg["dataset"]["train"]["dataset_name"]

tr_preprocessor = PREPROCESSOR_REGISTRY[tr_dataname](cfg["preprocessor"]["classification"])
training_data = DATASET_REGISTRY[tr_dataname](
    "train", cfg["dataset"], save_dir=f"run_results/{exp_name}", preprocessor=tr_preprocessor)

validation_datas = {}
for cfg_val in cfg["dataset"]["val"]:
    val_dataname = cfg_val["dataset_name"]
    val_preprocessor = PREPROCESSOR_REGISTRY[val_dataname](cfg["preprocessor"]["classification"])
    save_dataname, vd_ctr = val_dataname, 1
    while save_dataname in validation_datas:
        save_dataname = val_dataname + "_" + str(vd_ctr)
        vd_ctr += 1
    
    cfg_to_pass = copy.deepcopy(cfg["dataset"])
    cfg_to_pass["val"] = cfg_val
    validation_datas[save_dataname] = DATASET_REGISTRY[val_dataname](
        "val", cfg_to_pass, save_dir=f"run_results/{exp_name}", preprocessor=val_preprocessor)


bs = cfg["pipeline"]["batch_size"]
n_classes = training_data.num_categories   
trainloader = DataLoader(training_data, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)

valloaders = {}
for k in validation_datas:
    assert n_classes == validation_datas[k].num_categories
    valloaders[k] = DataLoader(validation_datas[k], batch_size=1, shuffle=False, num_workers=4)
    
# Model definition
print("* Loading the model..")
assert "classification" in cfg["model"]

cfg = set_nclasses_in_config(cfg, n_classes)

if tr_dataname == "single_sketch":
    model = SingleSketchModel(cfg["model"]["classification"])
else:
    model = CGATNet(cfg["model"]["classification"])
model = model.cuda()

with open(f"run_results/{exp_name}/training_config.json", "w") as f:
    json.dump(cfg, f)

# Loss & Optimizer
print("* Loading the optimizer, loss_fn, and lr_scheduler..")

loss_function = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=cfg["pipeline"]["learning_rate"])

if args.use_lr_scheduler:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, 
        threshold=0.001, min_lr=min(1e-6, cfg["pipeline"]["learning_rate"]))
else:
    lr_scheduler = None

max_train_steps = cfg["pipeline"]["max_train_steps"]
val_per_step = cfg["pipeline"]["val_per_step"]


if cfg["model"]["classification"]["model_path"] is not None:
    # if pretrained path given, loads the model, optimizer, and other parameters
    print("* Loading pretrained weights...")
    WEIGHTS = torch.load(cfg["model"]["classification"]["model_path"])
    model.load_state_dict(WEIGHTS["model"])
    optimizer.load_state_dict(WEIGHTS["optimizer"])
    global_step = WEIGHTS["global_step"]
    best_val_acc = WEIGHTS["acc"]
    
else:
    global_step = 0
    best_val_acc = {}
    for k in valloaders:
        best_val_acc[k] = 0
    
    for g in optimizer.param_groups:
        g['lr'] = cfg["pipeline"]["learning_rate"]
    
    
f = open(txt_name, "w")
f.write('############################## Training Started ################################ \n')
f.close()

print("* Training the model..")
while global_step < max_train_steps:
    
    # validation run
    model.eval()
    
    for vl_idx, k in enumerate(valloaders):
        acc_fn = Accuracy(top_k=1)
        for i, data in enumerate(tqdm(valloaders[k], desc=f"Evaluating {k}...")):
            sketch_images = data["obj_visuals"].cuda()
            gt_labels = data["labels"].cuda()
        
            if val_dataname != "single_sketch":
                attns = data["attns"].cuda()
                padding_mask = data["padding_mask"].cuda()
                    
            if tr_dataname == "single_sketch":
                c, h, w = sketch_images.shape[-3:]
                with torch.no_grad():
                    outputs = model(sketch_images.view(-1, c, h, w))
                
                # add batch dimension and make prev batch dimension as sequence dimension
                gt_labels = gt_labels.unsqueeze(0)
                outputs = outputs.unsqueeze(0)
            else:
                with torch.no_grad():
                    outputs = model(sketch_images, attns, padding_mask)
            
            acc_fn.add(outputs.cpu().numpy(), gt_labels.cpu().numpy())
            
        avg = acc_fn.calculate()
        if vl_idx == 0 and lr_scheduler is not None:
            lr_scheduler.step(avg)
    
        if avg > best_val_acc[k]:
            best_val_acc[k] = avg
            torch.save({
                "model": model.state_dict(), 
                "acc": best_val_acc, 
                "optimizer": optimizer.state_dict(), 
                "global_step": global_step
            }, model_save_name.replace(".", f"_{k}."))
        
    
        f = open(txt_name, "a")
        f.write('\nValidation of {} --> global_step: {}, acc: {}, best: {} \n\n'.format(k, global_step, avg, best_val_acc[k]))
        f.close()
    
    # training run
    val_step = 0
    model.train()
    
    while val_step < val_per_step:
        
        pbar = tqdm(total=val_per_step)
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            
            sketch_images = data["obj_visuals"].cuda()
            gt_labels = data["labels"].cuda()

            if tr_dataname != "single_sketch":
                attns = data["attns"].cuda()
                padding_mask = data["padding_mask"].cuda()
                outputs, backbone_outs = model(sketch_images, attns, padding_mask)
            
            else:
                c, h, w = sketch_images.shape[-3:]
                outputs = model(sketch_images.view(-1, c, h, w))
                backbone_outs = None
            
            loss = loss_function(outputs.view(-1, n_classes), gt_labels.long().view(-1))
            
            if (backbone_outs is not None and 
                "backbone_loss_weight" in cfg["pipeline"] and
                cfg["pipeline"]["backbone_loss_weight"] > 0):
                loss += cfg["pipeline"]["backbone_loss_weight"] * loss_function(
                    backbone_outs.view(-1, n_classes), 
                    gt_labels.long().view(-1))
            
            loss.backward()
            optimizer.step()
    
            running_loss = loss.item()
            val_step += 1
            global_step += 1
            pbar.update(1)
            pbar.set_description(f"Loss: {round(running_loss, 4)}")

            if i % 100 == 0:
                lr_val = optimizer.param_groups[-1]["lr"]
                f = open(txt_name, "a")
                f.write('Train --> global_step: {}, loss: {} , LR: {}\n'.format(global_step, running_loss, lr_val))
                f.close()
            
            if val_step == val_per_step:
                # break if number of train iterations per val is reached
                break
            
            

f = open(txt_name, "a")
f.write('############################## Finished Training ################################ \n')
f.close()