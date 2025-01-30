import os
import copy
import torch
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
parser.add_argument('-vdc', '--val-data-config', required=True, type=str, 
    help="Config file path for the dataset.")
parser.add_argument('-td', '--train-dir', default=None, type=str, 
    help="Directory of the training experiment, to get the model data and config.")
parser.add_argument('-t', '--thickness', default=None, type=int, 
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

if "cbsc" in args.val_data_config:
    map_file = "datasets/mapping_files/cbsc_to_qd_mappings.json"
elif "friss" in args.val_data_config:
    map_file = "datasets/mapping_files/friss_segmentation_mappings.json"
elif "fscoco" in args.val_data_config:
    map_file = "datasets/mapping_files/fscoco_to_sketch_mappings.json"
else:
    raise ValueError

cfg = parse_configs(
    val_data_path=args.val_data_config,
    whole_config_path=os.path.join(args.train_dir, "training_config.json"),
    override_params=(f"mapping_file={map_file} ; "
                     f"extra_filter_file=datasets/mapping_files/"
                     f"common_classes/{args.common_class_set}.txt")
)

exp_name = args.val_data_config.split("/")[-1].replace(".json", "")
# os.system(f"mkdir -p demo/{exp_name}")
# txt_name = f"demo/{exp_name}/test_log.txt"

# --------------------------------
# Loading Preprocessors
# --------------------------------

print("* Loading preprocessors..")

partition = "test"

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
    raise ValueError("At least one preprocessor should be active!")
    
# --------------------------------
# Loading Dataset and Dataloaders
# --------------------------------
    
print("* Loading dataset and dataloader..")

DATASET_REGISTRY = {
    "coco": COCODataset,
    # "vg": VGDataset,
    "cbsc": CBSCDataset,
    "friss": CBSCDataset,
    "fscoco": CBSCDataset,
}

dataset_name = cfg["dataset"][partition]["dataset_name"]

test_data = DATASET_REGISTRY[dataset_name](
    partition, cfg["dataset"], 
    save_dir=args.train_dir, 
    preprocessor=pass_preprocessor)

testloader = DataLoader(
    test_data, batch_size=1, shuffle=False, num_workers=4)

# --------------------------------
# Loading Models
# --------------------------------

print("* Loading model options..")

if_single_model = cfg["dataset"]["train"]["dataset_name"] == "single_sketch"
count_each_object_in_scene_as_separate_scenes = False

if "classification" in cfg["model"]:
    
    cfg["model"]["classification"]["model_path"] = None
    
    if cfg["model"]["classification"]["model_path"] is None:
        cls_model_paths = []
        for pth in os.listdir(args.train_dir):
            if "best_model" in pth:
                cls_model_paths.append(os.path.join(args.train_dir, pth))
    else:
        cls_model_paths = [cfg["model"]["classification"]["model_path"]]

    # Gets the valid indices for the evaluated dataset
    valid_idxs = set()
    for data in tqdm(testloader, desc="Dummy run for valid class generation"):
        gt_labels = data["labels"][0, :].numpy().tolist()
        for gt_label in gt_labels:
            if gt_label >= 0:
                valid_idxs.add(gt_label)
                
    valid_idx_dict = {}
    num_valid_cls = len(valid_idxs) + 1
    for v_idx, valid_idx in enumerate(valid_idxs):
        valid_idx_dict[valid_idx] = v_idx + 1

else:
    valid_idxs = set()
    cls_model_paths = None
    valid_idx_dict = {}
    num_valid_cls = 1
    

if "segmentation" in cfg["model"]:
    segm_model = CAVT(cfg["model"]["segmentation"], "cuda:0")
    segm_model = segm_model.eval().cuda()
else:
    segm_model = None   
    
# --------------------------------
# Testing Loop
# --------------------------------
    
# f = open(txt_name, "w")
# f.write('####################### Evaluation Started ####################### \n')
# f.close()

if segm_model is not None and not multi_preprocessors:
    
    aon_fn = AllOrNothing(filter_out_qd=filter_out_qd)
    siou_fn = SequenceIoU(filter_out_qd=filter_out_qd)
    
    for i, data in enumerate(tqdm(testloader)):
        scene_visuals = data["scene_visuals"].cuda()
        scene_strokes = data["vectors"].cuda()
        if scene_strokes[0, 0, -1] < 0: continue
        stroke_areas = data["stroke_areas"].cuda()
        stroke_area_inds = data["stroke_area_inds"].cuda()
        segmentation_sizes = data["segmentation_sizes"].cuda()
        labels = data["labels"].cuda()
        divisions = data["divisions"].cuda()

        with torch.no_grad():
            scores, boxes, ranges = segm_model(
                scene_visuals, 
                scene_strokes, 
                stroke_areas, 
                stroke_area_inds, 
                segmentation_sizes)
        
        if filter_out_qd:
            aon_fn.add(ranges, divisions, labels)
            siou_fn.add(ranges, divisions, labels)
        else:
            aon_fn.add(ranges, divisions)
            siou_fn.add(ranges, divisions)
    
    aon_score = aon_fn.calculate()
    siou_score = siou_fn.calculate()
    
    print("#"*60)
    
    print(f"--> AoN-SCORE  : {aon_score}")
    print(f"--> SIoU-SCORE : {siou_score}")
    print(f"--> Dataset    : {dataset_name} ({partition} partition)")
    print(f"--> Model      : CAVT")
    
    print("#"*60)
    
    print()
    print("-"*60)
    print()
    
    
elif segm_model is None and not multi_preprocessors:

    for bmp in sorted(cls_model_paths):
        
        if "_cbsc" in bmp and "cbsc" not in args.val_data_config: continue
        elif "_friss" in bmp and "friss" not in args.val_data_config: continue

        if not if_single_model:
            model = CGATNet(cfg["model"]["classification"])
        else:
            model = SingleSketchModel(cfg["model"]["classification"])
        
        model.load_state_dict(
            torch.load(bmp)["model"], 
            strict=True)
        model = model.cuda().eval()
        
        acc_fn = Accuracy(top_k=args.topk, valid_ids=valid_idxs)
        
        for i, data in enumerate(tqdm(testloader)):
            sketch_images = data["obj_visuals"].cuda()
            gt_labels = data["labels"].cuda()
            attns = data["attns"].cuda()
            padding_mask = data["padding_mask"].cuda()
    
            if count_each_object_in_scene_as_separate_scenes:
                b, s, c, h, w = sketch_images.shape
                sketch_images = sketch_images.view(-1, 1, c, h, w)
                gt_labels = gt_labels.view(-1, 1)
                padding_mask = padding_mask.view(-1, 1)
                attns = torch.zeros(b * s, attns.shape[1], 1, 1)

            if not if_single_model:
                with torch.no_grad():
                    outputs = model(sketch_images, attns, padding_mask)
            else:
                c, h, w = sketch_images.shape[-3:]
                with torch.no_grad():
                    outputs = model(sketch_images.view(-1, c, h, w))
                gt_labels = gt_labels.view(-1)
            
            acc_fn.add(outputs.cpu().numpy(), gt_labels.cpu().numpy())
            
        acc = acc_fn.calculate()
        
        # f = open(txt_name, "a")   
        # f.write("Model: {} --> Avg acc: {} \n".format(bmp, acc)) 
        # f.close()
        
        dataset = cfg["dataset"][partition]["dataset_name"]
        model = args.train_dir.split("run_results/")[-1]
        
        print("#"*60)
        
        print(f"--> SCORE   : {acc}")
        print(f"--> Dataset : {dataset} ({partition} partition)")
        
        model_dir = model[:-1] if "/" == model[-1] else model
        model_dir = model_dir[model_dir.rfind('/')+1:]
        print(f"--> Model   : {model_dir}")
        
        weight_dir = bmp.replace(args.train_dir, '')
        if "/" == weight_dir[0]: weight_dir = weight_dir[1:]
        print(f"--> Weights : {weight_dir}")
        
        print("#"*60)
        
        print()
        print("-"*60)
        print()
        
elif multi_preprocessors:
    
    for bmp in sorted(cls_model_paths):
        
        segm_fn = SegmentationMetrics(
            num_valid_cls, 
            ignore_bg=True)
        
        if "_1" in bmp: continue
        
        if "_cbsc" in bmp and "cbsc" not in args.val_data_config: continue
        elif "_friss" in bmp and "cbsc" in args.val_data_config: continue
        
        if not if_single_model:
            model = CGATNet(cfg["model"]["classification"])
        else:
            model = SingleSketchModel(cfg["model"]["classification"])
        
        model.load_state_dict(
            torch.load(bmp)["model"], 
            strict=True)
        model = model.cuda().eval()
        
        for i, data in enumerate(tqdm(testloader)):
            
            scene_visuals = data["scene_visuals"].cuda()
            scene_strokes = data["vectors"].cuda()
            if scene_strokes[0, 0, -1] < 0: continue
            stroke_areas = data["stroke_areas"].cuda()
            stroke_area_inds = data["stroke_area_inds"].cuda()
            segmentation_sizes = data["segmentation_sizes"].cuda()
            gt_labels = data["labels"].cuda()
            divisions = data["divisions"].cuda()
            img_size = data["img_size"].cuda()
            
            with torch.no_grad():
                scores, boxes, ranges = segm_model(
                    scene_visuals, 
                    scene_strokes, 
                    stroke_areas, 
                    stroke_area_inds, 
                    segmentation_sizes)
                
            # pass from classification preprocessor
            try:
                sketch_images, attns = cls_preprocessor(
                    scene_strokes.cpu(), 
                    ranges.long().cpu(), 
                    img_size.cpu())
                sketch_images = sketch_images.cuda()
                attns = attns.cuda()
                padding_mask = torch.ones(
                    (1, sketch_images.shape[1], 1), dtype=torch.long).cuda()
            except:
                print("Could not do for:", i)
                continue

            # pass from classification module
            if count_each_object_in_scene_as_separate_scenes:
                b, s, c, h, w = sketch_images.shape
                sketch_images = sketch_images.view(-1, 1, c, h, w)
                gt_labels = gt_labels.view(-1, 1)
                padding_mask = padding_mask.view(-1, 1)
                attns = torch.zeros(b * s, attns.shape[1], 1, 1)
    
            if not if_single_model:
                with torch.no_grad():
                    outputs = model(sketch_images, attns, padding_mask)
            else:
                c, h, w = sketch_images.shape[-3:]
                with torch.no_grad():
                    outputs = model(sketch_images.view(-1, c, h, w))
                
            outputs = outputs.view(-1, outputs.shape[-1])
            pred_classes = outputs.argsort(dim=-1, descending=True).long()
            
            valid_preds, gt_opts = [], set(gt_labels.view(-1).cpu().numpy().tolist())
            for s in range(pred_classes.shape[0]):
                for obj in range(pred_classes.shape[-1]):
                    pred_result = pred_classes[s, obj].item()
                    if pred_result not in valid_idx_dict:
                        continue
                    
                    mapped_result = valid_idx_dict[pred_result] 
                    if ov_style_preds:
                        if pred_result in gt_opts:
                            valid_preds.append(mapped_result)
                            break
                    else:
                        valid_preds.append(mapped_result)
                        break

            # generate CLASS MATRICES
            
            str_starts = [0] + (torch.where(scene_strokes.squeeze(0)[:, -1] == 1)[0] + 1).cpu().numpy().tolist()
            str_to_pnt_maps = {}
            for str_idx, str_start in enumerate(str_starts):
                str_to_pnt_maps[str_idx] = str_start
            # str_to_pnt_maps[len(str_starts)] = scene_strokes.shape[-2]
            
            orig_color_fg = copy.deepcopy(segm_preprocessor.color_fg)
            segm_preprocessor.color_fg = True

            gt_mtx = segm_preprocessor.create_class_segmentation_mtx(
                scene_strokes.squeeze(0).cpu().numpy(),
                [str_to_pnt_maps[div] for div in divisions.squeeze(0).cpu().numpy().tolist()],
                [valid_idx_dict[cls] for cls in gt_labels.squeeze(0).cpu().numpy().tolist()],
                *img_size.squeeze(0).cpu().numpy().tolist())
            
            pred_mtx = segm_preprocessor.create_class_segmentation_mtx(
                scene_strokes.squeeze(0).cpu().numpy(),
                [str_to_pnt_maps[div] for div in ranges.squeeze(0).cpu().numpy().tolist()],
                valid_preds,
                *img_size.squeeze(0).cpu().numpy().tolist())
            
            segm_fn.add(pred_mtx, gt_mtx)
            
            segm_preprocessor.color_fg = orig_color_fg
        
        ova_acc, mean_acc, mean_iou, fw_iou = segm_fn.calculate()
        
        print("#"*60)
        print(f"--> OVA-acc  : {ova_acc}")
        print(f"--> Mean-acc : {mean_acc}")
        print(f"--> Mean-IoU : {mean_iou}")
        print(f"--> FW-IoU   : {fw_iou}")
        print(f"--> Dataset  : {dataset_name} ({partition} partition)")
        weight_dir = bmp.replace(args.train_dir, '')
        print(f"--> Model    : CAVT + {weight_dir}")
        print("#"*60)
        
        print()
        print("-"*60)
        print()