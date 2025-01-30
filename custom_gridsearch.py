import os
import torch
import copy

from tqdm import tqdm
from torch.utils.data import DataLoader

from relnet.data import CBSCDataset, COCODataset
from relnet.data.preprocessors import CGATNetPreprocessor, CAVTPreprocessor
from relnet.models import CGATNet, CAVT, SingleSketchModel
from relnet.metrics import Accuracy, AllOrNothing, SequenceIoU, SegmentationMetrics
from relnet.utils.cfg_utils import parse_configs

# --------------------------------
# Parsing the config file
# --------------------------------

train_dir = "run_results/cavt"
segm_config = "run_results/cavt/training_config.json"
extra_filter_file = "datasets/mapping_files/common_classes/complete.txt"
txt_name = "grid_search_results_updated.txt"

cfg = parse_configs(whole_config_path=segm_config)
segm_model = CAVT(cfg["model"]["segmentation"], "cuda:0")
segm_model = segm_model.eval().cuda()

exps = {
    1: [
        [0.75, 3],
        [0.8, 3],
        [0.35, 5], 
    ],
    2: [
        [0.75, 3],
        [0.6, 3],
        [0.7, 3],
        [0.55, 5],
        [0.75, 1],
        [0.7, 1],
        [0.45, 1],
        [0.5, 5],
        [0.5, 1]
    ],
    3: [
        [0.6, 3],
        [0.65, 5],
        [0.7, 1],
        [0.45, 3],
        [0.5, 7],
        [0.45, 1],
        [0.8, 1]
    ],
    4: [
        [0.75, 5]
    ]
}

params = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
slt = 0.1

f = open(txt_name, "w")
f.write("dataset_name,initOR,thickness,remOR,nrep,slt,aon_score,siou_score\n")
f.close()

for dataset_name in ["friss", "cbsc"]:
    if dataset_name == "cbsc":
        map_file = "datasets/mapping_files/cbsc_to_qd_mappings.json"
    elif dataset_name == "friss":
        map_file = "datasets/mapping_files/friss_segmentation_mappings.json"
    
    for thickness in exps:
        cfg = parse_configs(
            val_data_path=f"configs/dataset/{dataset_name}.json", 
            whole_config_path=segm_config,
            override_params=(
                f"mapping_file={map_file} ; "
                f"extra_filter_file={extra_filter_file}"))
        cfg["dataset"]["val"] = cfg["dataset"]["val"][0]
        
        cfg["preprocessor"]["segmentation"]["thickness"] = thickness
        segm_preprocessor = CAVTPreprocessor(cfg["preprocessor"]["segmentation"])

        test_data = CBSCDataset(
            "val", cfg["dataset"], 
            save_dir=train_dir, 
            preprocessor=segm_preprocessor)
        
        testloader = DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=4)
        
        pass_runs = []
        
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
                scores, boxes = segm_model.forward_only_model(scene_visuals)
                
            pass_runs.append([
                divisions, scores, boxes, 
                scene_strokes, stroke_areas, 
                stroke_area_inds, segmentation_sizes])
            
        for remOR, nrep in exps[thickness]:
            segm_model.rem_OR = remOR
            segm_model.num_repeats = nrep
            segm_model.select_larger_thold = slt
            
            pbar = tqdm(params)
            for initOR in pbar:
                segm_model.init_OR = initOR
                pbar.set_description(f"D:{dataset_name} | Tck:{thickness} | "
                                     f"initOR:{initOR} | remOR:{remOR} | "
                                     f"NR:{nrep} | SLT:{slt}")
                    
                aon_fn = AllOrNothing()
                siou_fn = SequenceIoU()
                
                for data in pass_runs:
                    data_cpy = copy.deepcopy(data)
                    divs = data_cpy[0]
                    _, _, ranges = segm_model.postprocess(*data_cpy[1:])
                    aon_fn.add(ranges, divs)
                    siou_fn.add(ranges, divs)

                aon_score = aon_fn.calculate()
                siou_score = siou_fn.calculate()
                
                f = open(txt_name, "a")
                f.write(f"{dataset_name},{initOR},{thickness},"
                        f"{remOR},{nrep},{slt},{aon_score},{siou_score}\n")
                f.close()
        