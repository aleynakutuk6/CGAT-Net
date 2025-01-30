import os
import json
import numpy as np
import random
import copy

import sys
sys.path.append('/scratch/users/akutuk21/hpc_run/SketchNet-Tubitak-Project/')
from sknet.utils.visualize_utils import draw_sketch
from sknet.utils.sketch_utils import *

from tqdm import tqdm

json_paths = "/kuacc/users/akutuk21/hpc_run/Collected-Scene-Dataset/custom-dataset/test/scenes/jsons/"
files_path = "/kuacc/users/akutuk21/hpc_run/Collected-Scene-Dataset/custom-dataset/test/path_info.json"
info_path  = "/kuacc/users/akutuk21/hpc_run/Collected-Scene-Dataset/custom-dataset/test/data_info.json"
class_changes_path = "/kuacc/users/akutuk21/hpc_run/Collected-Scene-Dataset/json_files/class_changes.json"
friss_qd_mapping_path = "/kuacc/users/akutuk21/hpc_run/Sketch-Graph-Network/datasets/mapping_files/friss_to_qd_mappings.json"


def read_class_changes(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def get_classid_to_classname_maps(info_path):
    with open(info_path, "r") as f:
        data = json.load(f)["data"]
    
    mapping = {}
    for el in data:
        cid = el["class_id"]
        cname = el["class_name"]
        if cid not in mapping:
            mapping[cid] = cname
    
    return mapping
    
    
def split_partitions(files_path):
    with open(files_path, "r") as f:
        data = json.load(f)

    class_groups = {}
    for k in data:
        _, scene_id, _, user_id = k.split("-")
        class_ids = data[k]["class_ids"]
        for cid in class_ids:
            if cid not in class_groups:
                class_groups[cid] = []
            class_groups[cid].append(k)
            
    tr_friss_set = set()
    for cid in class_groups:
        ks = class_groups[cid]
        if_found = False
        for k in ks:
            if k in tr_friss_set:
                if_found = True
                break
        if not if_found:
            tr_friss_set.add(random.choice(ks))
    
    scene_groups, num_obj_thold = {}, 3
    for k in data:
        _, scene_id, _, user_id = k.split("-")
        num_objs = len(data[k]["class_ids"])
        
        if scene_id not in scene_groups:
            scene_groups[scene_id] = {"lower": [], "higher": [], "in_tr": []}
        
        if k in tr_friss_set:
            scene_groups[scene_id]["in_tr"].append(k)
        elif num_objs < num_obj_thold:
            scene_groups[scene_id]["lower"].append(k)
        else:
            scene_groups[scene_id]["higher"].append(k)
    
    
    tr_ratio, test_ratio = 0.35, 0.5
    tr_friss, val_friss, test_friss = [], [], []
    for scene_id in scene_groups:
        in_tr = scene_groups[scene_id]["in_tr"]
        lowers = scene_groups[scene_id]["lower"]
        highers = scene_groups[scene_id]["higher"]
        random.shuffle(highers)
        
        total = len(in_tr) + len(lowers) + len(highers)
        num_tr = max(len(in_tr) + len(lowers), round(total * tr_ratio))
        num_test = min(len(highers), round(total * test_ratio))
        num_val = max(0, total - num_tr - num_test)
        
        assert total == num_tr + num_val + num_test

        # generate train set for that scene
        tr_friss.extend(in_tr)
        num_tr -= len(in_tr)
        tr_friss.extend(lowers)
        num_tr -= len(lowers)
        if num_tr > 0:
            tr_friss.extend(highers[:num_tr])
        
        # generate test set for that scene
        if num_test > 0:
            test_friss.extend(highers[num_tr:num_tr + num_test])
        
        # generate val set for that scene
        if num_val > 0:
            val_friss.extend(highers[num_tr + num_test:])
        
    print("# of training images   :", len(tr_friss))
    print("# of validation images :", len(val_friss))
    print("# of test images       :", len(test_friss))
    
    return tr_friss, val_friss, test_friss

  
def process_individual_jsons(root, scene_list, save_dir, class_changes, id_to_name_map):
    
    with open(os.path.join(friss_qd_mapping_path), "r") as f:
        friss_to_qd = json.load(f)["friss_to_qd"]
    
    for scene in tqdm(scene_list):
        with open(os.path.join(root, scene + ".json"), "r") as f:
            data = json.load(f)
        
        clean_json = {
            "img_id": data["img_id"],
            "img_w": data["img_w"],
            "img_h": data["img_h"],
            "object_divisions": data["object_divisions"],
            "scene_strokes": data["scene_strokes"],
            "gt_class_names": []
        }
        
        for cls_id in data["gt_classes"]:
            class_name = id_to_name_map[cls_id]
            if class_name in class_changes:
                class_name = class_changes[class_name]
            if class_name in friss_to_qd and friss_to_qd[class_name] is not None:
                class_name = friss_to_qd[class_name]
            clean_json["gt_class_names"].append(class_name)
        
        with open(os.path.join(save_dir, scene + ".json"), "w") as f:
            json.dump(clean_json, f)
            
            

def extract_friss_wh_ratio(data_lst, path_info, save_path, class_changes):
    
    with open(path_info, "r") as f:
        path_data = json.load(f)
        
    with open(os.path.join(friss_qd_mapping_path), "r") as f:
        friss_to_qd = json.load(f)["friss_to_qd"]
    
    sketches_path = os.path.join(save_path, 'sketches')
    os.system(f"mkdir -p {sketches_path}")
    
    ratios_folder = os.path.join(save_path, 'wh_ratios')
    os.system(f"mkdir -p {ratios_folder}")
        
    for file_name in tqdm(data_lst):
        sketches_lst = path_data[file_name]["vector_img_paths"]        
        
        for sk_path in sketches_lst:
            with open(os.path.join(path_info.replace("path_info.json", ""), sk_path), "r") as f:
                sk_data = json.load(f)
            
            sketch = np.asarray(sk_data["stroke"])
            sk_name = sk_path.split("/")[-1].replace(".json", "")
            npz_name, npz_userid, npz_extension = sk_name.split("_")
            
            # update npz name 
            if npz_name in class_changes:
                npz_name = class_changes[npz_name]
            if npz_name in friss_to_qd and friss_to_qd[npz_name] is not None:
                npz_name = friss_to_qd[npz_name]
            
            sk_name = npz_name + npz_userid + npz_extension
            
            npz_path = os.path.join(sketches_path, npz_name)
            if not os.path.isdir(npz_path):
                os.mkdir(npz_path)

            np.save(os.path.join(npz_path, sk_name + '.npy'), sketch)
            
            sketch_temp = copy.deepcopy(sketch)
            sketch_temp = apply_RDP(sketch_temp)
            sketch_temp = normalize(sketch_temp)
            min_x, min_y, max_x, max_y = get_relative_bounds(sketch_temp)
            w = max_x - min_x
            h = max_y - min_y
            
            ratios_file = os.path.join(ratios_folder, npz_name + ".json")
            if not os.path.isfile(ratios_file):
                ratios_data = {}
            else:
                with open(ratios_file, "r") as f:
                    ratios_data = json.load(f)
                    
            if h == 0.:
                ratios_data.update({sk_name: 0.})
            else:
                ratios_data.update({sk_name: w/h})
        
            with open(ratios_file, "w") as f:
                json.dump(ratios_data, f)

#######################

id_to_name_map = get_classid_to_classname_maps(info_path)
tr_friss, val_friss, test_friss = split_partitions(files_path)

data_dict = {"train": tr_friss, "val": val_friss, "test": test_friss}
os.system(f"mkdir -p ../datasets/FrISS")
with open("../datasets/FrISS/data_partitions.json", "w") as f:
    json.dump(data_dict, f)

class_changes = read_class_changes(class_changes_path)
    
for partition in data_dict:
    os.system(f"mkdir -p ../datasets/FrISS/{partition}")
    process_individual_jsons(
        json_paths, 
        data_dict[partition], 
        f"../datasets/FrISS/{partition}", 
        class_changes, 
        id_to_name_map)
    
    os.system(f"mkdir -p ../datasets/FrISS/ratios/{partition}")
    extract_friss_wh_ratio(data_dict[partition], files_path, f"../datasets/FrISS/ratios/{partition}", class_changes)