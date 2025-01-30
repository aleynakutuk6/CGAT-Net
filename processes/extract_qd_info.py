import copy
import numpy as np
import json
import os
import sys
import random
import cv2

from tqdm import tqdm
sys.path.append('/scratch/users/akutuk21/hpc_run/SketchNet-Tubitak-Project/')
from sknet.utils.visualize_utils import draw_sketch
from sknet.utils.sketch_utils import *


def get_qd_classes(qd_txt_path="qd_coco_files/list_quickdraw.txt"):
    classes = []
    with open(qd_txt_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        classes.append(line.replace("\n",""))
    return classes
    

def read_quickdraw_npz(filepath: str, partition: str=None, idx=None):

    assert partition is not None or (partition is None and idx is None)
    assert partition is None or partition in ['train', 'valid', 'test']
    assert idx is None or idx == "random" or idx >= 0
    
    if not os.path.isfile(filepath):
        raise ValueError(f"No NPZ file exists in: {filepath}")
    
    sketches = np.load(filepath, allow_pickle=True, encoding="bytes")
    
    def clean_sketch(sketch: np.ndarray) -> np.ndarray:
        sk_new = sketch.astype(float)
        if sk_new[0, -1] == 1:
            sk_new = sk_new[1:, :]
        
        end_checks = sk_new[1:, -1] + sk_new[:-1, -1]
        vals = np.where(end_checks > 1)[0]
        if vals.shape[0] > 0:
            for j in np.flip(vals):
                sk_new = np.delete(sk_new, j+1, axis=0)
        
        return sk_new

    
    if partition is not None:
        sketches = sketches[partition]
        if idx is None:
            return [clean_sketch(sk) for sk in sketches]
        elif idx == "random":
            n_samples = len(sketches)
            idx = random.randint(0, n_samples-1)
            return clean_sketch(sketches[idx]), idx
        else:
            return clean_sketch(sketches[idx])
    else:
        tr = [clean_sketch(sk) for sk in sketches["train"]]
        val = [clean_sketch(sk) for sk in sketches["valid"]]
        test = [clean_sketch(sk) for sk in sketches["test"]]
        return tr, val, test


def extract_quickdraw_wh_ratio(data_path, save_path, set_type, sample_cnt=5):
    
    save_path = os.path.join(save_path, set_type)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    sketches_path = os.path.join(save_path, "sketches")
    if not os.path.isdir(sketches_path):
        os.mkdir(sketches_path)
                
    ratios_folder = os.path.join(save_path, "wh_ratios")
    if not os.path.isdir(ratios_folder):
        os.mkdir(ratios_folder)
                
    for file_name in tqdm(os.listdir(data_path)):
        if "npz" not in file_name:
            continue

        npz_name = file_name.split(".")[0]
        qd_data = read_quickdraw_npz(os.path.join(data_path, npz_name + ".npz"), partition=set_type, idx=None)
        n_samples = len(qd_data)
        rand_ids = random.sample(np.arange(0, n_samples-1).tolist(), sample_cnt)
        
        ratios_data = {}
        
        for idx in tqdm(rand_ids):
                
            sketch = np.asarray(qd_data[idx])
            
            npz_path = os.path.join(sketches_path, npz_name)
            if not os.path.isdir(npz_path):
                os.mkdir(npz_path)
                
            save_name = npz_name + '_' + str(idx)
            np.save(os.path.join(npz_path, save_name + '.npy'), sketch)
            
            sketch_temp = copy.deepcopy(sketch)
            sketch_temp = apply_RDP(sketch_temp)
            sketch_temp = normalize(sketch_temp)
            min_x, min_y, max_x, max_y = get_relative_bounds(sketch_temp)
            w = max_x - min_x
            h = max_y - min_y
            
            if h == 0.:
                ratios_data[save_name] = 0.
            else:
                ratios_data[save_name] = w/h
        
        with open(os.path.join(ratios_folder, npz_name + ".json"), "w") as f:
            json.dump(ratios_data, f)  


data_path = "/datasets/quickdraw/sketchrnn/npz/"
save_path = "../QD-Dataset"
set_type = "test"

extract_quickdraw_wh_ratio(data_path, save_path, set_type)
