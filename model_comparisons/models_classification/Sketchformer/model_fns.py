import os
import sys
sys.path.append('models_classification')

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from tqdm import tqdm
from relnet.utils.sketch_utils import *

import Sketchformer.utils as utils 
from Sketchformer.models import *
from Sketchformer.dataloaders import *
from Sketchformer.utils.hparams import *
from Sketchformer.utils.tokenizer import Tokenizer

def load_model():
    Model = get_model_by_name("sketch-transformer-tf2")
    DataLoader = get_dataloader_by_name('stroke3-distributed')
    hps = combine_hparams_into_one(Model.default_hparams(), DataLoader.default_hparams())
    load_config(hps, Model.get_config_filepath("weights/sketchformer/", "cvpr_tform_tok_dict"))
    utils.gpu.setup_gpu([0])
    dataset = DataLoader(hps, "/datasets/quickdraw/sketchformer_processed/")
    model = Model(hps, dataset, "weights/sketchformer/", "cvpr_tform_tok_dict")
    model.restore_checkpoint_if_exists("latest")
    
    tokenizer = Tokenizer("models_classification/Sketchformer/prep_data/sketch_token/token_dict.pkl")
    tokenizer.max_seq_len = 200

    with open("models_classification/Sketchformer/qd_list.txt", "r") as f:
        lines = f.readlines()
        
    labels_info = {}
    for l_idx, line in enumerate(lines):
        cls_name = line.replace("\n", "").strip()
        labels_info[l_idx] = cls_name
        
    return {
        "model": model,
        "tokenizer": tokenizer,
        "labels_info": labels_info
    }


def pass_from_model(info, abs_stroke3, divisions):
    if abs_stroke3 is None or abs_stroke3[0, -1] < 0: return None

    model = info["model"]
    tokenizer = info["tokenizer"]
    labels_info = info["labels_info"]

    stroke_starts = [0] + (np.where(abs_stroke3[:, -1] == 1)[0] + 1).tolist()
    start_dict = {num: st for num, st in enumerate(stroke_starts)}
    assert divisions[0] == 0 and start_dict[divisions[-1]] == abs_stroke3.shape[0]
    
    pred_names = []
    for i in range(len(divisions)-1):
        st, end = divisions[i], divisions[i+1]
        pnt_st, pnt_end = start_dict[st], start_dict[end]
        sketch = abs_stroke3[pnt_st:pnt_end, :].astype(int).astype(float)
        sketch_temp = copy.deepcopy(sketch)
        
        min_x, min_y, max_x, max_y = get_absolute_bounds(sketch_temp)
        sketch_temp[:, 0] -= min_x
        sketch_temp[:, 1] -= min_y               
        sketch_temp = normalize(absolute_to_relative(sketch_temp))
        if sketch_temp is None:
            sketch_temp = np.asarray([[0,0,0], [1,1,1]]).astype(float)
        
        sketch_encoded = tokenizer.encode(sketch_temp)
        encoded_list = np.asarray([sketch_encoded])
        
        pred = model.encode_from_seq(encoded_list)["class"].numpy()
        pred = pred[0, :].argsort()[::-1]
        pred_names.append([labels_info[val] for val in pred])
    
    return pred_names
