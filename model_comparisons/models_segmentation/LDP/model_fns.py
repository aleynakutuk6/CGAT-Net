import os
import cv2
import math
import json
import random
import time
import argparse
import scipy.io
import numpy as np
import multiprocess as mp

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from PIL import Image

from relnet.utils.visualize_utils import *

import sys
sys.path.append("models_segmentation/LDP")
sys.path.append('models_segmentation/LDP/libs')
sys.path.append('models_segmentation/LDP/tools')

import adapted_deeplab_model

from data_loader import load_image, load_label, preload_dataset
from edgelist_utils import refine_label_with_edgelist
from segment_densecrf import seg_densecrf
from semantic_visualize import visualize_semantic_segmentation


def preprocessor(np_img: np.ndarray):
    np_img = np_img.astype(float)
    mu = (104.00698793, 116.66876762, 122.67891434)
    np_img = np_img[:, :, ::-1]  # rgb -> bgr
    np_img -= mu  # subtract mean
    np_img = np.expand_dims(np_img, axis=0) # shape = [1, H, W, 3]
    return np_img
    
    
def load_model(FLAGS):
    data_aug, mode = FLAGS.data_aug, 'test'
    n_classes = FLAGS.nSketchClasses-1
    
    model = adapted_deeplab_model.DeepLab(
        num_classes=n_classes,
        lrn_rate=FLAGS.learning_rate,
        lrn_rate_end=FLAGS.learning_rate_end,
        optimizer=FLAGS.optimizer,
        upsample_mode=FLAGS.upsample_mode,
        data_aug=data_aug,
        image_down_scaling=FLAGS.image_down_scaling,
        ignore_class_bg=FLAGS.ignore_class_bg,
        mode=mode)
    
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())
    
    load_var = {
        var.op.name: var for var in tf.global_variables()
        if var.op.name.startswith('ResNet')
        and 'global_step' not in var.op.name  # count from 0
    }           
    snapshot_loader = tf.train.Saver(load_var)
    ckpt_file = FLAGS.ckpt_file
    snapshot_loader.restore(sess, ckpt_file)
    
    return {
        "model": model,
        "session": sess,
        "n_classes": n_classes
    }

def load_model_sky():
    class Flags:
        def __init__(self):
            self.nSketchClasses = 31
            self.learning_rate = 0.0001
            self.learning_rate_end = 0.00001
            self.optimizer = 'adam'
            self.upsample_mode = 'deconv'
            self.data_aug = False
            self.image_down_scaling = False
            self.ignore_class_bg = True
            self.ckpt_file = 'weights/LDP/LDP_SKY_nocrf_NM/iter_150000.tfmodel'
    
    FLAGS = Flags()
    out = load_model(FLAGS)
    
    f = open("models_segmentation/LDP/sky_to_cgatnet_maps.json", "r")
    labels_info = json.load(f)
    f.close()
    
    out.update({
        "model_type": "sky", 
        "labels_info": labels_info
    })
    
    return out


def load_model_sketchy():
    class Flags:
        def __init__(self):
            self.nSketchClasses = 47
            self.learning_rate = 0.0001
            self.learning_rate_end = 0.00001
            self.optimizer = 'adam'
            self.upsample_mode = 'deconv'
            self.data_aug = False
            self.image_down_scaling = False
            self.ignore_class_bg = True
            self.ckpt_file = 'weights/LDP/LDP_SS/iter_150000.tfmodel'
    
    FLAGS = Flags()
    out = load_model(FLAGS)
    
    f = open("models_segmentation/LDP/sketchy_to_cgatnet_maps.json", "r")
    labels_info = json.load(f)
    f.close()
    
    out.update({
        "model_type": "sketchy",
        "labels_info": labels_info
    })
    return out


def map_predictions(pred_mtx, labels_info, labels_mapping):
    H, W = pred_mtx.shape[:2]
    pred_mtx = np.argsort(pred_mtx, axis=-1)[..., ::-1]
    final_preds = np.zeros((H, W))
    
    overall_map = {}
    for idx in labels_info["idx_to_label"]:
        ldp_name = labels_info["idx_to_label"][idx]
        cgatnet_name = labels_info["label_to_cgatnet"][ldp_name]
        if (cgatnet_name is not None and 
            cgatnet_name in labels_mapping["label_to_idx"]):
            overall_map[int(idx)] = labels_mapping["label_to_idx"][cgatnet_name]
        else:
            overall_map[int(idx)] = None

    for h in range(H):
        for w in range(W):
            sorted_inds = pred_mtx[h, w, :]
            for sel_ind in sorted_inds:
                cgat_idx = overall_map[sel_ind]
                if cgat_idx is not None:
                    final_preds[h, w] = cgat_idx
                    break

    return final_preds.astype(int)
        


def pass_from_model(
    info, scene_strokes, 
    labels, labels_mapping, 
    W, H, thickness, to_vis=False):
    
    model = info["model"]
    sess = info["session"]
    model_type = info["model_type"]
    labels_info = info["labels_info"]
    n_classes = info["n_classes"]

    if not to_vis:
        scene_visuals, _ = draw_sketch(
            scene_strokes,
            canvas_size=[W, H],
            margin=0,
            white_bg=True,
            color_fg=False,
            shift=False,
            scale_to=-1,
            is_absolute=True,
            thickness=thickness
        )
        scale_size = 750
    
    else:
        scene_visuals, _ = draw_sketch(
            scene_strokes,
            canvas_size=[800, 800],
            margin=50,
            white_bg=True,
            color_fg=False,
            shift=True,
            scale_to=800,
            is_absolute=True,
            thickness=thickness
        )
        W, H = 800, 800
        scale_size = 750

    scene_visuals = scene_visuals.astype(np.uint8)
    nobg_mtx = (255 - scene_visuals.astype(float)[..., 0]) / 255.
    
    new_scene_visuals = np.full((max(H, W), max(H, W), 3), 255, dtype=np.uint8)
    new_scene_visuals[:H, :W, :] = scene_visuals
    scene_visuals = new_scene_visuals
    ratio = float(scale_size) / float(max(H, W))
    
    scene_visuals = cv2.resize(
        scene_visuals, (scale_size, scale_size), 
        interpolation = cv2.INTER_NEAREST)
    scene_visuals = preprocessor(scene_visuals)

    pred, pred_label_no_crf = sess.run(
        [model.pred, model.pred_label], 
        feed_dict={
            model.images: scene_visuals, 
            model.labels: 0
        }
    )
    pred = pred[0, ...]

    pred = seg_densecrf(
        pred, 
        scene_visuals[0, ...].astype(np.uint8), 
        n_classes)

    pred = map_predictions(
        pred, labels_info, labels_mapping)

    pred = cv2.resize(
        pred, (max(H, W), max(H, W)), 
        interpolation=cv2.INTER_NEAREST)
    pred = pred[:H, :W]
    pred = (pred * nobg_mtx).astype(int)
    
    return pred