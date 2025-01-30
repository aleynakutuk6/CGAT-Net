import random
import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.io
import os
import argparse
import math
import json
import sys
sys.path.append('libs')
sys.path.append('tools')

from data_loader import load_image, load_label, preload_dataset
import adapted_deeplab_model
from segment_densecrf import seg_densecrf
from semantic_visualize import visualize_semantic_segmentation
from edgelist_utils import refine_label_with_edgelist


if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('CUDA:', os.environ['CUDA_VISIBLE_DEVICES'])



def filter_predictions(pred, class_mappings=None, gt_class_ids=None):
    
    # pred: 1 x H x W x Nclasses
    if len(pred.shape) == 3:
        pred = pred[np.newaxis,...]
    B, H, W, Nclasses = pred.shape
    results = np.zeros((B, H, W, 1))
    gt_class_set = set(gt_class_ids.flatten())
    for b in range(B):
        for h in range(H):
            for w in range(W):
                pred_cell = pred[b,h,w]
                sorted_pred_cell = np.argsort(pred_cell)[::-1]
                for class_id_iter in sorted_pred_cell:
                    class_id = class_id_iter + 1
                    if class_mappings is not None:
                        if class_id in class_mappings:
                            results[b,h,w,0] = class_id_iter
                            break
                    elif class_id in gt_class_set:
                        results[b,h,w,0] = class_id_iter
                        break
    return results
        

def segment_main(**kwargs):

    mu = (104.00698793, 116.66876762, 122.67891434)
    
    if kwargs['cfg_file'] == 'sky':
        from cfgs.sky_scene_cfg import FLAGS
    elif kwargs['cfg_file'] == 'tub':
        from cfgs.tub_scene_cfg import FLAGS
    elif kwargs['cfg_file'] == 'sketchy':
        from cfgs.sketchyscene_cfg import FLAGS
        
    if kwargs['mapping_path'] is not None:
        with open(kwargs['mapping_path'], "r") as f:
            class_mappings = json.load(f)
        class_mappings = set([int(num) for num in class_mappings.keys()])
        gt_label = None
    else:
        class_mappings = None
        
    if FLAGS.ignore_class_bg:
        nSketchClasses = FLAGS.nSketchClasses - 1
        print('Ignore BG;', nSketchClasses, 'classes')
    else:
        nSketchClasses = FLAGS.nSketchClasses
        print('Not Ignore BG;', nSketchClasses, 'classes')

    data_aug = False
    mode = 'test'
    
    model = adapted_deeplab_model.DeepLab(num_classes=nSketchClasses,
                                          lrn_rate=FLAGS.learning_rate,
                                          lrn_rate_end=FLAGS.learning_rate_end,
                                          optimizer=FLAGS.optimizer,
                                          upsample_mode=FLAGS.upsample_mode,
                                          data_aug=data_aug,
                                          image_down_scaling=FLAGS.image_down_scaling,
                                          ignore_class_bg=FLAGS.ignore_class_bg,
                                          mode=mode)

    snapshot_saver = tf.train.Saver(max_to_keep=0)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    # os.system('sudo nvidia-smi -i %s -c 0' % os.environ['CUDA_VISIBLE_DEVICES'])
    sess = tf.Session(config=tfconfig)
    # os.system('sudo nvidia-smi -i %s -c 2' % os.environ['CUDA_VISIBLE_DEVICES'])
    sess.run(tf.global_variables_initializer())
    # snapshot_saver.restore(sess, tf.train.latest_checkpoint(kwargs['ckpt_file']))

    snapshot_dir = os.path.join(FLAGS.outputs_base_dir, FLAGS.snapshot_folder_name)
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_dir = os.path.join(snapshot_dir, FLAGS.run_name)
    # os.makedirs(snapshot_dir, exist_ok=(mode != 'train'))

    ckpt = tf.train.get_checkpoint_state(snapshot_dir)
    start_iter = 0
    
    load_var = {var.op.name: var for var in tf.global_variables()
                if var.op.name.startswith('ResNet')
                and 'global_step' not in var.op.name  # count from 0
                }
                
    snapshot_loader = tf.train.Saver(load_var)
    ckpt_file = FLAGS.ckpt_file
    print('Trained model found, loaded', ckpt_file)
    snapshot_loader.restore(sess, ckpt_file)
    
    use_dcrf = kwargs['use_dcrf']
    use_edgelist = kwargs['use_edgelist']
    eval_base_dir = os.path.join(FLAGS.outputs_base_dir, 'eval_results')
    os.makedirs(eval_base_dir, exist_ok=True)
    nImgs = len([p for p in os.listdir(os.path.join(kwargs['data_dir'], mode, 'DRAWING_GT')) if ".png" in p])
    print("# of test images:", nImgs)
    outstr = mode + ' mode\n'
    cat_max_len = 16
    data_name = kwargs['data_dir'].split("/")[-1]
    print("{} is processing...".format(data_name))
    
    for imgIndex in range(1, nImgs + 1):
        ## load images
        image_name = 'L0_sample' + str(imgIndex) + '.png'  # e.g. L0_sample5564.png
        image_path = os.path.join(kwargs['data_dir'], mode, 'DRAWING_GT', image_name)
        test_image = load_image(image_path, mu)  # shape = [1, H, W, 3]
        ## load gt_label
        label_name = 'sample_' + str(imgIndex) + '_class.mat'  # e.g. sample_1_class.mat
        label_path = os.path.join(kwargs['data_dir'], mode, 'CLASS_GT', label_name)
        gt_label = np.squeeze(load_label(label_path))  # [1, H, W] -> [1, H, W]
        
        print("gt set: ", set(gt_label.flatten()))
        print('#' + str(imgIndex) + '/' + str(nImgs) + ': ' + image_path)
        feed_dict = {model.images: test_image, model.labels: 0}
        pred, pred_label_no_crf = sess.run([model.pred, model.pred_label], feed_dict=feed_dict)
        
        pred_label_no_crf = filter_predictions(pred, class_mappings, gt_label)
        
        if FLAGS.ignore_class_bg:
            pred_label_no_crf = pred_label_no_crf + 1  # [1, 46]
        # print('@ pred.shape ', pred.shape)  # (1, H, W, nSketchClasses)
        # print(pred_label_no_crf.shape)  # shape = [1, H, W, 1]
        if use_dcrf:
            prob_arr = np.squeeze(pred)
            prob_arr = prob_arr.transpose((2, 0, 1))  # shape = (nSketchClasses, H, W)
            d_image = np.array(np.squeeze(test_image), dtype=np.uint8)  # shape = (H, W, 3)
            pred_label = seg_densecrf(prob_arr, d_image, nSketchClasses)  # shape=[H, W]
            if FLAGS.ignore_class_bg:
                pred_label = pred_label + 1  # [1, 46]
        else:
            pred_label = np.squeeze(pred_label_no_crf)  # [H, W], [1,46]
        # ignore background pixel prediction (was random)
        # must before edgelist
        pred_label[gt_label == 0] = 0
        if use_edgelist:
            pred_label = \
            refine_label_with_edgelist(imgIndex, mode, \
                                       kwargs['data_dir'],
                                       pred_label.copy())
        
        classes_dir = os.path.join(eval_base_dir, kwargs['cfg_file'] + '_model', data_name, 'CLASS_PRED')            
        os.system(f"mkdir -p {classes_dir}")
        
        scipy.io.savemat(os.path.join(classes_dir, "sample_" + str(imgIndex) + "_class.mat"), {'CLASS_PRED': pred_label})
        
        gt_flat = gt_label.flatten()
        pred_flat = pred_label.flatten()
        
        ints = np.where(gt_flat > 0)[0]
        print("gt:", gt_flat[ints])
        print("pred:", pred_flat[ints])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--black_bg', '-bl', type=int, choices=[0, 1],
                        default=0, help="use black or white background for inference")
    parser.add_argument('--dcrf', '-crf', type=int, choices=[0, 1],
                        default=0, help="use dense crf or not")
    parser.add_argument('--edgelist', '-el', type=int, choices=[0, 1],
                        default=0, help="use edgelist or not")
    parser.add_argument('--cfg_file', '-cfg', type=str, choices=["sky", "tub", "sketchy"],
                        default="sketchy", help="config file of the model")
    parser.add_argument('--data_dir', '-d', type=str, 
                        default="../datasets/FrISS-Sketchy", help="path of the dataset")
    parser.add_argument('-m', '--mapping-path', 
                        default="../datasets/model_to_dataset_mappings/ldp_sketchy_to_friss_sketchy.json", 
                        type=str, help="path of the class mappings")
    args = parser.parse_args()

    run_params = {
        "black_bg": args.black_bg,
        "use_dcrf": args.dcrf,
        "use_edgelist": args.edgelist,
        "cfg_file": args.cfg_file,
        "data_dir": args.data_dir,
        "mapping_path": args.mapping_path
    }

    segment_main(**run_params)
    