import os
import json
import torch
import random
import numpy as np
import h5py
import cv2

from PIL import Image
from tqdm import tqdm
from .base_dataset import SyntheticBaseSceneDataset

class VGDataset(SyntheticBaseSceneDataset):

    def __init__(self, split, cfg, save_dir: str=None):
        super().__init__(split, cfg, save_dir)
        self.img_dir = cfg[split]["img_dir"]
        self.dict_file = cfg[split]["dict_file"]
        self.roidb_file = cfg[split]["roidb_file"]
        self.image_file = cfg[split]["image_file"]
        self.box_scale = cfg[split]["box_scale"]
        num_val_im = cfg["val"]["num_val_im"]
        
        self.object_idx_to_name = load_info(self.dict_file)
        self.split_mask, self.gt_boxes, self.gt_classes = load_graphs(
            self.roidb_file, self.split, num_val_im=num_val_img, box_scale=self.box_scale)
        self.filenames, self.img_info = load_image_filenames(self.img_dir, self.image_file)
        self.filenames = [self.filenames[i]
                          for i in np.where(self.split_mask)[0]]
        self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]
        self.idx_list = list(range(len(self.filenames)))

        
    def __len__(self):
        return len(self.idx_list)

    
    def __getitem__(self, index):
        
        # img = Image.open(self.filenames[index]).convert("RGB")
        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        
        gt_bboxes = self.gt_boxes[index] / self.box_scale * max(w, h)
        gt_classes = np.asarray(self.gt_classes[index])
        
        sketches, gt_labels = self.get_sketch_mappings(gt_classes, gt_bboxes)
        gt_labels, gt_bboxes, sketches = self.remove_nones(gt_labels, gt_bboxes, sketches)
        assert len(gt_labels) == len(gt_bboxes)
        assert len(gt_labels) == len(sketches)
        
        if len(gt_labels) > 0:
            sketch_images = self.draw_sketch_scene(sketches)
            attns = self.generate_attention_mtxs(gt_bboxes, (w, h))
            sketch_images, gt_labels, gt_bboxes, attns, padding_mask = self.pad_items(sketch_images, gt_labels, gt_bboxes, attns)
        else:
            idx = random.randint(0, len(self.idx_list)-1)
            sketch_images, gt_labels, gt_bboxes, attns, padding_mask, index = self.__getitem__(idx)
        
        return sketch_images, gt_labels, gt_bboxes, attns, padding_mask, index
    

def correct_img_info(img_dir, image_file):
    print("correct img info")
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in tqdm(range(len(data)), total=len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file.replace(".json", "_corrected.json"), 'w') as outfile:
        json.dump(data, outfile)
        
    
def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    
    # correct_img_info(img_dir, image_file)
    
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(roidb_file, split, num_val_im, box_scale):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_val_im: Number of validation images
    Return:
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
    """
    
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[: num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(box_scale)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, : 2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, : 2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]

    # Get everything by image.
    boxes = []
    gt_classes = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        
        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)

    return split_mask, boxes, gt_classes


def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    object_idx_to_name = sorted(class_to_ind, key=lambda k: class_to_ind[k])

    return object_idx_to_name