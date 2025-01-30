import models
import torch
import os
import copy
import json
import cv2

from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import torch.nn.functional as F
from utils import setup, get_similarity_map, display_segmented_sketch
from vpt.launch import default_argument_parser
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import scipy.io            
            
def main(args):
    
    cfg = setup(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    Ours, preprocess = models.load("CS-ViT-B/16", device=device,cfg=cfg,zero_shot=False)
    state_dict = torch.load(args.checkpoint_path)
    # Trained on 2 gpus so we need to remove the prefix "module." to test it on a single GPU
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v 
    Ours.load_state_dict(new_state_dict)     
    Ours.eval()     
    
    mode = "test"
    sketch_path = os.path.join(args.data_path, f"{mode}/DRAWING_GT")
    sketch_imgs = []
    if not os.path.isdir(sketch_path):
        sketch_imgs.append(sketch_path)
    else:
        for file_ in os.listdir(sketch_path):
            if ".png" in file_:
                sketch_imgs.append(os.path.join(sketch_path, file_))
    
    for idx, sketch_img_path in enumerate(tqdm(sketch_imgs)):       
            
        file_name = sketch_img_path.split("/")[-1]
        img_id = file_name.replace(".png", "").replace("L0_sample", "")
            
        with open(os.path.join(args.data_path, f"{mode}/jsons", "sample_" + img_id + ".json"), "r") as f:
            data_dict = json.load(f)
        
        classes = data_dict["class_names"]
        
        for i in range(len(classes)):
            if "yoga" == classes[i]:
                classes[i] = "person"
        
        class_ids = data_dict["gt_classes"]
        
        colors = plt.get_cmap("tab20").colors
        if len(colors) < len(classes):
            colors = colors + colors
        classes_colors = colors[:len(classes)]
        
        pil_img = Image.open(sketch_img_path).convert('RGB')
        binary_sketch = np.array(pil_img)
        binary_sketch = np.where(binary_sketch>0, 255, binary_sketch)
        sketch_tensor = preprocess(copy.deepcopy(pil_img)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            text_features = models.encode_text_with_prompt_ensemble(Ours, classes, device,no_module=True)
            redundant_features = models.encode_text_with_prompt_ensemble(Ours, [""], device,no_module=True)            

        num_of_tokens = cfg.MODEL.PROMPT.NUM_TOKENS
    
        with torch.no_grad():
            sketch_features = Ours.encode_image(sketch_tensor,layers=12,text_features=text_features-redundant_features,mode=mode)
            sketch_features = sketch_features / sketch_features.norm(dim=1, keepdim=True)
        
        similarity = sketch_features @ (text_features - redundant_features).t()
        patches_similarity = similarity[0, num_of_tokens +1:, :]
        pixel_similarity = get_similarity_map(patches_similarity.unsqueeze(0),pil_img.size)
        pixel_similarity[pixel_similarity<args.threshold] = 0
        pixel_similarity_array = pixel_similarity.cpu().numpy().transpose(2,0,1)
        
        os.system(f"mkdir -p {args.output_path}")
        
        classes_dir = os.path.join(args.output_path, 'CLASS_PRED')
        if not os.path.isdir(classes_dir):
            os.mkdir(classes_dir)
        
        drawings_dir = os.path.join(args.output_path, 'DRAWING_PRED')
        if not os.path.isdir(drawings_dir):
            os.mkdir(drawings_dir)
        
        display_segmented_sketch(pixel_similarity_array,binary_sketch,classes,classes_colors,class_ids,img_id,save_path=args.output_path)
                              
        
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    args = default_argument_parser().parse_args()
    main(args)


