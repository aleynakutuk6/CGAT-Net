import math
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from relnet.utils.sketch_utils import *
from relnet.utils.visualize_utils import *

class SingleSketchPreprocessor:
    
    def __init__(self, cfg: dict):
        self.task = "classification"
        self.margin_size = cfg["margin_size"]
        self.out_sketch_size = cfg["out_sketch_size"]
        self.color_images = cfg["color_images"] if "color_images" in cfg else False
        
        self.sketch_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ]
        )
    
    
    def __call__(self, stroke3: torch.Tensor):
        """
        * stroke3 -> B x S x 3 with padding
        """
        
        no_batch_dim = len(stroke3.shape) == 2
        if no_batch_dim:
            stroke3 = stroke3.unsqueeze(0)
        
        batch_visuals = []
        for b in range(stroke3.shape[0]):
            scene = stroke3[b, ...]
            pad_start = torch.where(scene[:, -1] < 0)[0]
            if len(pad_start) > 0:
                pad_start = pad_start[0]
            else:
                pad_start = scene.shape[0]
            
            # if no valid class is in a scene
            if pad_start == 0:
                sketch_vis = torch.zeros(max_obj_cnt, 3, self.out_sketch_size, self.out_sketch_size)
            else:
                sketch_vis = self.draw_sketch(scene[:pad_start, :])
            
            batch_visuals.append(sketch_vis)    
        
        batch_visuals = torch.stack(batch_visuals, dim=0)
        
        if no_batch_dim:
            batch_visuals = batch_visuals.squeeze(0)
        
        return batch_visuals
    
    
    def draw_sketch(self, sketch, save_path=None):
        
        sketch_divisions = [0] + (np.where(sketch[..., -1] == 1)[0] + 1).tolist()
        
        sketch_img, _ = draw_sketch(
            np.asarray(sketch).astype(float),
            sketch_divisions,
            margin=self.margin_size,
            scale_to=self.out_sketch_size - (2 * self.margin_size),
            is_absolute=True,
            color_fg=self.color_images,
            white_bg=True,
            shift=True,
            canvas_size=self.out_sketch_size,
            save_path=save_path)
    
        sketch_img = Image.fromarray(cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB))
        sketch_img = self.sketch_transforms(sketch_img)

        return sketch_img