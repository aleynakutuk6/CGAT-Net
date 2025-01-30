import math
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from relnet.utils.sketch_utils import *
from relnet.utils.visualize_utils import *

class CAVTPreprocessor:
    
    def __init__(self, cfg: dict):
        self.task = "segmentation"
        # if true, crops the margins on left, right, top, and bottom
        self.crop_margins = cfg["crop_margins"] 
        # only used if crop_margins. The margin to leave before sketch begin
        self.side_pad = cfg["segm_side_padding"]
        # w, h dims of the output image. Cannot be None if crop_margins
        self.max_side_dim = cfg["max_side_dim"]
        # visualization-related parameters
        self.white_bg = cfg["white_bg"]
        self.color_fg = cfg["color_fg"]
        self.thickness = cfg["thickness"] if "thickness" in cfg else None
        
        assert not self.crop_margins or self.max_side_dim is not None
    
    def set_sketch_to_image_shape(self, sketch: np.ndarray, size: list):
        
        if self.crop_margins:
            sketch, (w, h) = shift_to_origin(sketch, is_absolute=True)

            if self.max_side_dim is not None:
                side_wo_margin = self.max_side_dim - 2 * self.side_pad
                if w > h:
                    scale_ratio = side_wo_margin / w
                    nw, nh = self.max_side_dim, int(h * scale_ratio) + 2 * self.side_pad
                else:
                    scale_ratio = side_wo_margin / h
                    nh, nw = self.max_side_dim, int(w * scale_ratio) + 2 * self.side_pad

                sketch = normalize_to_scale(
                    sketch, is_absolute=True, scale_ratio=scale_ratio)

            else:
                nw = int(w + 2 * self.side_pad)
                nh = int(h + 2 * self.side_pad)

            sketch[:, 0] += self.side_pad
            sketch[:, 1] += self.side_pad
        
        elif self.max_side_dim is None:
            nw, nh = size

        else:
            raise ValueError("max_side_dim cannot be given if not crop_margins is enabled!")

        return sketch, (nw, nh)
    
    def get_stroke_begin_pair_areas(self, abs_sketch, stroke_begins=None):

        if stroke_begins is None:
            stroke_begins = [0] + (np.where(abs_sketch[:,-1] == 1.0)[0] + 1).tolist()
        
        stroke_areas, stroke_area_inds = [], []
        for i in range(len(stroke_begins)-1):
            for j in range(i+1, len(stroke_begins)):
                st, end = stroke_begins[i], stroke_begins[j]
                xmin, ymin, xmax, ymax = get_absolute_bounds(abs_sketch[st:end, :])
                stroke_areas.append([xmin, ymin, xmax, ymax])
                stroke_area_inds.append([st, end])
        
        return stroke_areas, stroke_area_inds
    
    
    def __call__(self, stroke3: torch.Tensor, divisions: torch.LongTensor, img_sizes: torch.Tensor):
        """
        * stroke3 -> B x S x 3 with padding
        * divisions -> B x (max_obj_count + 1) with padding 
        * img_sizes -> B x 2 (scene image sizes)
        """
        no_batch_dim = len(stroke3.shape) == 2
        
        if no_batch_dim:
            stroke3 = stroke3.unsqueeze(0)
            divisions = divisions.unsqueeze(0)
            img_sizes = img_sizes.unsqueeze(0)
            
        assert stroke3.shape[0] == 1
        
        batch_images, batch_boxes, batch_sizes = [], [], []
        batch_stroke_areas, batch_stroke_inds = [], []
        for b in range(stroke3.shape[0]):
            w, h = img_sizes[b, 0], img_sizes[b, 1]
            division = divisions[b, :]
            scene = stroke3[b, ...]
            
            pad_start = torch.where(scene[:, -1] < 0)[0]
            pad_start = pad_start[0] if len(pad_start) > 0 else scene.shape[0]
            sketch = copy.deepcopy(scene[:pad_start, ...])
            sketch, (W, H) = self.set_sketch_to_image_shape(sketch.numpy(), (w, h))
            
            image, stroke_begins = self.draw_sketch(sketch, W, H)
            batch_images.append(image)
            batch_sizes.append([W, H])
            
            stroke_areas, stroke_area_inds = self.get_stroke_begin_pair_areas(
                sketch, stroke_begins)
            
            batch_stroke_areas.append(stroke_areas)
            batch_stroke_inds.append(stroke_area_inds)

            bboxes = []
            for i in range(1, division.shape[0]):
                start_str, end_str = division[i - 1], division[i]
                start, end = stroke_begins[start_str], stroke_begins[end_str]
                xmin, ymin, xmax, ymax = get_absolute_bounds(sketch[start:end, :])
                bboxes.append([xmin, ymin, xmax, ymax])
            
            batch_boxes.append(bboxes)
        
        batch_images = torch.stack(batch_images, dim=0)
        batch_boxes = torch.Tensor(batch_boxes)
        batch_stroke_areas = torch.Tensor(batch_stroke_areas)
        batch_stroke_inds = torch.Tensor(batch_stroke_inds)
        batch_sizes = torch.LongTensor(batch_sizes)

        if no_batch_dim:
            batch_images = batch_images.squeeze(0)
            batch_boxes = batch_boxes.squeeze(0)
            batch_stroke_areas = batch_stroke_areas.squeeze(0)
            batch_stroke_inds = batch_stroke_inds.squeeze(0)
            batch_sizes = batch_sizes.squeeze(0)
            
        return batch_images, batch_boxes, batch_stroke_areas, batch_stroke_inds, batch_sizes
        
        
    def draw_sketch(self, sketch, W, H, save_path=None):
        
        sketch_divisions = [0] + (np.where(sketch[:, -1] == 1)[0] + 1).tolist()
        
        sketch_img, _ = draw_sketch(
                sketch, 
                division_begins=sketch_divisions,
                canvas_size=[W, H],
                is_absolute=True,
                white_bg=self.white_bg,
                color_fg=self.color_fg,
                save_path=None,
                thickness=self.thickness)

        sketch_img = torch.LongTensor(sketch_img)

        return sketch_img, sketch_divisions
    
    def create_class_segmentation_mtx(self, scene_strokes, obj_divisons, class_ids, W, H):
        class_arr, _ = draw_sketch(        
            scene_strokes,
            obj_divisons,
            class_ids,
            canvas_size=[W, H], 
            margin=0,
            white_bg=False,
            color_fg=self.color_fg,
            shift=False,
            scale_to=-1,
            is_absolute=True, 
            thickness=self.thickness)
        
        # class_arr, _ = draw_sketch(        
        #     scene_strokes,
        #     obj_divisons,
        #     class_ids,
        #     canvas_size=[512, 512], 
        #     margin=50,
        #     white_bg=False,
        #     color_fg=self.color_fg,
        #     shift=True,
        #     scale_to=412,
        #     is_absolute=True)
        
        return class_arr[:,:,0]