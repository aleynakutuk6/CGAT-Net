import math
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from relnet.utils.sketch_utils import *
from relnet.utils.visualize_utils import *

class CGATNetPreprocessor:
    
    def __init__(self, cfg: dict):
        self.task = "classification"
        self.margin_size = cfg["margin_size"]
        self.out_sketch_size = cfg["out_sketch_size"]
        self.color_images = cfg["color_images"] if "color_images" in cfg else False
        self.attn_maps = cfg["attn_maps"]
        self.calc_attn_with_self = cfg["calc_attn_with_self"] if "calc_attn_with_self" in cfg else True
        
        self.sketch_transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
            ]
        )
        self.global_ctr = 0
        self.max_pnt_cnts = 0
        self.max_num_objects = 0
    
    
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
        
        max_obj_cnt = divisions.shape[-1] - 1
        batch_visuals, batch_attns = [], []
        for b in range(stroke3.shape[0]):
            w, h = img_sizes[b, 0], img_sizes[b, 1]
            stroke_starts = divisions[b, :]
            scene = stroke3[b, ...]
            
            # os.system(f"mkdir -p demo/eval_vis_results/")
            # sketch_vis = self.draw_sketch(scene.numpy(), save_path=os.path.join("demo/eval_vis_results/", str(self.global_ctr) + "_scene.png"))
            
            pad_start = torch.where(stroke_starts < 0)[0]
            if len(pad_start) > 0:
                pad_start = pad_start[0]
            else:
                pad_start = stroke_starts.shape[0]
                
            # pnt_pad_start = torch.where(scene[:, -1] < 0)[0]
            # if len(pnt_pad_start) > 0:
            #     pnt_pad_start = pnt_pad_start[0]
            # else:
            #     pnt_pad_start = scene.shape[0]
            # self.max_pnt_cnts = max(self.max_pnt_cnts, pnt_pad_start)
            # print("Max num pnts:", self.max_pnt_cnts)
            
            # if no valid class is in a scene
            if pad_start == 0:
                sketch_visuals = torch.zeros(max_obj_cnt, 3, self.out_sketch_size, self.out_sketch_size)
                num_attn_maps = max(1, len(self.attn_maps))
                attns = torch.zeros(num_attn_maps, max_obj_cnt, max_obj_cnt)
                
            else:
                stroke_starts = stroke_starts[:pad_start]
                
                # self.max_num_objects = max(self.max_num_objects, len(stroke_starts) - 1)
                # print("Max num objs:", self.max_num_objects)
                
                point_starts = [0] + (torch.where(scene[..., -1] == 1)[0] + 1).tolist()
            
                sketch_visuals, boxes = [], []
                for str_start in range(1, len(stroke_starts)):
                    
                    start_str, end_str = stroke_starts[str_start - 1], stroke_starts[str_start]
                    start, end = point_starts[start_str], point_starts[end_str]
                    sketch = scene[start:end, ...].numpy()
                    boxes.append(get_absolute_bounds(sketch))
                    
                    # print(sketch)
                    # self.global_ctr += 1
                    # os.system(f"mkdir -p demo/eval_vis_results/")
                    sketch_vis = self.draw_sketch(sketch) #, save_path=os.path.join("demo/eval_vis_results/", str(self.global_ctr) + ".png"))
                    sketch_visuals.append(sketch_vis)   
                    
            
                attns = self.generate_attention_mtxs(np.asarray(boxes), (w, h))
                sketch_visuals, attns = self.pad_items(sketch_visuals, attns, max_obj_cnt)
            
            batch_visuals.append(sketch_visuals.tolist())
            batch_attns.append(attns.tolist())
        
        batch_visuals = torch.Tensor(batch_visuals)
        batch_attns = torch.Tensor(batch_attns)
        
        if no_batch_dim:
            batch_visuals = batch_visuals.squeeze(0)
            batch_attns = batch_attns.squeeze(0)
            
        return batch_visuals, batch_attns

    
    def pad_items(self, sketch_images: np.ndarray, attns: np.ndarray, max_obj_cnt: int): 
        
        diff_size = max(0, max_obj_cnt - len(sketch_images))
        sketch_images = torch.stack(sketch_images, dim=0)
        
        if diff_size > 0:
            # padding for sketch_images
            empty_images = torch.zeros(diff_size, 3, self.out_sketch_size, self.out_sketch_size)
            sketch_images = torch.cat([sketch_images, empty_images], dim=0)

            # padding for attns
            s, w, h = attns.shape
            new_attns = torch.zeros(s, max_obj_cnt, max_obj_cnt)
            new_attns[:, :w, :h] = attns[:, ...]
            
        elif diff_size < 0: # and self.split == "train":            
            sketch_images = sketch_images[:max_obj_cnt]

            # padding for attns
            new_attns = attns[:, :max_obj_cnt, :max_obj_cnt]
        
        else:
            # padding for attns
            new_attns = attns 

        return sketch_images, new_attns
    
    
    
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
    
    
    def generate_attention_mtxs(self, boxes, img_size):
        mtxs = []
        if 0 in self.attn_maps: # occlusion
            mtxs.append(self.generate_occlusion_mtx(boxes))
        if 1 in self.attn_maps: # distance
            mtxs.append(self.generate_distance_mtx(boxes, img_size))
        if 2 in self.attn_maps: # vertical directed distance
            mtxs.append(self.generate_vertical_distance_mtx(boxes, img_size))
        if 3 in self.attn_maps: # horizontal directed distance
            mtxs.append(self.generate_horizontal_distance_mtx(boxes, img_size))
        if 4 in self.attn_maps: # sizes ratio
            mtxs.append(self.generate_sizes_ratio_mtx(boxes, img_size))
        
        if len(mtxs) > 0:
            return torch.stack(mtxs)
        else:
            return torch.zeros(1, len(boxes), len(boxes))
    
    
    def generate_occlusion_mtx(self, boxes):
        
        occl_mtx = torch.zeros(len(boxes), len(boxes))
        
        if self.calc_attn_with_self:
            i_end = len(boxes)
            j_start = 0
        else:
            i_end = len(boxes) - 1
            j_start = 1
        
        for i in range(i_end):
            min_x_i, min_y_i, w_i, h_i = boxes[i]
            max_x_i = min_x_i + w_i
            max_y_i = min_y_i + h_i
            obj_area_i = w_i * h_i
            for j in range(i + j_start, len(boxes)):
                min_x_j, min_y_j, w_j, h_j = boxes[j]
                max_x_j = min_x_j + w_j
                max_y_j = min_y_j + h_j
                obj_area_j = w_j * h_j
                if max(min_x_i, min_x_j) < min(max_x_i, max_x_j):
                    int_w = min(max_x_i, max_x_j) - max(min_x_i, min_x_j)
                    int_h = min(max_y_i, max_y_j) - max(min_y_i, min_y_j)
                    int_area = int_w * int_h
                    occ_ratio_i = int_area / max(1, obj_area_i)
                    occ_ratio_j = int_area / max(1, obj_area_j)
                    occl_mtx[i, j] = occ_ratio_i
                    occl_mtx[j, i] = occ_ratio_j
                else:
                    occl_mtx[i, j] = 0
                    occl_mtx[j, i] = 0
                
        return occl_mtx
    
    
    def generate_distance_mtx(self, boxes, img_size):
        w, h = img_size
        img_diagonal = math.dist([0, 0], [w, h]) 
        dist_mtx = torch.zeros(len(boxes), len(boxes))
        
        if self.calc_attn_with_self:
            i_end = len(boxes)
            j_start = 0
        else:
            i_end = len(boxes) - 1
            j_start = 1
            
        for i in range(i_end):
            min_x_i, min_y_i, w_i, h_i = boxes[i]
            x_center_i = (w_i / 2) + min_x_i
            y_center_i = (h_i / 2) + min_y_i
            for j in range(i + j_start, len(boxes)):
                min_x_j, min_y_j, w_j, h_j = boxes[j]
                x_center_j = (w_j / 2) + min_x_j
                y_center_j = (h_j / 2) + min_y_j
                objs_diagonal = math.dist([x_center_i, y_center_i], [x_center_j, y_center_j])
                diagonal_ratio = (img_diagonal - objs_diagonal) / img_diagonal
                dist_mtx[i, j] = diagonal_ratio
                dist_mtx[j, i] = diagonal_ratio
                
        return dist_mtx
    
    
    def generate_vertical_distance_mtx(self, boxes, img_size):
        w, h = img_size
        dist_mtx = torch.zeros(len(boxes), len(boxes))
        
        if self.calc_attn_with_self:
            i_end = len(boxes)
            j_start = 0
        else:
            i_end = len(boxes) - 1
            j_start = 1
        
        for i in range(i_end):
            min_x_i, min_y_i, w_i, h_i = boxes[i]
            y_center_i = (h_i / 2) + min_y_i
            for j in range(i + j_start, len(boxes)):
                min_x_j, min_y_j, w_j, h_j = boxes[j]
                y_center_j = (h_j / 2) + min_y_j
                # can also be negative according to the direction
                dist = (h - abs(y_center_i - y_center_j)) / h
                i_j_sign = (y_center_i - y_center_j) / max(1, abs(y_center_i - y_center_j))
                dist_mtx[i, j] = i_j_sign * dist
                dist_mtx[j, i] = -1 * i_j_sign * dist
        
        return dist_mtx
               
    
    def generate_horizontal_distance_mtx(self, boxes, img_size):
        w, h = img_size
        dist_mtx = torch.zeros(len(boxes), len(boxes))
        
        if self.calc_attn_with_self:
            i_end = len(boxes)
            j_start = 0
        else:
            i_end = len(boxes) - 1
            j_start = 1
        
        for i in range(i_end):
            min_x_i, min_y_i, w_i, h_i = boxes[i]
            x_center_i = (w_i / 2) + min_x_i
            for j in range(i + j_start, len(boxes)):
                min_x_j, min_y_j, w_j, h_j = boxes[j]
                x_center_j = (w_j / 2) + min_x_j
                # can also be negative according to the direction
                dist = (w - abs(x_center_i - x_center_j)) / w
                i_j_sign = (x_center_i - x_center_j) / max(1, abs(x_center_i - x_center_j))
                dist_mtx[i, j] = i_j_sign * dist
                dist_mtx[j, i] = -1 * i_j_sign * dist

        return dist_mtx
    
    
    def generate_sizes_ratio_mtx(self, boxes, img_size):
        w, h = img_size
        img_area = w * h
        dist_mtx = torch.zeros(len(boxes), len(boxes))
        
        if self.calc_attn_with_self:
            i_end = len(boxes)
            j_start = 0
        else:
            i_end = len(boxes) - 1
            j_start = 1
            
        for i in range(i_end):
            min_x_i, min_y_i, w_i, h_i = boxes[i]
            area_i = w_i * h_i
            for j in range(i + j_start, len(boxes)):
                min_x_j, min_y_j, w_j, h_j = boxes[j]
                area_j = w_j * h_j
                dist_mtx[i, j] = (area_i / max(1, area_j)) / max(1, img_area)
                dist_mtx[j, i] = (area_j / max(1, area_i)) / max(1, img_area)
        
        return dist_mtx
