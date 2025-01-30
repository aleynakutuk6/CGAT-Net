import math
import torch
import numpy as np

from mmdet.apis import init_detector, inference_detector
from relnet.utils.bbox_utils import intersection_over_union, intersection_over_first

class CAVT(torch.nn.Module):
    
    def __init__(self, cfg: dict, device):
        super().__init__()
        self.model = init_detector(
            cfg["mmdet_cfg"],
            cfg["weight"],
            device=device)
        self.device = device
        # self.keep_small_percent = cfg["keep_small_percent"]
        self.simple_eval = cfg["simple_eval"] if "simple_eval" in cfg else False
        
        self.OR_thold = cfg["OR_thold"]
        self.num_repeats = cfg["num_repeats"]
        self.select_larger_thold = cfg["select_larger_thold"]
        self.IoU_thold = cfg["IoU_thold"]


    def forward_only_model(self, scene_visuals: torch.Tensor):
        assert scene_visuals.shape[0] == 1
        result = inference_detector(
            self.model, 
            scene_visuals[0, ...].cpu().numpy().astype(np.uint8))
        scores = result.pred_instances.scores.unsqueeze(0).cpu()
        bboxes = result.pred_instances.bboxes.unsqueeze(0).cpu()
        labels = result.pred_instances.labels.unsqueeze(0).cpu()
        # print(bboxes)
        # print(scores)
        # print(labels)
        return scores, bboxes
        
    
    def forward(
        self, 
        scene_visuals: torch.Tensor, 
        scene_strokes: torch.Tensor,
        stroke_areas: torch.Tensor, 
        stroke_area_inds: torch.Tensor,
        segmentation_sizes: torch.Tensor,
        return_pnt_level_ranges: bool=False,
    ):
        scores, bboxes = self.forward_only_model(scene_visuals)
        return self.postprocess(
            scores, bboxes, scene_strokes, 
            stroke_areas, stroke_area_inds, 
            segmentation_sizes, return_pnt_level_ranges)


    def postprocess(
        self, 
        b_scores, b_boxes, scene_strokes, 
        stroke_areas, stroke_area_inds, 
        segmentation_sizes, 
        return_pnt_level_ranges: bool=False):
        
        out_scores, out_boxes, out_ranges = [], [], []
        
        for b in range(b_scores.shape[0]):
            scores = b_scores[b, ...]
            boxes  = b_boxes[b, ...]
            ranges = None
            W, H   = segmentation_sizes[b, :].cpu()
            stroke_areas = stroke_areas[b, :].cpu()
            stroke_area_inds = stroke_area_inds[b, :].cpu()
            the_same, total_iter = False, 0

            if self.simple_eval:
                scores, boxes, ranges = self.simple_stroke_assigning_to_boxes(
                    boxes, stroke_areas, stroke_area_inds
                )
                
                if not return_pnt_level_ranges:
                    stroke_end_locs = torch.where(
                        scene_strokes[b, :, -1] == 1
                    )[0].cpu().numpy().tolist()
                    
                    pnt_to_stroke_dict = {0: 0}
                    for s_idx, sel in enumerate(stroke_end_locs):
                        pnt_to_stroke_dict[sel+1] = s_idx+1
 
                    ranges = [pnt_to_stroke_dict[int(num)] for num in ranges]
                
                out_scores.append(scores)
                out_boxes.append(boxes)
                out_ranges.append(ranges)
            else:
                while not the_same and total_iter < self.num_repeats: # iterate until convergence
                    the_same = True
                    total_iter += 1
                    # start from the smallest box to largest
                    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    box_area_sort = torch.argsort(box_areas)
                    boxes = boxes[box_area_sort, :]
                    scores = scores[box_area_sort]
                    # keep track of the covered strokes
                    covered_strokes = np.zeros((int(stroke_area_inds[-1, -1].data.item())))
                    # make the first assignment


                    new_scores, new_boxes, new_ranges = self.assign_strokes_to_closest_boxes(
                        boxes, scores, covered_strokes, stroke_areas, stroke_area_inds)

                    self.assign_non_classified_strokes_to_neighbors(
                        new_boxes, new_ranges, covered_strokes, stroke_areas, stroke_area_inds)

                    # self.remove_small_boxes(new_scores, new_boxes, new_ranges, W, H)

                    new_scores, new_boxes, new_ranges, covered_strokes = self.define_objects_from_non_classified_strokes(
                        new_scores, new_boxes, new_ranges, covered_strokes, stroke_areas, stroke_area_inds)

                    the_same = self.check_changes(ranges, new_ranges)
                    boxes  = new_boxes
                    scores = new_scores
                    ranges = new_ranges

                    if the_same: break 

                out_boxes.append(boxes.cpu().numpy().tolist())
                out_scores.append(scores.cpu().numpy().tolist())

                order = torch.argsort(ranges[:, 0])
                ranges = ranges[order, :]
                out_range = ranges[:, 0].cpu().numpy().tolist() + [ranges[-1, -1].data.item()]
                
                pad_start = torch.where(scene_strokes[b, :, -1] < 0)[0]
                pad_start = pad_start[0] if len(pad_start) > 0 else scene_strokes.shape[-2]
                if pad_start not in out_range:
                    out_range = out_range + [pad_start]
                if out_range[0] != 0:
                    out_range = [0] + out_range

                if not return_pnt_level_ranges:
                    stroke_end_locs = torch.where(
                        scene_strokes[b, :, -1] == 1
                    )[0].cpu().numpy().tolist()
                    
                    pnt_to_stroke_dict = {0: 0}
                    for s_idx, sel in enumerate(stroke_end_locs):
                        pnt_to_stroke_dict[sel+1] = s_idx+1
 
                    out_range = [pnt_to_stroke_dict[int(num)] for num in out_range]
                
                out_ranges.append(out_range)

        return (
            torch.Tensor(out_scores).to(self.device),
            torch.Tensor(out_boxes).to(self.device),
            torch.Tensor(out_ranges).to(self.device))
        
        
    def simple_stroke_assigning_to_boxes(
        self, 
        boxes: torch.Tensor,
        stroke_areas: torch.Tensor,
        stroke_area_inds: torch.Tensor):

        if boxes is None or boxes.shape[0] == 0:
            out_scores = [-1]
            out_ranges = [0, int(stroke_area_inds[-1, 1].data.item())]
            for i in range(stroke_areas.shape[0]):
                st, end = stroke_area_inds[i]
                if st == 0 and end == stroke_area_inds[-1, 1]:
                    out_boxes = [stroke_areas[i].cpu().numpy().tolist()]

            return out_scores, out_boxes, out_ranges
        
        covered_strokes = np.zeros((int(stroke_area_inds[-1, -1].data.item()),))

        obj_begins = set(stroke_area_inds[:,0].cpu().numpy().tolist())
        obj_begins = list(obj_begins) + [stroke_area_inds[-1, 1].data.item()]
        obj_begins = [int(num) for num in obj_begins]
        obj_begins.sort()

        obj_begin_dict = {}
        for o_idx, o in enumerate(obj_begins):
            obj_begin_dict[int(o)] = int(o_idx)

        s_area_dict, s_total_dict = {}, {}
        for s in range(stroke_areas.shape[0]):
            st, end = stroke_area_inds[s].cpu().numpy().astype(int).tolist()
            if st not in s_area_dict:
                s_area_dict[st] = [stroke_areas[s], s, end]
            elif s_area_dict[st][-1] > end:
                s_area_dict[st] = [stroke_areas[s], s, end]

            if st not in s_total_dict:
                s_total_dict[st] = {}
            s_total_dict[st][end] = stroke_areas[s].cpu().numpy().tolist()
        
        iou = intersection_over_union(boxes, stroke_areas)
        b_cx = (boxes[:, 2] + boxes[:, 0]) / 2
        b_cy = (boxes[:, 3] + boxes[:, 1]) / 2

        # print(s_total_dict)
        # print(s_area_dict)
        # print(stroke_area_inds)

        for i in range(len(obj_begins) - 1):
            s = obj_begins[i]
            s_area, s_idx, end = s_area_dict[s]
            iou_stroke = iou[:, s_idx]
            max_box_id = torch.argmax(iou_stroke)
            if iou_stroke[max_box_id] > 0:
                covered_strokes[s:end] = max_box_id
            else:
                s_cx = (s_area[2] + s_area[0]) / 2
                s_cy = (s_area[3] + s_area[1]) / 2
                dists =  (b_cx - s_cx) * (b_cx - s_cx) + (b_cy - s_cy) * (b_cy - s_cy)
                closest_box_id = torch.argmin(dists)
                covered_strokes[s:end] = closest_box_id

        out_scores, out_boxes, out_ranges = [], [], []
        s_prev = 0
        for s in range(1, covered_strokes.shape[0]):
            if covered_strokes[s-1] != covered_strokes[s]:
                out_scores.append(-1)
                out_boxes.append(s_total_dict[s_prev][s])
                out_ranges.append(s_prev)
                s_prev = s
    
        out_scores.append(-1)
        out_boxes.append(s_total_dict[s_prev][covered_strokes.shape[0]])
        out_ranges.append(s_prev)
        out_ranges.append(covered_strokes.shape[0])

        return out_scores, out_boxes, out_ranges


    def assign_strokes_to_closest_boxes(
        self, 
        boxes: torch.Tensor, 
        scores: torch.Tensor,
        covered_strokes: torch.Tensor,
        stroke_areas: torch.Tensor,
        stroke_area_inds: torch.Tensor):

        new_boxes, new_scores, new_ranges = [], [], []
        iou = intersection_over_union(boxes, stroke_areas)
        for b in range(boxes.shape[0]):
            max_iou, sel_coords, sel_rng = -1, None, None
            for s in range(stroke_areas.shape[0]):
                iou_val = iou[b, s]
                st, end = stroke_area_inds[s, :]
                st, end = int(st), int(end)
                if max_iou - iou_val <= self.select_larger_thold and covered_strokes[st:end].sum() == 0:
                    if sel_rng is None or iou_val - max_iou > self.select_larger_thold:
                        max_iou = iou_val
                        sel_coords = stroke_areas[s, :]
                        sel_rng = [st, end]
                    elif sel_rng[1] - sel_rng[0] < end - st:
                        max_iou = iou_val
                        sel_coords = stroke_areas[s, :]
                        sel_rng = [st, end]

            if max_iou > self.IoU_thold:
                new_boxes.append(sel_coords.tolist())
                new_scores.append(scores[b].data.item())
                new_ranges.append(sel_rng)
                covered_strokes[sel_rng[0]:sel_rng[1]] = 1.0

        return torch.Tensor(new_scores), torch.Tensor(new_boxes), torch.Tensor(new_ranges)


    def assign_non_classified_strokes_to_neighbors(
        self,
        new_boxes: torch.Tensor,
        new_ranges: torch.Tensor,
        covered_strokes: torch.Tensor,
        stroke_areas: torch.Tensor,
        stroke_area_inds: torch.Tensor
    ):
        if new_boxes.shape[0] == 0:
            return
            
        changed = True
        while changed:
            changed = False
            io_first = intersection_over_first(stroke_areas, new_boxes)
            for s in range(stroke_area_inds.shape[0]):
                st, end = stroke_area_inds[s, :]
                st, end = int(st), int(end)
                xmin, ymin, xmax, ymax = stroke_areas[s, :]
                if covered_strokes[st:end].sum() != 0.0: continue
                
                box_opts = []
                for b in range(new_boxes.shape[0]):
                    rngs = new_ranges[b, :]
                    if rngs[0] != end and rngs[1] != st: continue # has to be adjacent to assigned strokes
                    iof = io_first[s, b]
                    if iof > self.OR_thold or (
                        iof > 0 and max(xmax - xmin, ymax - ymin) / min(xmax - xmin, ymax - ymin) > 5):
                        # either the overlap of the stroke object has to be large
                        # or the shape of the stroke object has to be too long or too wide
                        # (like table legs)
                        box_opts.append([b, (xmax - xmin) * (ymax - ymin)])
                    
                if len(box_opts) > 0:
                    # if at least a box is found
                    changed = True
                    min_idx, min_area = None, 10000000000
                    for idx, area in box_opts:
                        if area < min_area:
                            min_area = area
                            min_idx = idx
                    covered_strokes[st:end] = 1.0
                    new_ranges[min_idx, 0] = min(st, new_ranges[min_idx, 0])
                    new_ranges[min_idx, 1] = max(end, new_ranges[min_idx, 1])
                    new_boxes[min_idx, 0] = min(xmin, new_boxes[min_idx, 0])
                    new_boxes[min_idx, 1] = min(ymin, new_boxes[min_idx, 1])
                    new_boxes[min_idx, 2] = max(xmax, new_boxes[min_idx, 2])
                    new_boxes[min_idx, 3] = max(ymax, new_boxes[min_idx, 3])


    def remove_small_boxes(self, new_scores, new_boxes, new_ranges, W, H):
        
        if new_boxes.shape[0] == 0:
            return
        
        small_boxes = []
        for b in range(new_boxes.shape[0]-1, -1, -1):
            w = new_boxes[b, 2] - new_boxes[b, 0]
            h = new_boxes[b, 3] - new_boxes[b, 1]
            if min(w, h) / min(W, H) < self.keep_small_percent: 
                small_boxes.append(b)
        
        for el in small_boxes:
            if el == 0:
                new_boxes = new_boxes[el+1:, :]
                new_scores = new_scores[el+1:]
                new_ranges = new_ranges[el+1:, :]
            elif el == new_boxes.shape[0] - 1:
                new_boxes = new_boxes[:el, :]
                new_scores = new_scores[:el]
                new_ranges = new_ranges[:el, :]
            else:
                new_boxes = torch.cat([new_boxes[:el, :], new_boxes[el+1:, :]], dim=0)
                new_scores = torch.cat([new_scores[:el], new_scores[el+1:]])
                new_ranges = torch.cat([new_ranges[:el, :], new_ranges[el+1:, :]], dim=0)


    def define_objects_from_non_classified_strokes(
        self, new_scores, new_boxes, new_ranges, covered_strokes, stroke_areas, stroke_area_inds):

        st = -1
        for s in range(covered_strokes.shape[0] + 1):
            
            if s < len(covered_strokes) and covered_strokes[s] == 0 and st < 0:
                st = s
            elif st >= 0 and (s == covered_strokes.shape[0] or covered_strokes[s] == 1):
                for si in range(stroke_area_inds.shape[0]):
                    if stroke_area_inds[si, 0] == st and stroke_area_inds[si, 1] == s:
                        if new_boxes.shape == 0:
                            new_boxes = stroke_areas[si:si+1, :]
                            new_ranges = stroke_area_inds[si:si+1, :]
                            new_scores = torch.full((1,), fill_value=-1)
                        else:
                            new_boxes = torch.cat([new_boxes, stroke_areas[si:si+1, :]], dim=0)
                            new_ranges = torch.cat([new_ranges, stroke_area_inds[si:si+1, :]], dim=0)
                            new_scores = torch.cat([new_scores, torch.full((1, ), fill_value=-1)])
                        covered_strokes[st:s] = 1
                        st = -1
                        break
                        
        return new_scores, new_boxes, new_ranges, covered_strokes

    def check_changes(self, ranges: torch.Tensor, new_ranges: torch.Tensor):
        if ranges is None: return False
        if ranges.shape[0] != new_ranges.shape[0]: return False
        rng_list = ranges.cpu().numpy().astype(int).tolist()  
        new_rng_list = new_ranges.cpu().numpy().astype(int).tolist()  
        for val in rng_list:
            if val not in new_rng_list:
                return False
        return True
