import cv2
import copy
import numpy as np
import json

from colorsys import hls_to_rgb

from .sketch_utils import relative_to_absolute, get_absolute_bounds, normalize_to_scale


def _rainbow_color_stops(n, end=3/4):
    if n <= 1: return [[0, 0, 1]]
    else: 
        outs = []
        for i in range(n):
            results = hls_to_rgb(end * i/(n-1), 0.5, 1)
            results = [round(num, 3) for num in results]
            outs.append(results)
        return outs
    
def draw_sketch(
    sketch: np.ndarray,
    # if given, requires 0 as start index and also the last sketch index
    division_begins: np.ndarray=None,
    class_ids=None,
    canvas_size=None,
    margin: int=10, # only applied when canvas_size is None
    is_absolute: bool=False,
    white_bg: bool=False,
    color_fg: bool=False, # gray / color
    shift: bool=False,
    scale_to: int=-1,
    save_path: str=None,
    use_for_matfile: bool=False,
    class_names=None,
    thickness=None
):
    
    if division_begins is not None:
        assert division_begins[0] == 0
        assert division_begins[-1] == sketch.shape[0]
    else:
        division_begins = [0, sketch.shape[0]]
        
    if thickness is None:
        thickness = 2

    # Convert to absolute
    abs_sketch = copy.deepcopy(sketch)
    if not is_absolute:
        abs_sketch = relative_to_absolute(abs_sketch)
        
    # Shift, scale, and set canvas size
    if shift:
        xmin, ymin, xmax, ymax = get_absolute_bounds(abs_sketch)
        abs_sketch[:, 0] -= xmin
        abs_sketch[:, 1] -= ymin
          
    if scale_to > 0:
        abs_sketch = normalize_to_scale(
            abs_sketch, is_absolute=True, scale_factor=scale_to - 2 * margin)
        
    if shift:
        abs_sketch[:, 0] += margin
        abs_sketch[:, 1] += margin
        
    if canvas_size is None:
        xmin, ymin, xmax, ymax = get_absolute_bounds(abs_sketch)
        # if no canvas size given, sketch is shifted to top-left
        # and visualized with margin on each side
        if scale_to is None:
            canvas_size = [int(xmax - xmin + 2 * margin), int(ymax - ymin + 2 * margin)]
        else:
            canvas_size = [scale_to, scale_to]

    elif type(canvas_size) in [int, float]:
        canvas_size = [int(canvas_size), int(canvas_size)]
    
    # Create the canvas
    bg_px = 255 if white_bg else 0
    if not use_for_matfile:
        canvas = np.full((canvas_size[1], canvas_size[0], 3), bg_px, dtype=np.uint8)
    else:
        canvas = np.full((canvas_size[1], canvas_size[0], 3), bg_px, dtype=np.uint16)
    
    # sets the division begins and numberof divisions
    if division_begins is not None:
        division_begins = np.asarray(sorted(list(set(division_begins))))
        num_divisions = division_begins.shape[0]
    else:
        num_divisions = 2
       
    # initializes the spectrum
    if color_fg and class_ids is None:
        spectrum = _rainbow_color_stops(num_divisions - 1)
        spectrum = (np.asarray(spectrum) * 255).astype(int).tolist()
    elif color_fg:
        spectrum = [[int(class_ids[cc]), int(class_ids[cc]), int(class_ids[cc])] for cc in range(len(class_ids))]
    else:
        chosen_color = [0, 0, 0] if white_bg else [255, 255, 255]
        spectrum = [chosen_color] * (num_divisions - 1)
    
    name_color_list = []
    for i in range(1, len(division_begins)):
        st, end = division_begins[i-1], division_begins[i]
        class_name = class_names[i-1] if class_names is not None else None
        fill_color = spectrum[i-1]
        for pnt in range(st+1, end):
            px, py = int(abs_sketch[pnt-1, 0]), int(abs_sketch[pnt-1, 1])
            x, y   = int(abs_sketch[pnt, 0]), int(abs_sketch[pnt, 1])
            pstate = abs_sketch[pnt-1, 2]
            nstate = abs_sketch[pnt, 2]
            
            if pstate < 0.5:
                canvas = cv2.line(
                    canvas, (px, py), (x, y), 
                    color=fill_color, thickness=thickness)
        
        if class_name is not None:
            name_color_list.append([class_name, fill_color])
            # canvas = cv2.putText(
            #     canvas, class_name, (int(abs_sketch[end-1, 0]), int(abs_sketch[end-1, 1])), 
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.9, fill_color, 1, cv2.LINE_AA)

    text_size, _ = cv2.getTextSize("0", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    H, W = canvas.shape[:2]
    for c_idx, (class_name, fill_color) in enumerate(name_color_list):
        ow = 45
        oh = (H - 15) - 30 * c_idx
        canvas = cv2.circle(canvas, (ow - 20, oh - text_size[1] // 2), 5, fill_color, -1)
        canvas = cv2.putText(
            canvas, class_name, (ow, oh), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 0], 2, cv2.LINE_AA)
    
    if save_path is not None:
        cv2.imwrite(save_path, canvas)
    
    return canvas, spectrum


def visualize_results(
    sketch: np.ndarray,
    division_begins: np.ndarray,
    class_names,
    canvas_size,
    colormap_json,
    is_absolute: bool=False,
    save_path: str=None
):
    if division_begins is not None:
        assert division_begins[0] == 0
        assert division_begins[-1] == sketch.shape[0]
        
    thickness = 2
    margin = 50
    scale_to = 800

    # Convert to absolute
    abs_sketch = copy.deepcopy(sketch)
    if not is_absolute:
        abs_sketch = relative_to_absolute(abs_sketch)     
        
    # Shift, scale, and set canvas size
    xmin, ymin, xmax, ymax = get_absolute_bounds(abs_sketch)
    abs_sketch[:, 0] -= xmin
    abs_sketch[:, 1] -= ymin
          
    abs_sketch = normalize_to_scale(
        abs_sketch, is_absolute=True, scale_factor=scale_to - 2 * margin)
        
    abs_sketch[:, 0] += margin
    abs_sketch[:, 1] += margin
          
    canvas_size = [int(scale_to), int(scale_to)]
    
    # Create the canvas
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 255, dtype=np.uint8)
    
    # sets the division begins and numberof divisions
    division_begins = np.asarray(sorted(list(set(division_begins))))
    num_divisions = division_begins.shape[0]
       
    with open(colormap_json, "r") as f:
        color_map = json.load(f)
        
    # print(color_map)
    # print(class_names)
    
    valid_class_set = set()
    for i in range(1, len(division_begins)):
        st, end = division_begins[i-1], division_begins[i]
        class_name = class_names[i-1]
        valid_class_set.add(class_name)
        fill_color = color_map[class_name]
        for pnt in range(st+1, end):
            px, py = int(abs_sketch[pnt-1, 0]), int(abs_sketch[pnt-1, 1])
            x, y   = int(abs_sketch[pnt, 0]), int(abs_sketch[pnt, 1])
            pstate = abs_sketch[pnt-1, 2]
            nstate = abs_sketch[pnt, 2]
            
            if pstate < 0.5:
                canvas = cv2.line(
                    canvas, (px, py), (x, y), 
                    color=fill_color, thickness=thickness)
            
        # scene_img = cv2.putText(
        #     canvas, class_name, (int(abs_sketch[end-1, 0]), int(abs_sketch[end-1, 1])), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.9, fill_color, 1, cv2.LINE_AA)
        
    text_size, _ = cv2.getTextSize("0", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    
    H, W = canvas.shape[:2]
    for c_idx, class_name in enumerate(valid_class_set):
        ow = 45
        oh = (H - 15) - 30 * c_idx
        canvas = cv2.circle(canvas, (ow - 20, oh - text_size[1] // 2), 5, color_map[class_name], -1)
        canvas = cv2.putText(
            canvas, class_name, (ow, oh), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 0], 2, cv2.LINE_AA)
            
    if save_path is not None:
        cv2.imwrite(save_path, canvas)
    
    return canvas

