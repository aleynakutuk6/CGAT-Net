import copy
import numpy as np
import os
import random
import math

from rdp import rdp

def get_relative_bounds(data: np.ndarray) -> list:
    min_x, max_x, min_y, max_y = 10000000, 0, 10000000, 0
    abs_x, abs_y = 0, 0
    
    for i in range(data.shape[0]):
        x = float(data[i, 0])
        y = float(data[i, 1])
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return min_x, min_y, max_x, max_y


def get_absolute_bounds(data: np.ndarray) -> list:
    min_x, max_x, min_y, max_y = 10000000, 0, 10000000, 0
    
    for i in range(data.shape[0]):
        x = float(data[i, 0])
        y = float(data[i, 1])
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    return min_x, min_y, max_x, max_y


def relative_to_absolute(sketch: np.ndarray) -> np.ndarray:
    absolute_sketch = np.zeros_like(sketch)
    absolute_sketch[0, :] = sketch[0, :]
    for i in range(1, sketch.shape[0]):
        absolute_sketch[i, 0] = absolute_sketch[i-1, 0] + sketch[i, 0]
        absolute_sketch[i, 1] = absolute_sketch[i-1, 1] + sketch[i, 1]
        absolute_sketch[i, 2:] = sketch[i, 2:]
    return absolute_sketch


def absolute_to_relative(sketch: np.ndarray, start_from_zero: bool=True) -> np.ndarray:
    """ returns a relative sketch with the start point as [0, 0] """
    relative_sketch = np.zeros_like(sketch)
    for i in range(1, sketch.shape[0]):
        relative_sketch[i, :2] = sketch[i, :2] - sketch[i-1, :2]
        relative_sketch[i, 2:] = sketch[i, 2:]
    if not start_from_zero:
        relative_sketch[0, :2] = sketch[0, :2]
    return relative_sketch


def stroke3_to_strokelist(sketch: np.ndarray, is_absolute: bool=False) -> list:
    """ Makes conversion:
    - From: stroke-3 format 
    - To: [[[x1, y1], [x2, y2], ...], [[x3, y3], [x4, y4], ...], ...]
    """
    if not is_absolute:
        abs_sketch = relative_to_absolute(copy.deepcopy(sketch))
    else:
        abs_sketch = sketch
    
    strokes, points = [], []
    for i in range(sketch.shape[0]):
        points.append([abs_sketch[i, 0], abs_sketch[i, 1]])
        if abs_sketch[i, 2] == 1:
            strokes.append(points)
            points = []
    
    return strokes


def strokelist_to_stroke3(strokelist: list, return_is_absolute: bool=False) -> np.ndarray:
    """ Makes conversion:
    - From: [[[x1, y1], [x2, y2], ...], [[x3, y3], [x4, y4], ...], ...]
    - To: stroke-3 format 
    """
    stroke3 = []
    for stroke in strokelist:
        for x, y in stroke:
            stroke3.append([x, y, 0.0])
        stroke3[-1][-1] = 1.0
    
    stroke3 = np.asarray(stroke3)
    
    if not return_is_absolute:
        stroke3 = absolute_to_relative(stroke3)
        
    return stroke3


def sep_strokelist_to_stroke3(strokelist: list, return_is_absolute: bool=False) -> np.ndarray:
    """ Makes conversion:
    - From: [[[x1, x2, ...], [y1, y2, ...]], [[x3, x4, ...], [y3, y4, ...]], ...]
    - To: stroke-3 format 
    """
    stroke3 = []
    for stroke in strokelist:
        for x, y in zip(stroke[0], stroke[1]):
            stroke3.append([x, y, 0.0])
        stroke3[-1][-1] = 1.0
    
    stroke3 = np.asarray(stroke3)
    
    if not return_is_absolute:
        stroke3 = absolute_to_relative(stroke3)
        
    return stroke3
        
      
def rotate_sketch(sketch: np.ndarray, angle: int, is_absolute: bool=False) -> np.ndarray:
    assert angle >= 0 and angle <= 360
    
    def rotate(p, origin=(0, 0), degrees=0):
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T-o.T) + o.T).T)
    
    if angle == 0 or angle == 360:
        return sketch
    elif angle < 0 or angle > 360:
        angle = angle % 360
    
    if not is_absolute:
        abs_sketch = relative_to_absolute(sketch)
    else:
        abs_sketch = copy.deepcopy(sketch)
    
    xmin, ymin, xmax, ymax = get_absolute_bounds(abs_sketch)
    cx, cy = (xmax + xmin) / 2, -(ymax + ymin) / 2
    
    for i in range(abs_sketch.shape[0]):
        ox, oy = abs_sketch[i, 0], - abs_sketch[i, 1] 
        new_p = rotate([ox, oy], (cx, cy), degrees=angle)
        abs_sketch[i, 0] = new_p[0]
        abs_sketch[i, 1] = -new_p[1]
    
    xmin_new, ymin_new, xmax_new, ymax_new = get_absolute_bounds(abs_sketch)
    # shift the sketch so that the center remains the same with the initial
    for i in range(abs_sketch.shape[0]):
        abs_sketch[i, 0] = abs_sketch[i, 0] - xmin_new
        abs_sketch[i, 1] = abs_sketch[i, 1] - ymin_new

    if not is_absolute:
        return absolute_to_relative(abs_sketch)
    else:
        return abs_sketch
    


def normalize_to_scale(
    sketch: np.ndarray, is_absolute: bool=False, 
    scale_factor: float=1.0, scale_ratio: float=None) -> np.ndarray:
    
    if is_absolute: bounds = get_absolute_bounds(sketch)
    else: bounds = get_relative_bounds(sketch)  
    max_dimension = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    if scale_ratio is not None:
        sketch[:, 0:2] = sketch[:, 0:2] * scale_ratio
    else:
        if max_dimension <= 0.0:
            return None
        else:
            sketch[:, 0:2] = (sketch[:, 0:2] * scale_factor) / max_dimension
    return sketch


def normalize(sketch: np.ndarray, is_absolute: bool=False) -> np.ndarray:
    return normalize_to_scale(sketch, is_absolute=is_absolute, scale_factor=1.0)


def apply_RDP(sketch: np.ndarray, is_absolute: bool=False, epsilon: float=2.0):
    # function takes stroke-3 format in relative coordinates
    
    if is_absolute:
        rel_sketch = absolute_to_relative(sketch, start_from_zero=False)
    else:
        rel_sketch = sketch
    
    los = stroke3_to_strokelist(rel_sketch)
    new_lines = []
    for stroke in los:
        simplified_stroke = rdp(stroke, epsilon=epsilon)
        if len(simplified_stroke) > 1:
            new_lines.append(simplified_stroke)
    
    simplified_sketch = strokelist_to_stroke3(new_lines, return_is_absolute=is_absolute)
    return simplified_sketch


def split_scene_to_parts(scene: np.ndarray, split_inds: np.ndarray) -> list:
    # function takes the scene as stroke-3 format in relative coordinates
    # split_inds as an np.ndarray of part begin indices
    
    assert len(split_inds.shape) == 1 and len(scene.shape) == 2
    
    obj_sketches = []
    for b in range(split_inds.shape[0]):
        start = int(split_inds[b])
        end = int(split_inds[b + 1]) if b != split_inds.shape[0] - 1 else int(scene.shape[0])
        obj_sketches.append(scene[start:end, :])
        
    return obj_sketches
    

def lines_to_strokes(lines):
    """Convert polyline format to stroke-3 format."""
    eos = 0
    strokes = [[0, 0, 0]]
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :]

def shift_to_origin(sketch: np.ndarray, is_absolute: bool=False):
    """
    Shifts the sketch's topleft side to origin (0, 0). Returns both
    the sketch and its W x H dimensions.
    """
    if is_absolute:
        xmin, ymin, xmax, ymax = get_absolute_bounds(sketch)
        sketch[:, 0] -= xmin
        sketch[:, 1] -= ymin
    else:
        xmin, ymin, xmax, ymax = get_relative_bounds(sketch)
        sketch[0, 0] -= xmin
        sketch[0, 1] -= ymin
    return sketch, [xmax - xmin, ymax - ymin]


def read_npy(filepath: str):
    
    if not os.path.isfile(filepath):
        raise ValueError(f"No NPY file exists in: {filepath}")
    
    sk = np.load(filepath, allow_pickle=True, encoding="bytes")
    
    def clean_sketch(sketch: np.ndarray) -> np.ndarray:
        sk_new = sketch.astype(float)
        if sk_new[0, -1] == 1:
            sk_new = sk_new[1:, :]
        
        end_checks = sk_new[1:, -1] + sk_new[:-1, -1]
        vals = np.where(end_checks > 1)[0]
        if vals.shape[0] > 0:
            for j in np.flip(vals):
                sk_new = np.delete(sk_new, j+1, axis=0)
        
        return sk_new

    return clean_sketch(sk)