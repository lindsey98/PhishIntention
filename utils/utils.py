import json
import torch
import cv2
import numpy as np
import os
import math


def coord_reshape(coords, image_shape, reshaped_size=(256, 512)):
    '''
    Revise coordinates when the image is resized
    '''
    height, width = image_shape
    new_coords = []
    for c in coords:
        x1, y1, x2, y2 = c
        x1n, y1n, x2n, y2n = reshaped_size[1] * x1 / width, reshaped_size[0] * y1 / height, \
                             reshaped_size[1] * x2 / width, reshaped_size[0] * y2 / height
        new_coords.append([x1n, y1n, x2n, y2n])

    return np.asarray(new_coords)


def coord2pixel_reverse(img_path, coords, types, num_types=5, reshaped_size=(256, 512)) -> torch.Tensor:
    '''
    Convert coordinate to multi-hot encodings for coordinate class
    '''
    img = cv2.imread(img_path) if not isinstance(img_path, np.ndarray) else img_path
    coords = coords.numpy() if not isinstance(coords, np.ndarray) else coords
    coords = coord_reshape(coords, img.shape[:2], reshaped_size)  # reshape coordinates
    types = types.numpy() if not isinstance(types, np.ndarray) else types

    # Incorrect path/empty image
    if img is None:
        raise AttributeError('Image is None')
    height, width = img.shape[:2]
    # Empty image
    if height == 0 or width == 0:
        raise AttributeError('Empty image')

    # grid array of shape ClassxHxW
    grid_arrs = np.zeros((num_types, reshaped_size[0], reshaped_size[1]))

    for j, coord in enumerate(coords):
        x1, y1, x2, y2 = coord
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue  # ignore

        # multi-hot encoding for type?
        class_position = types[j]
        grid_arrs[class_position, y1:y2, x1:x2] = 1.

    return torch.from_numpy(grid_arrs)


def coord2pixel(img_path, coords, types, num_types=5, reshaped_size=(256, 512)) -> torch.Tensor:
    '''
    Convert coordinate to multi-hot encodings for coordinate class
    '''
    img = cv2.imread(img_path) if not isinstance(img_path, np.ndarray) else img_path
    coords = coords.numpy() if not isinstance(coords, np.ndarray) else coords
    coords = coord_reshape(coords, img.shape[:2], reshaped_size)  # reshape coordinates
    types = types.numpy() if not isinstance(types, np.ndarray) else types

    # Incorrect path/empty image
    if img is None:
        raise AttributeError('Image is None')
    height, width = img.shape[:2]
    # Empty image
    if height == 0 or width == 0:
        raise AttributeError('Empty image')

    # grid array of shape ClassxHxW = 5xHxW
    grid_arrs = np.zeros((num_types, reshaped_size[0], reshaped_size[1]))
    type_dict = {'logo': 1, 'input': 2, 'button': 3, 'label': 4, 'block': 5}

    for j, coord in enumerate(coords):
        x1, y1, x2, y2 = coord
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue  # ignore

        # multi-hot encoding for type?
        class_position = type_dict[types[j]] - 1
        grid_arrs[class_position, y1:y2, x1:x2] = 1.

    return torch.from_numpy(grid_arrs)


def topo2pixel(img_path, coords, knn_matrix, reshaped_size=(256, 512)) -> torch.Tensor:
    '''
    Convert coordinate to multi-hot encodings for coordinate class
    '''
    img = cv2.imread(img_path) if not isinstance(img_path, np.ndarray) else img_path
    coords = coords.numpy() if not isinstance(coords, np.ndarray) else coords
    coords = coord_reshape(coords, img.shape[:2], reshaped_size)  # reshape coordinates
    knn_matrix = knn_matrix.numpy() if not isinstance(knn_matrix, np.ndarray) else knn_matrix

    # Incorrect path/empty image
    if img is None:
        raise AttributeError('Image is None')
    height, width = img.shape[:2]
    # Empty image
    if height == 0 or width == 0:
        raise AttributeError('Empty image')

    # grid array of shape (KxZ)xHxW = 12xHxW
    topo_arrs = np.zeros((12, reshaped_size[0], reshaped_size[1]))
    if len(coords) <= 1:  # num of components smaller than 2
        return torch.from_numpy(topo_arrs)

    for j, coord in enumerate(coords):
        x1, y1, x2, y2 = coord
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue  # ignore

        # fill in topological info (zero padding if number of neighbors is less than 3)
        topo_arrs[:min(len(knn_matrix[j]), 12), y1:y2, x1:x2] = knn_matrix[j][:, np.newaxis][:, np.newaxis]

    return torch.from_numpy(topo_arrs)


def read_img_reverse(img, coords, types, num_types=5, grid_num=10) -> torch.Tensor:
    '''
    Convert image with bbox predictions as into grid format
    :param img: image path in str or image in np.ndarray
    :param coords: Nx4 tensor/np.ndarray for box coords
    :param types: Nx1 tensor/np.ndarray for box types (logo, input etc.)
    :param num_types: total number of box types
    :param grid_num: number of grids needed
    :return: grid tensor
    '''

    img = cv2.imread(img) if not isinstance(img, np.ndarray) else img
    coords = coords.numpy() if not isinstance(coords, np.ndarray) else coords
    types = types.numpy() if not isinstance(types, np.ndarray) else types

    # Incorrect path/empty image
    if img is None:
        raise AttributeError('Image is None')

    height, width = img.shape[:2]

    # Empty image
    if height == 0 or width == 0:
        raise AttributeError('Empty image')

    # grid array of shape CxHxW
    grid_arrs = np.zeros((4 + num_types, grid_num, grid_num))  # Must be [0, 1], use rel_x, rel_y, rel_w, rel_h

    for j, coord in enumerate(coords):
        x1, y1, x2, y2 = coord
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w == 0 or h == 0:
            continue  # ignore

        # get the assigned grid index
        assigned_grid_w, assigned_grid_h = int(((x1 + x2) / 2) // (width // grid_num)), int(
            ((y1 + y2) / 2) // (height // grid_num))

        # bound above
        assigned_grid_w = min(grid_num - 1, assigned_grid_w)
        assigned_grid_h = min(grid_num - 1, assigned_grid_h)

        # if this grid has been assigned before, check whether need to re-assign
        if grid_arrs[0, assigned_grid_h, assigned_grid_w] != 0:  # visted
            exist_type = np.where(grid_arrs[:, assigned_grid_h, assigned_grid_w] == 1)[0][0] - 4
            new_type = types[j]
            if new_type > exist_type:  # if new type has lower priority than existing type
                continue

        # fill in rel_xywh
        grid_arrs[0, assigned_grid_h, assigned_grid_w] = float(x1 / width)
        grid_arrs[1, assigned_grid_h, assigned_grid_w] = float(y1 / height)
        grid_arrs[2, assigned_grid_h, assigned_grid_w] = float(w / width)
        grid_arrs[3, assigned_grid_h, assigned_grid_w] = float(h / height)

        # one-hot encoding for type
        cls_arr = np.zeros(num_types)
        cls_arr[types[j]] = 1

        grid_arrs[4:, assigned_grid_h, assigned_grid_w] = cls_arr

    return torch.from_numpy(grid_arrs)


import torch.nn.functional as F
from PIL import Image
import math

def resolution_alignment(img1, img2):
    '''
    Resize two images according to the minimum resolution between the two
    :param img1: first image in PIL.Image
    :param img2: second image in PIL.Image
    :return: resized img1 in PIL.Image, resized img2 in PIL.Image
    '''
    w1, h1 = img1.size
    w2, h2 = img2.size
    w_min, h_min = min(w1, w2), min(h1, h2)
    if w_min == 0 or h_min == 0:  ## something wrong, stop resizing
        return img1, img2
    if w_min < h_min:
        img1_resize = img1.resize((int(w_min), math.ceil(h1 * (w_min/w1)))) # ceiling to prevent rounding to 0
        img2_resize = img2.resize((int(w_min), math.ceil(h2 * (w_min/w2))))
    else:
        img1_resize = img1.resize((math.ceil(w1 * (h_min/h1)), int(h_min)))
        img2_resize = img2.resize((math.ceil(w2 * (h_min/h2)), int(h_min)))
    return img1_resize, img2_resize

def load_brand_mapping(config_file="brand_mapping.json"):
    """Load brand name mapping configuration"""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_file} not found, using empty mapping")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Configuration file {config_file} format error")
        return {}

# Pre-load mapping table for better performance
_brand_mapping = load_brand_mapping()

def brand_converter(brand_name):
    """
    Helper function to deal with inconsistency in brand naming
    """
    return _brand_mapping.get(brand_name, brand_name)

def reload_brand_mapping(config_file="brand_mapping.json"):
    """Reload brand mapping configuration (for hot updates)"""
    global _brand_mapping
    _brand_mapping = load_brand_mapping(config_file)

def l2_norm(x):
    """
    l2 normalization
    :param x:
    :return:
    """
    if len(x.shape):
        x = x.reshape((x.shape[0], -1))
    return F.normalize(x, p=2, dim=1)