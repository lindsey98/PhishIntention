import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


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


def l2_norm(x):
    """
    l2 normalization
    :param x:
    :return:
    """
    if len(x.shape):
        x = x.reshape((x.shape[0], -1))
    return F.normalize(x, p=2, dim=1)
