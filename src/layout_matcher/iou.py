import numpy as np
import math
from scipy.spatial.distance import cdist
from .misc import preprocess, read_coord
import time
import cv2

def bbox_dist(bboxes1, bboxes2):
    '''
    compute L2 distance between boxes
    :param bboxes1: Nx4
    :param bboxes2: Mx4
    :return: NxM similarity matrix
    '''

    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # compute width and height
    w1, h1 = x12 - x11, y12 - y11
    w2, h2 = x22 - x21, y22 - y21

    box1_mat = np.concatenate([x11, y11, w1, h1], axis=1)
    box2_mat = np.concatenate([x21, y21, w2, h2], axis=1)
    dist = cdist(box1_mat, box2_mat, metric='minkowski', p=2.)
    sim = 1. - dist/200.
    return sim

def bbox_boarder_dist_simple(bboxes1, bboxes2):
    '''
    Compute boarder distance simple version
    :param bboxes1: Nx4
    :param bboxes2: Mx4
    :return xd_mat: NxM boarder distance matrix in horizontal direction
    :return yd_mat: NxM boarder distance matrix in vertical direction
    '''
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # compute boarder distance distance in x dimension
    xd1 = np.subtract(x12, np.transpose(x21))
    xd2 = np.subtract(x11, np.transpose(x22))
    xd_mat = np.add(np.multiply(np.less(xd1, np.zeros_like(xd1)).astype('float'), np.abs(xd1)),
                    np.multiply(np.greater(xd2, np.zeros_like(xd2)).astype('float'), xd2))

    # compute boarder distance distance in y dimension
    yd1 = np.subtract(y12, np.transpose(y21))
    yd2 = np.subtract(y11, np.transpose(y22))
    yd_mat = np.add(np.multiply(np.less(yd1, np.zeros_like(yd1)).astype('float'), np.abs(yd1)),
                    np.multiply(np.greater(yd2, np.zeros_like(yd2)).astype('float'), yd2))

    return xd_mat, yd_mat


def bbox_boarder_dist(bboxes1, bboxes2):
    '''
    Compute boarder distance
    :param bboxes1: Nx4
    :param bboxes2: Mx4
    :return: NxM boarder distance similarity matrix
    '''
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # compute boarder distance distance in x dimension
    xd1 = np.subtract(x12, np.transpose(x21))
    xd2 = np.subtract(x11, np.transpose(x22))
    xd_mat = np.add(np.multiply(np.less(xd1, np.zeros_like(xd1)).astype('float'), np.abs(xd1)),
                    np.multiply(np.greater(xd2, np.zeros_like(xd2)).astype('float'), xd2))

    # compute boarder distance distance in y dimension
    yd1 = np.subtract(y12, np.transpose(y21))
    yd2 = np.subtract(y11, np.transpose(y22))
    yd_mat = np.add(np.multiply(np.less(yd1, np.zeros_like(yd1)).astype('float'), np.abs(yd1)),
                    np.multiply(np.greater(yd2, np.zeros_like(yd2)).astype('float'), yd2))

    # combine x_d and y_d
    boarder_dist_mat = np.sqrt(np.add(np.square(xd_mat), np.square(yd_mat)))
    boarder_sim_mat = 1 - boarder_dist_mat/(100*np.sqrt(2))
    return boarder_sim_mat


def bbox_overlaps_iou(bboxes1, bboxes2, type):
    '''
    Compute IoU matrix
    :param bboxes1: Nx4
    :param bboxes2: Mx4
    :param type: iou type
    :return: NxM iou similarity matrix 
    '''
    assert type in ['iou', 'giou', 'diou', 'ciou']
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    # compute intersection coordinates
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # compute outer coordinates
    xA_out = np.minimum(x11, np.transpose(x21))
    yA_out = np.minimum(y11, np.transpose(y21))
    xB_out = np.maximum(x12, np.transpose(x22))
    yB_out = np.maximum(y12, np.transpose(y22))

    # compute center
    center_x1, center_y1 = (x12 + x11) / 2., (y12 + y11) / 2.
    center_x2, center_y2 = (x22 + x21) / 2., (y22 + y21) / 2.

    # compute width and height
    w1, h1 = x12 - x11, y12 - y11
    w2, h2 = x22 - x21, y22 - y21

    # compute IoU
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    if type == 'iou':
        iou = np.clip(iou, a_min=0., a_max=1.)
        return iou

    if type == 'giou':
        includeArea = np.maximum((xB_out - xA_out + 1), 0) * np.maximum((yB_out - yA_out + 1), 0)
        factor = ((includeArea - (boxAArea + np.transpose(boxBArea) - interArea)) / includeArea)
        giou = np.subtract(iou, factor)
        giou = np.clip(giou, a_min=-1., a_max=1.)
        giou = (giou + 1) / 2. # rescale it to be between [0, 1]
        return giou

    if type in ['diou','ciou']:
        outer_diag = np.add(np.square(np.subtract(xB_out, xA_out)),
                            np.square(np.subtract(yB_out, yA_out)))
        inter_diag = np.add(np.square(np.subtract(center_x1, np.transpose(center_x2))),
                            np.square(np.subtract(center_y1, np.transpose(center_y2))))

        diou = np.subtract(iou, np.divide(inter_diag, outer_diag))

        if type == 'diou':
            diou = np.clip(diou, a_min=-1., a_max=1.)
            diou = (diou + 1) / 2. # rescale it to be between [0, 1]
            return diou

        if type == 'ciou':
            v = (4./math.pi) * np.square(np.subtract(np.arctan2(w1, h1), np.transpose(np.arctan2(w2, h2))))
            alpha = np.multiply(np.less(iou, np.ones_like(iou)*0.5).astype('float'), np.zeros_like(v)) + \
                    np.multiply(np.greater_equal(iou, np.ones_like(iou)*0.5).astype('float'), np.divide(v, 1-iou+v))
            ciou = diou - np.multiply(alpha, v)
            ciou = np.clip(ciou, a_min=-1., a_max=1.)
            ciou = (ciou + 1) / 2. # rescale it to be between [0, 1]
            return ciou


if __name__ == '__main__':
    shot_path1 = './data/layout_testset/Amazon/TP/amtiwqogzbndkqorueig-usid990057306uvjp.com.tlztppb.cn/shot.png'
    shot_path2 = './data/layout_5brand/Amazon/Amazon.com Inc.+2020-06-22-11`51`35/shot.png'

    start_time = time.time()
    shot_size1 = cv2.imread(shot_path1).shape
    shot_size2 = cv2.imread(shot_path2).shape
    coord_path = shot_path1.replace('shot.png', 'rcnn_coord.txt')
    compos1, _ = read_coord(coord_path)
    coord_path = shot_path2.replace('shot.png', 'rcnn_coord.txt')
    compos2, _ = read_coord(coord_path)

    # Rescale all components
    compos1 = preprocess(shot_size1, compos1)
    compos2 = preprocess(shot_size2, compos2)

    # print(compos1[:, 2:])
    # check = np.min(compos1[:, 2:], compos2[:, 2:])
    # print(check)
    print(compos1.shape)
    print(compos2.shape)
    # bbox_overlaps_iou(compos1, compos2, type='ciou')
    # bbox_dist(compos1, compos2)
    bbox_boarder_dist(compos1, compos2)
