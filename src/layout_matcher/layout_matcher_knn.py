import time
import os
import cv2
import itertools
from .misc import read_coord, preprocess, load_yaml
import shutil
from .topology import *
from .iou import *
import numpy as np

def match_pair(x, t_s):
    '''
    :param x: vector
    :param t_s: threshold
    :return: vector with unmatched pair removed
    '''

    # compute only matched pairs
    x = np.asarray(x)
    x_remove = x[x >= t_s]
    return x_remove

def bipartite_graph_match(mat, t_s):
    '''
    Perform minimum weight graph matching
    :param mat: similarity matrix NxM
    :param t_s: threshold to decide a successful match
    :return similarity: scalar value similarity
    '''

    result_matrix_cp = mat.copy()
    sim_vec = []
    max_ind_ls = []
    
    # greedy approach to obtain bipartite graph matching
    while np.sum(result_matrix_cp) > 0:
        max_row, max_col = unravel_index(result_matrix_cp.argmax(), result_matrix_cp.shape)  # find maximum similarity index
        sim_vec.append(result_matrix_cp[max_row, max_col])  # get the similarity vector
        # zero out the row and column
        result_matrix_cp[max_row, :] = 0
        result_matrix_cp[:, max_col] = 0
        max_ind_ls.append([max_row, max_col])

    sim_vec_trim = match_pair(sim_vec, t_s) # apply threshold
    num_match_pair = len(sim_vec_trim) # get the number of matched pairs
    if num_match_pair == 0:
        similarity = 0.
    else:
        similarity = np.mean(sim_vec_trim) * (2*num_match_pair/(np.sum(result_matrix_cp.shape)))
    return similarity, np.asarray(max_ind_ls)

def bipartite_box(box_mat1, box_mat2, t_s):
    '''
    Bipartite matching between two box matrices
    :param box_mat1: KNN matrix for box 1 KxZ
    :param box_mat2: KNN matrix for box 2 KxZ
    :return: scalar-value similarity
    '''

    result_matrix = sim_mat(box_mat1, box_mat2) #KxK similarity matrix
    similarity, _ = bipartite_graph_match(result_matrix, t_s)
    return similarity

def bipartite_web(compos1, compos2, cls1, cls2,
                  shot_size1, shot_size2, cfg):
    '''
    Bipartite matching between two layouts
    :param compos1: Components for layout 1 Nx4
    :param compos2: Components for layout 2 Mx4
    :param cls1: Component types for components1 (N,)
    :param cls2: Component types for components2 (M,)
    :param shot_size1: screenshot1 size
    :param shot_size2: screenshot2 size
    :param cfg: config dictionary
    :return: scalar-value similarity
    '''
    
    # Remove block elements, the class code for block is 4
#     print('Before removing block:', len(compos1), len(compos2))
    compos1 = compos1[cls1 != 4]
    compos2 = compos2[cls2 != 4]
    cls1 = cls1[cls1 != 4]
    cls2 = cls2[cls2 != 4]
#     print('After removing block:', len(compos1), len(compos2))
    
    # Rescale all components
    compos1 = preprocess(shot_size1, compos1)
    compos2 = preprocess(shot_size2, compos2)

    # layout extraction -- KNN matrix computation
    box1_matrix, sort_ind1 = knn_matrix(compos1, cfg['MODEL']['K'])
    box2_matrix, sort_ind2 = knn_matrix(compos2, cfg['MODEL']['K'])

    # bipartite matching for box sim_mat is of shape NxM
    sim_mat = np.array([[bipartite_box(i, j, cfg['THRESHOLD']['topo_neigh_ts']) for j in box2_matrix] for i in box1_matrix])
    
    # set box pair of different types to be of similarity 0
    type_mask = (cls1[:, np.newaxis] == cls2[np.newaxis, :]).astype(float)
    sim_mat = np.multiply(type_mask, sim_mat)
#     print("After applying mask: ", np.sum(sim_mat > 0))
    
    # bipartite matching for layout to get topological similarity
    similarity, max_ind_list = bipartite_graph_match(sim_mat, t_s=cfg['THRESHOLD']['topo_box_ts'])

    box_similarity, box_sim_mat, max_ind_list2 = None, None, None

    # No box similarity
    if cfg['MODEL']['box_sim_type'] == 0:
        pass

    # compute IoU based similarity
    elif cfg['MODEL']['box_sim_type'] == 1:
        box_sim_mat = bbox_overlaps_iou(compos1, compos2, type=cfg['MODEL']['iou_type'])
        box_sim_mat = np.multiply(type_mask, box_sim_mat) # set box pair of different types to be of similarity 0
        box_similarity, max_ind_list2 = bipartite_graph_match(box_sim_mat, t_s=cfg['THRESHOLD']['iou_box_ts'])
        similarity = (1-cfg['MODEL']['weight_box_sim'])*similarity + cfg['MODEL']['weight_box_sim']*box_similarity

    # compute L2-distance based similarity
    elif cfg['MODEL']['box_sim_type'] == 2:
        box_sim_mat = bbox_dist(compos1, compos2)
        box_sim_mat = np.multiply(type_mask, box_sim_mat) # set box pair of different types to be of similarity 0
        box_similarity, max_ind_list2 = bipartite_graph_match(box_sim_mat, t_s=cfg['THRESHOLD']['dist_box_ts'])
        similarity = (1-cfg['MODEL']['weight_box_sim'])*similarity + cfg['MODEL']['weight_box_sim']*box_similarity

    # compute boarder distance based similarity
    elif cfg['MODEL']['box_sim_type'] == 3:
        box_sim_mat = bbox_boarder_dist(compos1, compos2)
        box_sim_mat = np.multiply(type_mask, box_sim_mat) # set box pair of different types to be of similarity 0
        box_similarity, max_ind_list2 = bipartite_graph_match(box_sim_mat, t_s=cfg['THRESHOLD']['dist_box_ts'])
        similarity = (1 - cfg['MODEL']['weight_box_sim']) * similarity + cfg['MODEL']['weight_box_sim'] * box_similarity
    
    return similarity, sim_mat, max_ind_list, sort_ind1, sort_ind2, \
           box_similarity, box_sim_mat, max_ind_list2



# if __name__ == '__main__':
#     cfg = load_yaml("./configs.yaml")

#     # from layout_clustering import read_coord
#     shot_path1 = './data/layout_testset/Amazon/TP/amtiwqogzbndkqorueig-usid990057306uvjp.com.tlztppb.cn/shot.png'
#     shot_path2 = './data/layout_5brand/Amazon/Amazon.com Inc.+2020-06-22-11`51`35/shot.png'


#     start_time = time.time()
#     shot_size1 = cv2.imread(shot_path1).shape
#     shot_size2 = cv2.imread(shot_path2).shape
#     coord_path = shot_path1.replace('shot.png', 'rcnn_coord.txt')
#     compos1, _ = read_coord(coord_path)
#     coord_path = shot_path2.replace('shot.png', 'rcnn_coord.txt')
#     compos2, _ = read_coord(coord_path)
#     # 
#     # knn_matrix(compos1, k=3)
#     similarity, sim_mat, max_ind_list, sort_ind1, sort_ind2, box_similarity, box_mat, max_ind_list2 \
#         = bipartite_web(compos1, compos2, shot_size1, shot_size2, cfg)
#     print(np.asarray(compos1).shape)
#     print(np.asarray(compos2).shape)
#     print(sim_mat.shape)
#     debug_viz(shot_path1, shot_path2,
#               compos1, compos2,
#               sort_ind1, sort_ind2,
#               max_ind_list,
#               max_ind_list2,
#               sim_mat,
#               similarity,
#               box_mat,
#               save_path='./data/output')
#     print("Similarity is:", similarity)
#     print(time.time() - start_time)


