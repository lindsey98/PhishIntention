from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np
from numpy import unravel_index

def boarder_distance(compos):
    '''
    Compute boarder distance between any pair of two box arrays
    :param compos1: Nx4
    :param compos2: Nx4
    :return: boarder distance matrix: NxN
    '''
    x1, y1, x2, y2 = np.split(compos, 4, axis=1)

    # compute boarder distance distance in x dimension
    xd1 = np.subtract(x2, np.transpose(x1))
    xd2 = np.subtract(x1, np.transpose(x2))
    xd_mat = np.add(np.multiply(np.less(xd1, np.zeros_like(xd1)).astype('float'), np.abs(xd1)),
                    np.multiply(np.greater(xd2, np.zeros_like(xd2)).astype('float'), xd2))

    # compute boarder distance distance in y dimension
    yd1 = np.subtract(y2, np.transpose(y1))
    yd2 = np.subtract(y1, np.transpose(y2))
    yd_mat = np.add(np.multiply(np.less(yd1, np.zeros_like(yd1)).astype('float'), np.abs(yd1)),
                    np.multiply(np.greater(yd2, np.zeros_like(yd2)).astype('float'), yd2))

    # combine x_d and y_d
    boarder_dist_mat = np.sqrt(np.add(np.square(xd_mat), np.square(yd_mat)))

    # add 1 to avoid boarder distance = 0
    dummy_mat = np.ones_like(boarder_dist_mat)
    boarder_dist_mat_final = np.add(boarder_dist_mat, dummy_mat)

    return boarder_dist_mat_final


def knn_matrix(compos, k, norm_method='log'):
    '''
    Compute KNN matrix for each element, matrix attributes include relative width, relative height, and distance
    :param compos: components list
    :param k: number of neighbors
    :param norm_method: normalization method used, default is log of base 2
    :return: KNN matrix of shape NxKxZ, N:number of components, K:number of neighbors, Z:number of attributes for each neighbor,
    '''

    # prepare width, height, aspect ratio, distance matrix/vectors
    compos = np.asarray(compos)
    center_arr = np.vstack(((compos[:, 0] + compos[:, 2]) / 2, (compos[:, 1] + compos[:, 3]) / 2,)).T

    # distance = sqrt(euclidean_dist * (boarder_dist + 1))
    e_sim_mat = squareform(pdist(center_arr))  # compute euclidean distance matrix of shape NxN
    b_sim_mat = boarder_distance(compos)
    dist_mat = np.sqrt(np.multiply(e_sim_mat, b_sim_mat))

    # get (x, y, w, h)
    width_vec = compos[:, 2] - compos[:, 0]
    height_vec = compos[:, 3] - compos[:, 1]
    x_center_vec = (compos[:, 2] + compos[:, 0]) / 2
    y_center_vec = (compos[:, 3] + compos[:, 1]) / 2

    # sort to get k nearest neighbors + itself
    dist_mat_pre = dist_mat.copy()  # this is to avoid the case when some elements happen to have distance 0 as well
    np.fill_diagonal(dist_mat_pre, -1)  # itself is always the 1st column in sim_mat_pre, the rest columns are for its k neighbors
    sort_ind = dist_mat_pre.argsort(axis=1)[:, :(k + 1)]

    # compute relative matrices
    ## relative width w/w0, relative height h/h0, relative aspect ratio r/r0  Nx(K+1)
    w_mat, h_mat = np.take(width_vec, sort_ind), np.take(height_vec, sort_ind)
    base_width, base_height = np.repeat(w_mat[:, 0][:, np.newaxis], min(k + 1, w_mat.shape[1]), axis=1), \
                              np.repeat(h_mat[:, 0][:, np.newaxis], min(k + 1, h_mat.shape[1]), axis=1)
    base_width, base_height = base_width + 1e-4, base_height + 1e-4 # avoid runtimeerror
    
    rel_w, rel_h = np.divide(w_mat, base_width), \
                   np.divide(h_mat, base_height)

    # relative distance (x-x0)/w0, (y-y0)/h0 Nx(K+1)
    x_center_mat, y_center_mat = np.take(x_center_vec, sort_ind), \
                                 np.take(y_center_vec, sort_ind)

    base_x, base_y = np.repeat(x_center_mat[:, 0][:, np.newaxis], min(k + 1, x_center_mat.shape[1]), axis=1), \
                     np.repeat(y_center_mat[:, 0][:, np.newaxis], min(k + 1, y_center_mat.shape[1]), axis=1)

    rel_x, rel_y = np.divide(np.subtract(x_center_mat, base_x), np.ones_like(x_center_mat) * base_width), \
                   np.divide(np.subtract(y_center_mat, base_y), np.ones_like(y_center_mat) * base_height)

    if norm_method == 'log':
        # log normalization
        rel_w, rel_h = np.log2(rel_w), np.log2(rel_h)
        rel_w, rel_h = np.clip(rel_w, a_min=-1, a_max=1), \
                       np.clip(rel_h, a_min=-1, a_max=1)  # TODO: clip or not?

        # Clip if use no normalization
        rel_x, rel_y = np.clip(rel_x, a_min=-1., a_max=1.), \
                       np.clip(rel_y, a_min=-1., a_max=1.)

    elif norm_method == 'n/n+1':
        # n/(n+1) normalization
        rel_w, rel_h = np.divide(rel_w, np.add(rel_w, np.ones_like(rel_w))), \
                       np.divide(rel_h, np.add(rel_h, np.ones_like(rel_h)))

        rel_w, rel_h = rel_w * 2. - 1., rel_h * 2. - 1.

        # n/n+1 normalization
        rel_x, rel_y = np.multiply(np.sign(rel_x),
                                   np.divide(np.abs(rel_x), np.add(np.abs(rel_x), np.ones_like(rel_x)))), \
                       np.multiply(np.sign(rel_y), np.divide(np.abs(rel_y), np.add(np.abs(rel_y), np.ones_like(rel_y))))

    elif norm_method == 'minmax':
        # max(a/2, 1)*2-1 normalization
        rel_w, rel_h = np.clip(rel_w / 2., a_min=None, a_max=1.) * 2. - 1., \
                       np.clip(rel_h / 2., a_min=None, a_max=1.) * 2. - 1.

        # min(max(a/2, 1), -1) normalization
        rel_x, rel_y = np.clip(rel_x / 2., a_min=-1., a_max=1.), np.clip(rel_y / 2., a_min=-1., a_max=1.)

    # Final matrix is of shape NxKxZ
    combine_matrix = np.concatenate([rel_w[np.newaxis, :, 1:], rel_h[np.newaxis, :, 1:],
                                     rel_x[np.newaxis, :, 1:], rel_y[np.newaxis, :, 1:]], axis=0)

    combine_matrix = np.swapaxes(combine_matrix, 0, 1)
    combine_matrix = np.swapaxes(combine_matrix, 1, 2)
    return combine_matrix, sort_ind


def sim_mat(bboxes1, bboxes2):
    '''
    Compute similarity matrix between two boxes based on its K neighbors
    :param bboxes1: KxZ, K is the number of neighbors, each neighbor has Z attributes
    :param bboxes2: KxZ, K is the number of neighbors, each neighbor has Z attributes
    :return: similarity matrix: KxK
    '''
    dist = cdist(bboxes1, bboxes2, metric='minkowski', p=2.)
    sim = 1. - dist/4.
    return sim