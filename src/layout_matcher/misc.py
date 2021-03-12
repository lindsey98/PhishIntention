import numpy as np
import re
import yaml
import os
import cv2
import shutil

def preprocess(shot_size, compos):
    '''
    :param shot_size: in HxW format
    :param compos:
    :return: rescaled compos within range of [0, 100]
    '''
    image_size = np.array(list(shot_size[:2][::-1] * 2))
    box_after = np.asarray(list(map(lambda x: np.divide(x, image_size) * 100, compos)))
    return box_after

def read_coord(coord_path):
    '''
    Read coordinates txt
    :param coord_path:
    :return: coordinates array Nx4, confidence array Nx1
    '''
    coords = [x.strip().split('\t')[0] for x in open(coord_path).readlines()]
    confidence = [x.strip().split('\t')[1] for x in open(coord_path).readlines()]

    coords_arr = []
    for coord in coords:
        testbox = list(map(float, re.search(r'\((.*?)\)', coord).group(1).split(",")))
        coords_arr.append(testbox)

    coords_arr = np.asarray(coords_arr)
    return coords_arr, confidence


def load_yaml(yaml_path):
    '''Load yaml file'''
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config



def debug_viz_single(shot_path1,
                     compos1, save_path):
    
    os.makedirs(save_path, exist_ok=True)
    save_path1 = os.path.join(save_path, 'shot_single.png')
    img1 = cv2.imread(shot_path1)

    for i, coord in enumerate(compos1):
        min_x, min_y, max_x, max_y = list(map(int, coord))
        cv2.rectangle(img1, (min_x, min_y), (max_x, max_y), (36, 255, 12), 2)
        cv2.putText(img1, str(i + 1), (min_x + 10, min_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite(save_path1, img1)

def debug_viz(shot_path1, shot_path2,
              compos1, compos2,
              sort_neigh_ind1, sort_neigh_ind2,
              max_ind_list,
              max_ind_list2,
              sim_mat,
              similarity,
              box_mat,
              save_path,
              clean_dir=True):
    
    if clean_dir:
        try:
            shutil.rmtree(save_path)
        except:
            pass
    os.makedirs(save_path, exist_ok=True)

    save_path1 = os.path.join(save_path, shot_path1.split('/')[-2]+'.png')
    save_path2 = os.path.join(save_path, shot_path1.split('/')[-2]+'_'+str(similarity)+'.png')

    img1 = cv2.imread(shot_path1)
    img2 = cv2.imread(shot_path2)

    for i, coord in enumerate(compos1):
        min_x, min_y, max_x, max_y = list(map(int, coord))
        cv2.rectangle(img1, (min_x, min_y), (max_x, max_y), (36, 255, 12), 2)
        if len(max_ind_list) == 0 or len(np.where(max_ind_list[:, 0] == i)[0].tolist()) == 0:
            cv2.putText(img1, str(i + 1),
                        (min_x + 10, min_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(img1, "KNN: " + str(sort_neigh_ind1[i, 1:] + 1), (min_x + 20, min_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            pass
        else:
            j = max_ind_list[np.where(max_ind_list[:, 0] == i)[0][0], 1]
            cv2.putText(img1, str(i + 1) + "+" + str(j + 1) + "+" + str(round(sim_mat[i, j], 2)),
                        (min_x + 10, min_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(img1, "KNN: " + str(sort_neigh_ind1[i, 1:] + 1), (min_x + 20, min_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        if len(max_ind_list2) == 0 or len(np.where(max_ind_list2[:, 0] == i)[0].tolist()) == 0:
            pass
        else:
            j = max_ind_list2[np.where(max_ind_list2[:, 0] == i)[0][0], 1]
            cv2.putText(img1, str(i + 1) + "+" + str(j + 1) + "+" + str(round(box_mat[i, j], 2)),
                        (min_x + 50, min_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)


    for j, coord in enumerate(compos2):
        min_x, min_y, max_x, max_y = list(map(int, coord))
        cv2.rectangle(img2, (min_x, min_y), (max_x, max_y), (36, 255, 12), 2)
        if len(max_ind_list)==0 or len(np.where(max_ind_list[:, 1] == j)[0].tolist()) == 0:
            cv2.putText(img2, str(j + 1),
                        (min_x + 10, min_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(img2, "KNN: " + str(sort_neigh_ind2[j, 1:] + 1), (min_x + 20, min_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            pass
        else:
            i = max_ind_list[np.where(max_ind_list[:, 1] == j)[0][0], 0]
            cv2.putText(img2, str(j + 1) + "+" + str(i + 1) + "+" + str(round(sim_mat[i, j], 2)),
                        (min_x + 10, min_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(img2, "KNN: " + str(sort_neigh_ind2[j, 1:] + 1), (min_x + 20, min_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        if len(max_ind_list2)==0 or len(np.where(max_ind_list2[:, 1] == j)[0].tolist()) == 0:
            pass
        else:
            i = max_ind_list2[np.where(max_ind_list2[:, 1] == j)[0][0], 0]
            cv2.putText(img2, str(j + 1) + "+" + str(i + 1) + "+" + str(round(box_mat[i, j], 2)),
                        (min_x + 50, min_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)

    cv2.imwrite(save_path1, img1)
    cv2.imwrite(save_path2, img2)