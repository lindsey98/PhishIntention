from .layout_matcher_knn import *
import os
import matplotlib.pyplot as plt
import shutil
from .misc import read_coord, load_yaml
import argparse
import numpy as np

def gen_result(testdir, brand, type, gt_site_arr, gt_coords_arr, gt_size_arr, cfg):
    assert type in ['TP_subset', 'TP', 'FP']
    print(brand + type)
    if os.path.exists('./results/' + brand + '_%s.txt'%type):
        os.unlink('./results/' + brand + '_%s.txt'%type)

    testsite_list, match_site_list, match_dist_list = test_site(testdir=testdir,
                                                                gt_site_arr=gt_site_arr,
                                                                gt_coords_arr=gt_coords_arr,
                                                                gt_size_arr=gt_size_arr,
                                                                cfg=cfg)
    plt.hist(match_dist_list)
    plt.savefig('./results/'+brand+'_%s.png'%type)
    plt.close()

    with open('./results/' + brand + '_%s.txt'%type, 'a+') as f:
        for i in range(len(testsite_list)):
            f.write(testsite_list[i]+'\t'+match_site_list[i]+'\t'+str(match_dist_list[i])+'\n')

def test_site(testdir, gt_site_arr, gt_coords_arr, gt_size_arr, cfg):
    testsite_list = []
    match_gtsite_list = []
    match_dist_list = []

    for site in os.listdir(testdir):
        rcnn_path = os.path.join(testdir, site, 'rcnn_coord.txt')
        shot_size = cv2.imread(rcnn_path.replace('rcnn_coord.txt', 'shot.png')).shape
        coords_arr, _ = read_coord(rcnn_path)

        if len(coords_arr) <= 1:
            continue

        max_s = 0
        for j, gt_c in enumerate(gt_coords_arr):
            similarity, sim_mat, max_ind_list, sort_ind1, sort_ind2, box_similarity, box_sim_mat, max_ind_list2 = \
                bipartite_web(gt_c, coords_arr, gt_size_arr[j], shot_size, cfg)

            if similarity >= max_s:
                max_s = similarity
                max_site = gt_site_arr[j]

        print(site, max_s)
        testsite_list.append(site)
        match_gtsite_list.append(max_site)
        match_dist_list.append(max_s)

    assert (len(testsite_list) == len(match_gtsite_list)) & (len(match_gtsite_list) == len(match_dist_list))
    return testsite_list, match_gtsite_list, match_dist_list

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",  required=True,
                    help="Path to configuration yaml")
    ap.add_argument("--out", required=True,
                    help="Path to save results")
    args = vars(ap.parse_args())
    cfg = load_yaml(args["cfg"])

    if not os.path.exists(args["out"]):
        with open(args["out"], 'w') as f:
            f.write('K\tBox_sim_type\tIoU_type\tWeight_box_sim\t')
            f.write('topo_neigh_ts\ttopo_box_ts\tiou_box_ts\tdist_box_ts\t')
            f.write('brand\tbest_ts\tbest_tp\tbest_fp\n')


    for brand in ['Amazon', 'Facebook', 'Google', 'Instagram', 'LinkedIn Corporation']:
        test_tpsubset = './data/layout_testset/' + brand + '/TP_subset'
        test_tp = './data/layout_testset/' + brand + '/TP'
        test_fp = './data/layout_testset/' + brand + '/FP'
        gtdir = './data/layout_5brand/' + brand

        ############### read ground-truth ########################
        gt_coords_arr = []
        gt_site_arr = []
        gt_size_arr = []
        for site in os.listdir(gtdir):
            rcnn_path = os.path.join(gtdir, site, 'rcnn_coord.txt')
            shot_size = cv2.imread(rcnn_path.replace('rcnn_coord.txt', 'shot.png')).shape
            coords_arr, _ = read_coord(rcnn_path)
            if len(coords_arr) <= 1:
                continue
            gt_coords_arr.append(coords_arr)
            gt_site_arr.append(os.path.join(gtdir, site))
            gt_size_arr.append(shot_size)

        assert len(gt_site_arr) == len(gt_coords_arr) & len(gt_size_arr) == len(gt_site_arr)
        ############################################################

        # # TP_subset
        # gen_result(testdir=test_tpsubset, brand=brand,
        #            type='TP_subset', gt_site_arr=gt_site_arr,
        #            gt_coords_arr=gt_coords_arr, gt_size_arr=gt_size_arr, cfg=cfg)
        # TP
        gen_result(testdir=test_tp, brand=brand,
                   type='TP', gt_site_arr=gt_site_arr,
                   gt_coords_arr=gt_coords_arr, gt_size_arr=gt_size_arr, cfg=cfg)
        # FP
        gen_result(testdir=test_fp, brand=brand,
                   type='FP', gt_site_arr=gt_site_arr,
                   gt_coords_arr=gt_coords_arr, gt_size_arr=gt_size_arr, cfg=cfg)

        # Threshold selection
        tp_save_path = './results/' + brand + '_TP.txt'
        fp_save_path = './results/' + brand + '_FP.txt'
        all_tp = [float(x.strip().split('\t')[-1]) for x in open(tp_save_path).readlines()]
        all_fp = [float(x.strip().split('\t')[-1]) for x in open(fp_save_path).readlines()]

        for ts in np.arange(0.2, 0.95, 0.05):
            num_tp = np.sum(np.asarray(all_tp) > ts)  # keep as many TPs as possible
            num_fp = np.sum(np.asarray(all_fp) > ts)  # reduce as many FPs as possible

            with open(args['out'], 'a+') as f:
                f.write(str(cfg['MODEL']['K']) + '\t')
                f.write(str(cfg['MODEL']['box_sim_type']) + '\t')
                f.write(cfg['MODEL']['iou_type'] + '\t')
                f.write(str(cfg['MODEL']['weight_box_sim']) + '\t')
                f.write(str(cfg['THRESHOLD']['topo_neigh_ts']) + '\t')
                f.write(str(cfg['THRESHOLD']['topo_box_ts']) + '\t')
                f.write(str(cfg['THRESHOLD']['iou_box_ts']) + '\t')
                f.write(str(cfg['THRESHOLD']['dist_box_ts']) + '\t')
                f.write(brand+'\t')
                f.write(str(ts)+'\t')
                f.write(str(num_tp)+'\t')
                f.write(str(num_fp)+'\n')

