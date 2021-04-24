import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use all devices

from tqdm import tqdm
import time
from src.siamese import *
from src.element_detector import *
from src.credential import *
from src.detectron2_pedia.inference import *
import argparse
import errno


def phishintention_eval(data_dir, mode, siamese_ts, write_txt):
    '''
    Run phishintention evaluation
    :param data_dir: data folder dir
    :param mode: phish|benign
    :param siamese_ts: siamese threshold
    :param write_txt: txt path to write results
    :return:
    '''
    with open(write_txt, 'w') as f:
        f.write('folder\t')
        f.write('true_brand\t')
        f.write('phish_category\t')
        f.write('pred_brand\t')
        f.write('runtime_element_recognition\t')
        f.write('runtime_credential_classifier\t')
        f.write('runtime_siamese\t')
        f.write('\n')

    for folder in tqdm(os.listdir(data_dir)):

        phish_category = 0  # 0 for benign, 1 for phish
        pred_target = None  # predicted target, default is None

        img_path = os.path.join(data_dir, folder, 'shot.png')
        html_path = os.path.join(data_dir, folder, 'html.txt')
        if mode == 'phish':
            url = eval(open(os.path.join(data_dir, folder, 'info.txt'), encoding="ISO-8859-1").read())
            url = url['url'] if isinstance(url, dict) else url
        else:
            try:
                url = open(os.path.join(data_dir, folder, 'info.txt'), encoding="ISO-8859-1").read()
            except:
                url = 'https://www' + folder

        # Element recognition module
        start_time = time.time()
        pred_classes, pred_boxes, pred_scores = element_recognition(img=img_path, model=ele_model)
        ele_recog_time = time.time() - start_time

        # If no element is reported
        if len(pred_boxes) == 0:
            phish_category = 0  # Report as benign

        # If at least one element is reported
        else:
            # Credential heuristic module
            start_time = time.time()
            cred_conf = None
            # CRP HTML heuristic
            cre_pred = html_heuristic(html_path)
            # Credential classifier module
            if cre_pred == 1:  # if HTML heuristic report as nonCRP
                cre_pred, cred_conf, _ = credential_classifier_mixed_al(img=img_path, coords=pred_boxes,
                                                                        types=pred_classes, model=cls_model)
            credential_cls_time = time.time() - start_time

            # Non-credential page
            if cre_pred == 1:  # if non-CRP
                phish_category = 0  # Report as benign

            # Credential page
            else:
                # Phishpedia module
                start_time = time.time()
                pred_target, _, _ = phishpedia_classifier(pred_classes=pred_classes, pred_boxes=pred_boxes,
                                                          domain_map_path=domain_map_path,
                                                          model=pedia_model,
                                                          logo_feat_list=logo_feat_list, file_name_list=file_name_list,
                                                          url=url,
                                                          shot_path=img_path,
                                                          ts=siamese_ts)
                siamese_time = time.time() - start_time

                # Phishpedia reports target
                if pred_target is not None:
                    phish_category = 1  # Report as suspicious

                # Phishpedia does not report target
                else:  # Report as benign
                    phish_category = 0

        # write to txt file
        with open(write_txt, 'a+') as f:
            f.write(folder + '\t')
            f.write(brand_converter(folder.split('+')[0]) + '\t')  # true brand
            f.write(str(phish_category) + '\t')  # phish/benign/suspicious
            f.write(brand_converter(pred_target) + '\t') if pred_target is not None else f.write(
                '\t')  # phishing target
            # element recognition time
            f.write(str(ele_recog_time) + '\t')
            # credential classifier/heuristic time
            f.write(str(credential_cls_time) + '\t') if 'credential_cls_time' in locals() else f.write('\t')
            # siamese time
            f.write(str(siamese_time) + '\t') if 'siamese_time' in locals() else f.write('\t')
            # layout time
            f.write('\n')

            # delete time variables
        try:
            del ele_recog_time
            del credential_cls_time
            del siamese_time
        except:
            pass


def phishpedia_eval(data_dir, mode, siamese_ts, write_txt):
    '''
    Run phishpedia evaluation
    :param data_dir: data folder dir
    :param mode: phish|benign
    :param siamese_ts: siamese threshold
    :param write_txt: txt path to write results
    :return:
    '''
    with open(write_txt, 'w') as f:
        f.write('folder\t')
        f.write('true_brand\t')
        f.write('phish_category\t')
        f.write('pred_brand\t')
        f.write('runtime_element_recognition\t')
        f.write('runtime_siamese\n')

    for folder in tqdm(os.listdir(data_dir)):

        phish_category = 0  # 0 for benign, 1 for phish
        pred_target = None  # predicted target, default is None

        img_path = os.path.join(data_dir, folder, 'shot.png')
        html_path = os.path.join(data_dir, folder, 'html.txt')
        if mode == 'phish':
            url = eval(open(os.path.join(data_dir, folder, 'info.txt'), encoding="ISO-8859-1").read())
            url = url['url'] if isinstance(url, dict) else url
        else:
            try:
                url = open(os.path.join(data_dir, folder, 'info.txt'), encoding="ISO-8859-1").read()
            except:
                url = 'https://www' + folder

        # Element recognition module
        start_time = time.time()
        pred_boxes, _, _, _ = pred_rcnn(im=img_path, predictor=ele_model)
        pred_boxes = pred_boxes.detach().cpu().numpy()
        ele_recog_time = time.time() - start_time

        # If no element is reported
        if len(pred_boxes) == 0:
            phish_category = 0  # Report as benign

        # If at least one element is reported
        else:
            # Phishpedia module
            start_time = time.time()
            pred_target, _, _ = phishpedia_classifier_logo(logo_boxes=pred_boxes, domain_map_path=domain_map_path,
                                                           model=pedia_model,
                                                           logo_feat_list=logo_feat_list,
                                                           file_name_list=file_name_list,
                                                           url=url,
                                                           shot_path=img_path,
                                                           ts=siamese_ts)

            siamese_time = time.time() - start_time

            # Phishpedia reports target
            if pred_target is not None:
                phish_category = 1  # Report as suspicious

            # Phishpedia does not report target
            else:  # Report as benign
                phish_category = 0

        # write to txt file
        with open(write_txt, 'a+') as f:
            f.write(folder + '\t')
            f.write(brand_converter(folder.split('+')[0]) + '\t')  # true brand
            f.write(str(phish_category) + '\t')  # phish/benign/suspicious
            f.write(brand_converter(pred_target) + '\t') if pred_target is not None else f.write('\t')  # phishing target
            # element recognition time
            f.write(str(ele_recog_time) + '\t')
            # siamese time
            f.write(str(siamese_time) + '\n') if 'siamese_time' in locals() else f.write('\n')

            # delete time variables
        try:
            del ele_recog_time
            del siamese_time
        except:
            pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--mode", choices=['phish', 'benign', 'discovery'], required=True,
                        help="Evaluate phishing or benign or discovery")
    parser.add_argument("--write-txt", required=True, help="Where to save results")
    parser.add_argument("--data-dir", required=True, help="Data Dir")
    parser.add_argument("--ts", required=True, help="Siamese threshold")
    parser.add_argument("--exp", choices=['intention', 'pedia'], required=True,
                        help="Which experiment to run")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_dir)
    mode = args.mode
    siamese_ts = float(args.ts)
    write_txt = args.write_txt

    if args.exp == 'intention':
        # Element recognition model
        ele_cfg, ele_model = element_config(
            rcnn_weights_path='src/element_detector/output/website_lr0.001/model_final.pth',
            rcnn_cfg_path='src/element_detector/configs/faster_rcnn_web.yaml')

        # CRP classifier
        cls_model = credential_config(
            checkpoint='src/credential_classifier/output/hybrid/hybrid_lr0.005/BiT-M-R50x1V2_0.005.pth.tar',
            model_type='mixed')

    elif args.exp == 'pedia':

        # element recognition model -- logo only
        cfg_path = 'src/detectron2_pedia/configs/faster_rcnn.yaml'
        weights_path = 'src/detectron2_pedia/output/rcnn_2/rcnn_bet365.pth'
        ele_model = config_rcnn(cfg_path, weights_path, conf_threshold=0.05)

    # Siamese
    pedia_model, logo_feat_list, file_name_list = phishpedia_config(num_classes=277,
                                                                    weights_path='src/phishpedia/resnetv2_rgb_new.pth.tar',
                                                                    targetlist_path='src/phishpedia/expand_targetlist_copy/')
    print('Number of protected logos = {}'.format(str(len(logo_feat_list))))

    # Domain map path
    domain_map_path = 'src/phishpedia/domain_map.pkl'

    # PhishIntention
    if args.exp == 'intention':
        phishintention_eval(data_dir, mode, siamese_ts, write_txt)
    # PhishPedia
    else:
        phishpedia_eval(data_dir, mode, siamese_ts, write_txt)