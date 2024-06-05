
import time
from datetime import datetime
import argparse
import os
import torch
import cv2
from configs import load_config
from modules.awl_detector import pred_rcnn, vis, find_element_type
from modules.logo_matching import check_domain_brand_inconsistency
from modules.crp_classifier import credential_classifier_mixed, html_heuristic
from modules.crp_locator import crp_locator
from utils.web_utils import driver_loader
from tqdm import tqdm
import re

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class PhishIntentionWrapper:
    _caller_prefix = "PhishIntentionWrapper"
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        self._load_config()

    def _load_config(self):
        self.AWL_MODEL, self.CRP_CLASSIFIER, self.CRP_LOCATOR_MODEL, self.SIAMESE_MODEL, self.OCR_MODEL, \
            self.SIAMESE_THRE, self.LOGO_FEATS, self.LOGO_FILES, self.DOMAIN_MAP_PATH = load_config()
        print(f'Length of reference list = {len(self.LOGO_FEATS)}')

    '''PhishIntention'''
    def test_orig_phishintention(self, url, screenshot_path):

        waive_crp_classifier = False
        phish_category = 0  # 0 for benign, 1 for phish, default is benign
        pred_target = None
        matched_domain = None
        siamese_conf = None
        awl_detect_time = 0
        logo_match_time = 0
        crp_class_time = 0
        crp_locator_time = 0
        print("Entering PhishIntention")

        while True:

            ####################### Step1: Layout detector ##############################################
            start_time = time.time()
            pred_boxes, pred_classes, _ = pred_rcnn(im=screenshot_path, predictor=self.AWL_MODEL)
            awl_detect_time += time.time() - start_time

            if pred_boxes is not None:
                pred_boxes = pred_boxes.numpy()
                pred_classes = pred_classes.numpy()
            plotvis = vis(screenshot_path, pred_boxes, pred_classes)

            # If no element is reported
            if pred_boxes is None or len(pred_boxes) == 0:
                print('No element is detected, reporte as benign')
                return phish_category, pred_target, matched_domain, plotvis, siamese_conf, \
                            str(awl_detect_time) + '|' + str(logo_match_time) + '|' + str(crp_class_time) + '|' + str(crp_locator_time), \
                            pred_boxes, pred_classes

            logo_pred_boxes, _ = find_element_type(pred_boxes, pred_classes, bbox_type='logo')
            if logo_pred_boxes is None or len(logo_pred_boxes) == 0:
                print('No logo is detected, reporte as benign')
                return phish_category, pred_target, matched_domain, plotvis, siamese_conf, \
                            str(awl_detect_time) + '|' + str(logo_match_time) + '|' + str(crp_class_time) + '|' + str(crp_locator_time), \
                            pred_boxes, pred_classes

            print('Entering siamese')

            ######################## Step2: Siamese (Logo matcher) ########################################
            start_time = time.time()
            pred_target, matched_domain, matched_coord, siamese_conf = check_domain_brand_inconsistency(logo_boxes=logo_pred_boxes,
                                                                                      domain_map_path=self.DOMAIN_MAP_PATH,
                                                                                      model = self.SIAMESE_MODEL,
                                                                                      ocr_model = self.OCR_MODEL,
                                                                                      logo_feat_list = self.LOGO_FEATS,
                                                                                      file_name_list = self.LOGO_FILES,
                                                                                      url=url,
                                                                                      shot_path=screenshot_path,
                                                                                      ts=self.SIAMESE_THRE)
            logo_match_time += time.time() - start_time

            if pred_target is None:
                print('Did not match to any brand, report as benign')
                return phish_category, pred_target, matched_domain, plotvis, siamese_conf, \
                            str(awl_detect_time) + '|' + str(logo_match_time) + '|' + str(crp_class_time) + '|' + str(crp_locator_time), \
                            pred_boxes, pred_classes

            ######################## Step3: CRP classifier (if a target is reported) #################################
            print('A target is reported by siamese, enter CRP classifier')
            if waive_crp_classifier:  # only run dynamic analysis ONCE
                break

            html_path = screenshot_path.replace("shot.png", "html.txt")
            start_time = time.time()
            cre_pred = html_heuristic(html_path)
            if cre_pred == 1:  # if HTML heuristic report as nonCRP
                # CRP classifier
                cre_pred = credential_classifier_mixed(img=screenshot_path,
                                                         coords=pred_boxes,
                                                         types=pred_classes,
                                                         model=self.CRP_CLASSIFIER)
            crp_class_time += time.time() - start_time

            ######################## Step4: Dynamic analysis #################################
            if cre_pred == 1:
                print('It is a Non-CRP page, enter dynamic analysis')
                # # load driver ONCE!
                driver = driver_loader()
                print('Finish loading webdriver')
                # load chromedriver
                url, screenshot_path, successful, process_time = crp_locator(url=url,
                                                                             screenshot_path=screenshot_path,
                                                                             cls_model=self.CRP_CLASSIFIER,
                                                                             ele_model=self.AWL_MODEL,
                                                                             login_model=self.CRP_LOCATOR_MODEL,
                                                                             driver=driver)
                crp_locator_time += process_time
                driver.quit()

                waive_crp_classifier = True  # only run dynamic analysis ONCE

                # If dynamic analysis did not reach a CRP
                if not successful:
                    print('Dynamic analysis cannot find any link redirected to a CRP page, report as benign')
                    return phish_category, pred_target, matched_domain, plotvis, siamese_conf, \
                            str(awl_detect_time) + '|' + str(logo_match_time) + '|' + str(crp_class_time) + '|' + str(crp_locator_time), \
                            pred_boxes, pred_classes

                else:  # dynamic analysis successfully found a CRP
                    print('Dynamic analysis found a CRP, go back to layout detector')

            else:  # already a CRP page
                print('Already a CRP, continue')
                break

        ######################## Step5: Return #################################
        if pred_target is not None:
            print('Phishing is found!')
            phish_category = 1
            # Visualize, add annotations
            cv2.putText(plotvis, "Target: {} with confidence {:.4f}".format(pred_target, siamese_conf),
                        (int(matched_coord[0] + 20), int(matched_coord[1] + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return phish_category, pred_target, matched_domain, plotvis, siamese_conf, \
                    str(awl_detect_time) + '|' + str(logo_match_time) + '|' + str(crp_class_time) + '|' + str(crp_locator_time), \
                    pred_boxes, pred_classes

if __name__ == '__main__':

    '''update domain map'''
    # with open('./lib/phishpedia/models/domain_map.pkl', "rb") as handle:
    #     domain_map = pickle.load(handle)
    #
    # domain_map['weibo'] = ['sina', 'weibo']
    #
    # with open('./lib/phishpedia/models/domain_map.pkl', "wb") as handle:
    #     pickle.dump(domain_map, handle)
    # exit()

    '''run'''
    today = datetime.now().strftime('%Y%m%d')

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=str)
    parser.add_argument("--output_txt", default=f'{today}_results.txt', help="Output txt path")
    args = parser.parse_args()

    request_dir = args.folder
    phishintention_cls = PhishIntentionWrapper()
    result_txt = args.output_txt

    os.makedirs(request_dir, exist_ok=True)

    for folder in tqdm(os.listdir(request_dir)):
        html_path = os.path.join(request_dir, folder, "html.txt")
        screenshot_path = os.path.join(request_dir, folder, "shot.png")
        info_path = os.path.join(request_dir, folder, 'info.txt')

        if not os.path.exists(screenshot_path):
            continue

        # url = eval(open(info_path).read())['url']
        url = open(info_path).read()

        if os.path.exists(result_txt) and url in open(result_txt, encoding='ISO-8859-1').read():
            continue

        _forbidden_suffixes = r"\.(mp3|wav|wma|ogg|mkv|zip|tar|xz|rar|z|deb|bin|iso|csv|tsv|dat|txt|css|log|sql|xml|sql|mdb|apk|bat|bin|exe|jar|wsf|fnt|fon|otf|ttf|ai|bmp|gif|ico|jp(e)?g|png|ps|psd|svg|tif|tiff|cer|rss|key|odp|pps|ppt|pptx|c|class|cpp|cs|h|java|sh|swift|vb|odf|xlr|xls|xlsx|bak|cab|cfg|cpl|cur|dll|dmp|drv|icns|ini|lnk|msi|sys|tmp|3g2|3gp|avi|flv|h264|m4v|mov|mp4|mp(e)?g|rm|swf|vob|wmv|doc(x)?|odt|rtf|tex|txt|wks|wps|wpd)$"
        if re.search(_forbidden_suffixes, url, re.IGNORECASE):
            continue

        phish_category, pred_target, matched_domain, \
                plotvis, siamese_conf, runtime_breakdown, \
                pred_boxes, pred_classes = phishintention_cls.test_orig_phishintention(url, screenshot_path)

        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                f.write(folder + "\t")
                f.write(url + "\t")
                f.write(str(phish_category) + "\t")
                f.write(str(pred_target) + "\t")  # write top1 prediction only
                f.write(str(matched_domain) + "\t")
                f.write(str(siamese_conf) + "\t")
                f.write(runtime_breakdown + "\n")
        except UnicodeError:
            with open(result_txt, "a+", encoding='utf-8') as f:
                f.write(folder + "\t")
                f.write(url + "\t")
                f.write(str(phish_category) + "\t")
                f.write(str(pred_target) + "\t")  # write top1 prediction only
                f.write(str(matched_domain) + "\t")
                f.write(str(siamese_conf) + "\t")
                f.write(runtime_breakdown + "\n")

        if phish_category:
            os.makedirs(os.path.join(request_dir, folder), exist_ok=True)
            cv2.imwrite(os.path.join(request_dir, folder, "predict.png"), plotvis)

