from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import numpy as np
import os
from src.util.chrome import *
import re
from src.credential import *
from src.element_detector import *
import time
import pandas as pd
from tqdm import tqdm

# global dict
class_dict = {0: 'login'}
inv_class_dict = {v: k for k, v in class_dict.items()}

def login_config(rcnn_weights_path: str, rcnn_cfg_path: str, threshold=0.05):
    '''
    Load login button detector configurations
    :param rcnn_weights_path: path to rcnn weights
    :param rcnn_cfg_path: path to configuration file
    :return cfg: rcnn cfg
    :return model: rcnn model
    '''
    # merge configuration
    cfg = get_cfg()
    cfg.merge_from_file(rcnn_cfg_path)
    cfg.MODEL.WEIGHTS = rcnn_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # lower this threshold to report more boxes
    
    # initialize model
    model = DefaultPredictor(cfg)
    return cfg, model

def login_recognition(img, model):
    '''
    Recognize login button from a screenshot
    :param img: [str|np.ndarray]
    :param model: rcnn model
    :return pred_classes: torch.Tensor of shape Nx1 0 for login
    :return pred_boxes: torch.Tensor of shape Nx4, bounding box coordinates in (x1, y1, x2, y2)
    :return pred_scores: torch.Tensor of shape Nx1, prediction confidence of bounding boxes
    '''
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
    else:
        img = img
        
    pred = model(img)
    pred_i = pred["instances"].to("cpu")
    pred_classes = pred_i.pred_classes # Boxes types
    pred_boxes = pred_i.pred_boxes.tensor # Boxes coords
    pred_scores = pred_i.scores # Boxes prediction scores

    return pred_classes, pred_boxes, pred_scores

def login_vis(img_path, pred_boxes, pred_classes):
    '''
    Visualize rcnn predictions
    :param img_path: str
    :param pred_boxes: torch.Tensor of shape Nx4, bounding box coordinates in (x1, y1, x2, y2)
    :param pred_classes: torch.Tensor of shape Nx1 0 for logo, 1 for input, 2 for button, 3 for label(text near input), 4 for block
    :return None
    '''
    
    check = cv2.imread(img_path)
    if len(pred_boxes) == 0: # no element
        return check
    
    pred_boxes = pred_boxes.numpy() if not isinstance(pred_boxes, np.ndarray) else pred_boxes
    pred_classes = pred_classes.numpy() if not isinstance(pred_classes, np.ndarray) else pred_classes
    
    # draw rectangles
    for j, box in enumerate(pred_boxes):
        cv2.rectangle(check, (box[0], box[1]), (box[2], box[3]), (36, 255, 12), 2)
        cv2.putText(check, class_dict[pred_classes[j].item()], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return check

def keyword_heuristic(driver, orig_url, page_text,
                      new_screenshot_path, new_html_path, new_info_path):
    '''
    Keyword based login finder
    :param driver: chrome driver
    :param orig_url: original url
    :param page_text: html text for original url
    :param new_screenshot_path: where to save redirected screenshot
    :param new_html_path: where to save redirected html
    :param new_info_path: where to save redirected url
    :return: reach_crp: reach a CRP page or not at the end
    '''
    ct = 0 # count number of sign-up/login links
    reach_crp = False # reach a CRP page or not

    for i in page_text:
        # looking for keyword
        keyword_finder = re.findall('(login)|(log in)|(signup)|(sign up)|(sign in)|(submit)|(register)|(create.*account)|(join now)|(new user)|(my account)',
                                    i.lower())
        if len(keyword_finder) > 0:
            print("found")
            click_text(i) # click that text
            # save redirected url
            current_url = driver.current_url
            driver.save_screenshot(new_screenshot_path)
            writetxt(new_html_path, driver.page_source)
            writetxt(new_info_path, str(current_url))
            ct += 1 # count +1

            # Call CRP classifier
            # CRP HTML heuristic
            cre_pred = html_heuristic(new_html_path)
            # Credential classifier module
            if cre_pred == 1: # if HTML heuristic report as nonCRP
                pred_classes, pred_boxes, pred_scores = element_recognition(img=new_screenshot_path, model=ele_model)
                cre_pred, cred_conf, _  = credential_classifier_mixed_al(img=new_screenshot_path, coords=pred_boxes,
                                                                     types=pred_classes, model=cls_model)
            if cre_pred == 0: # this is an CRP
                reach_crp = True
                break # stop when reach an CRP already
            # Back to the original site if CRP not found
            try:
                driver.get(orig_url)
                alert_msg = driver.switch_to.alert.text
                driver.switch_to.alert.dismiss()
                time.sleep(1)
            except Exception as e:
                print(str(e))
                print("no alert")

        # Only check Top 3
        if ct >= 3:
            break

    return reach_crp

def cv_heuristic(driver, orig_url, old_screenshot_path,
                 new_screenshot_path, new_html_path, new_info_path):
    '''
    CV based login finder
    :param driver: chrome driver
    :param orig_url: original URL
    :param old_screenshot_path: original screenshot path
    :return:
    '''
    # CV-based login finder
    # predict elements
    _, pred_boxes, _ = login_recognition(img=old_screenshot_path, model=login_model)
    # # visualize elements
    # check = login_vis(img_path, pred_boxes, pred_classes)
    reach_crp = False
    # if no prediction at all
    if len(pred_boxes) == 0:
        return reach_crp

    for bbox in pred_boxes.detach().cpu().numpy()[: min(3, len(pred_boxes))]: # only for top3 boxes
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        click_point(center[0], center[1])  # redirect to that page

        # save redirected url
        current_url = driver.current_url
        driver.save_screenshot(new_screenshot_path)
        writetxt(new_html_path, driver.page_source)
        writetxt(new_info_path, str(current_url))

        # Call CRP classifier
        # CRP HTML heuristic
        cre_pred = html_heuristic(new_html_path)
        # Credential classifier module
        if cre_pred == 1:  # if HTML heuristic report as nonCRP
            pred_classes_crp, pred_boxes_crp, _ = element_recognition(img=new_screenshot_path, model=ele_model)
            cre_pred, cred_conf, _ = credential_classifier_mixed_al(img=new_screenshot_path, coords=pred_boxes_crp,
                                                                    types=pred_classes_crp, model=cls_model)

        if cre_pred == 0: # this is an CRP
            reach_crp = True
            break # stop when reach an CRP already

        try:
            driver.get(orig_url)  # go back to original url
            alert_msg = driver.switch_to.alert.text
            driver.switch_to.alert.dismiss()
            time.sleep(1)
        except Exception as e:
            print(str(e))
            print("no alert")

    return reach_crp

if __name__ == '__main__':

    ############################ Temporal scripts ################################################################################################################
    # element recognition model
    ele_cfg, ele_model = element_config(rcnn_weights_path = './src/element_detector/output/website_lr0.001/model_final.pth',
                                        rcnn_cfg_path='./src/element_detector/configs/faster_rcnn_web.yaml')

    # CRP classifier -- mixed version
    cls_model = credential_config(checkpoint='./src/credential_classifier/output/hybrid/hybrid_lr0.005/BiT-M-R50x1V2_0.005.pth.tar',
                                  model_type='mixed')
    ##############################################################################################################################################################

    # load configurations ONCE
    login_cfg, login_model = login_config(rcnn_weights_path='./src/dynamic/login_finder/output/lr0.001_finetune/model_final.pth',
                                          rcnn_cfg_path='./src/dynamic/login_finder/configs/faster_rcnn_login_lr0.001_finetune.yaml')

    # 600 URLs
    legitimate_urls = list(pd.read_csv('./datasets/alexa.csv', header=None).iloc[:, 0])

    for url in tqdm(legitimate_urls):
        domain_name = url.split('//')[-1]
        urldir = './datasets/600_legitimate'

        if os.path.exists(os.path.join(urldir, domain_name)):
            continue

        os.makedirs(os.path.join(urldir, domain_name), exist_ok=True)

        # get url
        orig_url = url
        new_screenshot_path = os.path.join(urldir, domain_name, 'new_shot.png')
        new_html_path = new_screenshot_path.replace('new_shot.png', 'new_html.txt')
        new_info_path = new_screenshot_path.replace('new_shot.png', 'new_info.txt')

        try:
            driver.get(orig_url)
            alert_msg = driver.switch_to.alert.text
            driver.switch_to.alert.dismiss()
            time.sleep(1)
        except Exception as e:
            print(str(e))
            print("no alert")

        print("getting url")
        page_text = get_page_text(driver).split('\n') # tokenize by \n
        page_text.sort(key=len) # sort text according to length
        print(len(page_text))

        # write original url
        old_screenshot_path =os.path.join(urldir, domain_name, 'shot.png')
        driver.save_screenshot(old_screenshot_path)
        writetxt(old_screenshot_path.replace('shot.png', 'html.txt'), driver.page_source)
        writetxt(old_screenshot_path.replace('shot.png', 'info.txt'), str(orig_url))

        #FIXME: check CRP for original URL first
        reach_crp = False
        reach_crp = keyword_heuristic(driver=driver, orig_url=orig_url, page_text=page_text,
                                      new_screenshot_path=new_screenshot_path, new_html_path=new_html_path, new_info_path=new_info_path)
        print('After HTML keyword finder:', reach_crp)

        if not reach_crp:
            reach_crp = cv_heuristic(driver=driver, orig_url=orig_url, old_screenshot_path=old_screenshot_path,
                                     new_screenshot_path=new_screenshot_path, new_html_path=new_html_path, new_info_path=new_info_path)
            print('After CV finder', reach_crp)

    driver.quit()