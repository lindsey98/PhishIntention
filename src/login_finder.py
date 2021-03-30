from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import numpy as np
import os
from src.util.chrome import *
import re
from src.credential import *
from phishintention_config import ele_model, cls_model
from src.element_detector import *

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


if __name__ == '__main__':
    
#     img_path = "../test_sites/alexasetup.club/shot.png"
#
#     # load configurations ONCE
#     login_cfg, login_model = login_config(rcnn_weights_path='dynamic/login_finder/output/lr0.001_finetune/model_final.pth',                            rcnn_cfg_path='dynamic/login_finder/configs/faster_rcnn_login_lr0.001_finetune.yaml')
#
#     # predict elements
#     pred_classes, pred_boxes, pred_scores = login_recognition(img=img_path, model=login_model)
# #     print(len(pred_boxes))
#
#     # visualize elements
#     check = login_vis(img_path, pred_boxes, pred_classes)
#
#     cv2.imwrite('../test_sites/alexasetup.club/debug.png', check)

    # get url
    orig_url = "https://alexasetup.club/"
    new_screenshot_path = './test_sites/check.png'
    new_html_path = './test_sites/check.html'
    new_info_path = './test_sites/check.txt'

    driver.get(orig_url)
    print("getting url")
    page_text = get_page_text(driver)
    page_text = page_text.split('\n')

    ct = 0 # count number of sign-up/login links
    reach_crp = False # reach a CRP page or not
    for i in page_text:
        keyword_finder = re.findall('(login)|(log in)|(signup)|(sign up)|(sign in)|(submit)|(register)|(create.*account)|(join now)|(new user)|(my account)',
                                    i.lower())
        if len(keyword_finder) > 0:
            print("found")
            click_text(i)
            current_url = driver.current_url
            driver.save_screenshot(new_screenshot_path)
            writetxt(new_html_path, driver.page_source)
            writetxt(new_info_path, str(current_url))
            ct += 1

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
                break
            # Back to the original site
            driver.get(orig_url)

        # Only check Top 3
        if ct >= 3:
            break

    driver.close()