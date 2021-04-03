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
from selenium.common.exceptions import TimeoutException


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


def keyword_heuristic(driver, orig_url, page_text,
                      new_screenshot_path, new_html_path, new_info_path,
                      ele_model, cls_model):
    '''
    Keyword based login finder
   :param driver:
   :param orig_url:
   :param page_text:
   :param new_screenshot_path:
   :param new_html_path:
   :param new_info_path:
   :param ele_model:
   :param cls_model:
   :return:
    '''
    ct = 0 # count number of sign-up/login links
    reach_crp = False # reach a CRP page or not

    for i in page_text: # iterate over html text
        # looking for keyword
        keyword_finder = re.findall('(login)|(log in)|(signup)|(sign.*up)|(sign in)|(submit)|(register)|(create.*account)|(join now)|(new user)|(my account)',
                                    i.lower())
        if len(keyword_finder) > 0:
            print("found")
            click_text(i) # click that text

            # save redirected url
            try:
                current_url = driver.current_url
                driver.save_screenshot(new_screenshot_path)
                writetxt(new_html_path, driver.page_source)
                writetxt(new_info_path, str(current_url))
                ct += 1 # count +1

                # Call CRP classifier
                # CRP HTML heuristic
                cre_pred = html_heuristic(new_html_path)
                # Credential classifier module
                if cre_pred == 1:  # if HTML heuristic report as nonCRP
                    pred_classes, pred_boxes, pred_scores = element_recognition(img=new_screenshot_path,
                                                                                model=ele_model)
                    cre_pred, cred_conf, _ = credential_classifier_mixed_al(img=new_screenshot_path, coords=pred_boxes,
                                                                            types=pred_classes, model=cls_model)
                if cre_pred == 0:  # this is an CRP
                    reach_crp = True
                    break  # stop when reach an CRP already

            except TimeoutException as e:
                pass

            # Back to the original site if CRP not found
            try:
                driver.get(orig_url)
                time.sleep(5)
                if helium.Button("accept").exists():
                    helium.click(helium.Button("accept"))
                elif helium.Button("I accept").exists():
                    helium.click(helium.Button("I accept"))
                alert_msg = driver.switch_to.alert.text
                driver.switch_to.alert.dismiss()
                time.sleep(1)
            except TimeoutException as e:
                print(str(e))
                # continue
                break # cannot go back somehow
            except Exception as e:
                print(str(e))
                print("no alert")

        # Only check Top 3
        if ct >= 3:
            break

    return reach_crp

def cv_heuristic(driver, orig_url, old_screenshot_path,
                 new_screenshot_path, new_html_path, new_info_path,
                 login_model, ele_model, cls_model):
    '''
    CV based login finder
    :param driver:
    :param orig_url:
    :param old_screenshot_path:
    :param new_screenshot_path:
    :param new_html_path:
    :param new_info_path:
    :param login_model:
    :param ele_model:
    :param cls_model:
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
        click_point(center[0], center[1])  # click center point of predicted bbox

        # save redirected url
        try:
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

            if cre_pred == 0:  # this is an CRP
                reach_crp = True
                break  # stop when reach an CRP already

        except TimeoutException as e:
            pass

        try:
            driver.get(orig_url)  # go back to original url
            time.sleep(5)
            if helium.Button("accept").exists():
                helium.click(helium.Button("accept"))
            elif helium.Button("I accept").exists():
                helium.click(helium.Button("I accept"))
            alert_msg = driver.switch_to.alert.text
            driver.switch_to.alert.dismiss()
            time.sleep(1)
        except TimeoutException as e:
            print(str(e))
            # continue
            break # cannot go back somehow
        except Exception as e:
            print(str(e))
            print("no alert")

    return reach_crp


def dynamic_analysis(url, screenshot_path, login_model, ele_model, cls_model, driver):
    '''
    Dynamic analysis to find CRP
    :param url:
    :param screenshot_path:
    :param login_model:
    :param ele_model:
    :param cls_model:
    :param driver:
    :return:
    '''
    # get url
    orig_url = url
    successful = False # reach CRP or not?
    # path to save redirected URL
    new_screenshot_path = screenshot_path.replace('shot.png', 'new_shot.png')
    new_html_path = new_screenshot_path.replace('new_shot.png', 'new_html.txt')
    new_info_path = new_screenshot_path.replace('new_shot.png', 'new_info.txt')

    try:
        driver.get(orig_url)
        time.sleep(5)
        if helium.Button("accept").exists():
            helium.click(helium.Button("accept"))
        elif helium.Button("I accept").exists():
            helium.click(helium.Button("I accept"))
        alert_msg = driver.switch_to.alert.text
        driver.switch_to.alert.dismiss()
        time.sleep(1)
    except TimeoutException as e:
        print(str(e))
        return url, screenshot_path, successful # load URL unsucessful
    except Exception as e:
        print(str(e))
        print("no alert")

    print("Getting url")
    page_text = get_page_text(driver).split('\n')  # tokenize by \n
    page_text.sort(key=len)  # sort text according to length
    # print(len(page_text))

    # HTML heuristic based login finder
    reach_crp = keyword_heuristic(driver=driver, orig_url=orig_url, page_text=page_text,
                                  new_screenshot_path=new_screenshot_path, new_html_path=new_html_path,
                                  new_info_path=new_info_path, ele_model=ele_model, cls_model=cls_model)

    # print('After HTML keyword finder:', reach_crp)

    # If html login finder did not find CRP, call CV based login finder
    if not reach_crp:
        # Ensure that it goes back to the original URL
        try:
            driver.get(orig_url)
            time.sleep(5)
            if helium.Button("accept").exists():
                helium.click(helium.Button("accept"))
            elif helium.Button("I accept").exists():
                helium.click(helium.Button("I accept"))
            alert_msg = driver.switch_to.alert.text
            driver.switch_to.alert.dismiss()
            time.sleep(1)
        except TimeoutException as e:
            print(str(e))
            return url, screenshot_path, successful  # load URL unsucessful
        except Exception as e:
            print(str(e))
            print("no alert")

        reach_crp = cv_heuristic(driver=driver, orig_url=orig_url, old_screenshot_path=screenshot_path,
                                 new_screenshot_path=new_screenshot_path, new_html_path=new_html_path,
                                 new_info_path=new_info_path, login_model=login_model, ele_model=ele_model, cls_model=cls_model)
        # print('After CV finder', reach_crp)

    # Final URL
    if os.path.exists(new_info_path):
        current_url = open(new_info_path, encoding='utf-8').read()
        current_ss = new_screenshot_path
        if len(current_url) == 0:
            current_url = orig_url
            current_ss = screenshot_path
    else:
        current_url = orig_url
        current_ss = screenshot_path

    return current_url, current_ss, reach_crp
