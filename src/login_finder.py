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
from selenium.common.exceptions import TimeoutException, WebDriverException
# from src.element_detector import vis


# global dict
class_dict = {0: 'login'}
inv_class_dict = {v: k for k, v in class_dict.items()}

def cv_imread(filePath):
    '''
    When image path contains nonenglish characters, normal cv2.imread will have error
    :param filePath:
    :return:
    '''
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img

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
        if not isinstance(img, np.ndarray):
            img_init = cv2.imread(img)
            if img_init is None:
                img = cv_imread(img)
                if img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                img = img_init
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
    time_deduct = 0
    print(page_text)

    for i in page_text: # iterate over html text
        # looking for keyword
        keyword_finder = re.findall('(login)|(log in)|(signup)|(sign up)|(sign in)|(submit)|(register)|(create.*account)|(open an account)|(get free.*now)|(join now)|(new user)|(my account)|(come in)|(check in)|(personal area)|(登入)|(登录)|(登錄)|(登録)|(注册)|(Anmeldung)|(iniciar sesión)|(identifier)|(ログインする)|(サインアップ)|(ログイン)|(로그인)|(가입하기)|(시작하기)|(регистрация)|(войти)|(вход)|(accedered)|(gabung)|(daftar)|(masuk)|(girişi)|(üye ol)|(وارد)|(عضویت)|(regístrate)|(acceso)|(acessar)|(entrar)|(giriş)|(เข้าสู่ระบบ)|(สมัครสมาชิก)|(Přihlásit)|(mein konto)|(registrati)|(anmelden)|(me connecter)|(ingresar)|(mon allociné)|(accedi)|(мой профиль)|(حسابي)|(administrer)|(next)',
                                        i, re.IGNORECASE)
        if len(keyword_finder) > 0:
            ct += 1
            found_kw = [x for x in keyword_finder[0] if len(x) > 0][0]
            print("found {} in HTML".format(found_kw))

            # FIXME: If it is not a bulk of text, click on the original text, e.g. Please login signup ...
            if len(i) <= 20 or len(i) < len(found_kw):
                start_time = time.time()
                click_text(i)
                print('Successfully click')
                time_deduct += time.time() - start_time
            else: # otherwise click on keyword
                start_time = time.time()
                click_text(found_kw)
                print('Successfully click')
                time_deduct += time.time() - start_time

            # save redirected url
            try:
                start_time = time.time()
                current_url = driver.current_url
                driver.save_screenshot(new_screenshot_path)
                writetxt(new_html_path, driver.page_source)
                writetxt(new_info_path, str(current_url))
                time_deduct += time.time() - start_time
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
                print(e)
                pass
            except WebDriverException as e:
                print(e)
                pass
            except Exception as e:
                print(e)
                pass

            # FIXME: Back to the original site if CRP not found
            start_time = time.time()
            try:
                driver.get(orig_url)
                alert_msg = driver.switch_to.alert.text
                driver.switch_to.alert.dismiss()
            except TimeoutException as e:
                time_deduct += time.time() - start_time
                print(str(e))
                break  # FIXME: TIMEOUT Error
            except Exception as e:
                print(str(e))
                print("no alert")
            time_deduct += time.time() - start_time


        # Only check Top 3
        if ct >= 3:
            break

    return reach_crp, time_deduct

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
    pred_classes, pred_boxes, _ = login_recognition(img=old_screenshot_path, model=login_model)
    # # visualize elements
    # check = vis(old_screenshot_path, pred_boxes, pred_classes)
    # cv2.imshow(check)
    reach_crp = False
    time_deduct = 0
    # if no prediction at all
    if len(pred_boxes) == 0:
        return reach_crp, time_deduct

    for bbox in pred_boxes.detach().cpu().numpy()[: min(3, len(pred_boxes))]: # only for top3 boxes
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        start_time = time.time()
        click_point(center[0], center[1])  # click center point of predicted bbox for safe
        time_deduct += time.time() - start_time

        # save redirected url
        try:
            start_time = time.time()
            current_url = driver.current_url
            driver.save_screenshot(new_screenshot_path)
            writetxt(new_html_path, driver.page_source)
            writetxt(new_info_path, str(current_url))
            time_deduct += time.time() - start_time

            # Call CRP classifier
            # CRP HTML heuristic
            cre_pred = html_heuristic(new_html_path)
            # Credential classifier module
            if cre_pred == 1:  # if HTML heuristic report as nonCRP
                pred_classes_crp, pred_boxes_crp, _ = element_recognition(img=new_screenshot_path, model=ele_model)
                cre_pred, cred_conf, _ = credential_classifier_mixed_al(img=new_screenshot_path, coords=pred_boxes_crp,
                                                                        types=pred_classes_crp, model=cls_model)

            elif cre_pred == 0:  # this is an CRP already
                reach_crp = True
                break  # stop when reach an CRP already

        except TimeoutException as e:
            print(e)
            pass
        except WebDriverException as e:
            print(e)
            pass
        except Exception as e:
            print(e)

        # FIXME: Back to the original site if CRP not found
        start_time = time.time()
        try:
            driver.get(orig_url)
            alert_msg = driver.switch_to.alert.text
            driver.switch_to.alert.dismiss()
        except TimeoutException as e:
            print(str(e))
            time_deduct += time.time() - start_time
            break  # FIXME: TIMEOUT Error
        except Exception as e:
            print(str(e))
            print("no alert")
        time_deduct += time.time() - start_time

    return reach_crp, time_deduct


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
        # click_popup() # only click popup when first visit a website
        time.sleep(2)
        alert_msg = driver.switch_to.alert.text
        driver.switch_to.alert.dismiss()
    except TimeoutException as e:
        print(str(e))
        clean_up_window(driver)  # clean up the windows
        return url, screenshot_path, successful, 0
    except Exception as e:
        print(str(e))
        print("no alert") #FIXME: load twice because google translate not working the first time we visit a website
    try:
        driver.get(orig_url)
        alert_msg = driver.switch_to.alert.text
        driver.switch_to.alert.dismiss()
    except TimeoutException as e:
        print(str(e))
        clean_up_window(driver)  # clean up the windows
        return url, screenshot_path, successful, 0
    except Exception as e:
        print(str(e))
        print("no alert")
    time.sleep(5)

    start_time = time.time()
    print("Getting url")
    page_text = get_page_text(driver).split('\n')  # tokenize by \n

    # HTML heuristic based login finder
    reach_crp, time_deduct_html = keyword_heuristic(driver=driver, orig_url=orig_url, page_text=page_text,
                                  new_screenshot_path=new_screenshot_path, new_html_path=new_html_path,
                                  new_info_path=new_info_path, ele_model=ele_model, cls_model=cls_model)

    print('After HTML keyword finder:', reach_crp)
    total_time = time.time() - start_time - time_deduct_html

    # If HTML login finder did not find CRP, call CV-based login finder
    if not reach_crp:
        # FIXME: Ensure that it goes back to the original URL
        try:
            driver.get(orig_url)
            time.sleep(1)
            alert_msg = driver.switch_to.alert.text
            driver.switch_to.alert.dismiss()
        except TimeoutException as e:
            print(str(e))
            clean_up_window(driver)  # clean up the windows
            return url, screenshot_path, successful, total_time  # load URL unsucessful
        except Exception as e:
            print(str(e))
            print("no alert")

        start_time = time.time()
        reach_crp, time_deduct_cv = cv_heuristic(driver=driver, orig_url=orig_url, old_screenshot_path=screenshot_path,
                                 new_screenshot_path=new_screenshot_path, new_html_path=new_html_path,
                                 new_info_path=new_info_path, login_model=login_model, ele_model=ele_model, cls_model=cls_model)
        total_time += time.time() - start_time - time_deduct_cv
        print('After CV finder', reach_crp)

    # Final URL
    if os.path.exists(new_info_path):
        current_url = open(new_info_path, encoding='ISO-8859-1').read()
        current_ss = new_screenshot_path
        if len(current_url) == 0: # if current URL is empty
            current_url = orig_url # return original url and screenshot_path
            current_ss = screenshot_path
    else: # return original url and screenshot_path
        current_url = orig_url
        current_ss = screenshot_path

    clean_up_window(driver) # clean up the windows
    return current_url, current_ss, reach_crp, total_time
