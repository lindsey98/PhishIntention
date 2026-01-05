from selenium.common.exceptions import TimeoutException, WebDriverException
from utils.web_utils import click_text, get_page_text, visit_url, click_point
from modules.crp_classifier import html_heuristic, credential_classifier_mixed
from modules.awl_detector import pred_rcnn
import os
import re
import time

def keyword_heuristic(driver, orig_url, page_text,
                      new_screenshot_path, new_html_path, new_info_path,
                      ele_model, cls_model):
    '''
    Keyword based login finder
    '''
    ct = 0 # count number of sign-up/login links
    reach_crp = False # reach a CRP page or not
    time_deduct = 0

    # URL after loading might be different from orig_url
    start_time = time.time()
    try:
        orig_url = driver.current_url
    except TimeoutException as e:
        print(e)
        pass
    except WebDriverException as e:
        print(e)
        pass
    time_deduct += time.time() - start_time

    for i in page_text: # iterate over html text
        # looking for keyword
        start_time = time.time()
        keyword_finder = re.findall('(login)|(log in)|(log on)|(signup)|(sign up)|(sign in)|(sign on)|(submit)|(register)|(create.*account)|(open an account)|(get free.*now)|(join now)|(new user)|(my account)|(come in)|(check in)|(personal area)|(logg inn)|(Log-in)|(become a member)|(customer centre)|(登入)|(登录)|(登錄)|(登録)|(注册)|(Anmeldung)|(iniciar sesión)|(identifier)|(ログインする)|(サインアップ)|(ログイン)|(로그인)|(가입하기)|(시작하기)|(регистрация)|(войти)|(вход)|(accedered)|(gabung)|(daftar)|(masuk)|(girişi)|(Giriş)|(สมัครสม)|(üye ol)|(وارد)|(عضویت)|(regístrate)|(acceso)|(acessar)|(entrar )|(giriş)|(เข้าสู่ระบบ)|(สมัครสมาชิก)|(Přihlásit)|(mein konto)|(registrati)|(anmelden)|(me connecter)|(ingresa)|(mon allociné)|(accedi)|(мой профиль)|(حسابي)|(administrer)|(next)|(entre )|(cadastre-se)|(είσοδος)|(entrance)|(start now)|(accessibilité)|(accéder)|(zaloguj)|(otwórz konto osobiste)|(đăng nhập)|(devam)|(your account)',
                                        i, re.IGNORECASE)
        time_deduct += time.time() - start_time
        if len(keyword_finder) > 0:
            ct += 1
            found_kw = [y for x in keyword_finder for y in x if len(y) > 0]
            if len(found_kw) == 1: # find only 1 keyword
                 found_kw = found_kw[0]
                 if len(i) <= 2*len(found_kw): # if the text is not long, click on text
                     start_time = time.time()
                     click_text(i)
                     try:
                         current_url = driver.current_url
                         if current_url == orig_url:  # if page is not redirected, try clicking the keyword instead
                             print(found_kw)
                             click_text(found_kw)
                     except TimeoutException as e:
                         print(e)
                         pass
                     except WebDriverException as e:
                         print(e)
                         pass
                     print('Successfully click')
                     time_deduct += time.time() - start_time

                 else: # otherwise click on keyword
                     start_time = time.time()
                     click_text(found_kw)
                     print('Successfully click')
                     time_deduct += time.time() - start_time

            else: # find at least 2 keywords in same bulk of text
                 found_kw = found_kw[0] # only click the first keyword
                 start_time = time.time()
                 click_text(found_kw)
                 print('Successfully click')
                 time_deduct += time.time() - start_time

            # save redirected url
            try:
                start_time = time.time()
                current_url = driver.current_url
                driver.save_screenshot(new_screenshot_path)
                with open(new_html_path, 'w', encoding='utf-8') as fw:
                    fw.write(driver.page_source)
                with open(new_info_path, 'w', encoding='utf-8') as fw:
                    fw.write(str(current_url))
                time_deduct += time.time() - start_time

                # Call CRP classifier
                # CRP HTML heuristic
                cre_pred = html_heuristic(new_html_path)
                # Credential classifier module
                if cre_pred == 1:  # if HTML heuristic report as nonCRP
                    pred_boxes, pred_classes, _ = pred_rcnn(im=new_screenshot_path, predictor=ele_model)
                    cre_pred = credential_classifier_mixed(img=new_screenshot_path, coords=pred_boxes,
                                                            types=pred_classes, model=cls_model)
                if cre_pred == 0:  # this is an CRP
                    reach_crp = True
                    break  # stop when reach an CRP already

            except Exception as e:
                print(e)

            # Back to the original site if CRP not found
            start_time = time.time()
            return_success, driver = visit_url(driver, orig_url)
            if not return_success:
                time_deduct += time.time() - start_time
                break  # TIMEOUT Error
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
    :param driver: chromedriver
    :param orig_url: original URL
    :param old_screenshot_path: old screenshot path
    :param new_screenshot_path: new screenshot path
    :param new_html_path: new html path
    :param new_info_path: new info path
    :param login_model: login button detector
    :param ele_model: element detector
    :param cls_model: CRP classifier
    :return reach_crp: reach CRP or not
    :return time_deduct: URL loading/clicking time
    '''

    # CV-based login finder
    # predict elements
    pred_boxes, pred_classes, _ = pred_rcnn(im=old_screenshot_path,
                                            predictor=login_model)
    # # visualize elements
    # check = vis(old_screenshot_path, pred_boxes, pred_classes)
    # cv2.imshow(check)
    reach_crp = False
    time_deduct = 0
    # if no prediction at all
    if pred_boxes is None or len(pred_boxes) == 0:
        return reach_crp, time_deduct

    for bbox in pred_boxes.numpy()[: min(3, len(pred_boxes))]: # only for top3 boxes
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        start_time = time.time()
        click_point(center[0], center[1])  # click center point of predicted bbox for safe
        time_deduct += time.time() - start_time

        # save redirected url
        try:
            start_time = time.time()
            current_url = driver.current_url
            driver.save_screenshot(new_screenshot_path) # save new screenshot
            with open(new_html_path, 'w', encoding='utf-8') as fw:
                fw.write(driver.page_source)
            with open(new_info_path, 'w', encoding='utf-8') as fw:
                fw.write(str(current_url))
            time_deduct += time.time() - start_time

            # Call CRP classifier
            # CRP HTML heuristic
            cre_pred = html_heuristic(new_html_path)
            # Credential classifier module
            if cre_pred == 1:  # if HTML heuristic report as nonCRP
                pred_boxes, pred_classes, _ = pred_rcnn(im=new_screenshot_path, predictor=ele_model)
                cre_pred = credential_classifier_mixed(img=new_screenshot_path, coords=pred_boxes,
                                                       types=pred_classes, model=cls_model)
            # stop when reach an CRP already
            if cre_pred == 0:  # this is an CRP already
                reach_crp = True
                break

        except Exception as e:
            print(e)

        # Back to the original site if CRP not found
        start_time = time.time()
        return_success, driver = visit_url(driver, orig_url)
        if not return_success:
            time_deduct += time.time() - start_time
            break  # TIMEOUT Error
        time_deduct += time.time() - start_time

    return reach_crp, time_deduct


def crp_locator(url, screenshot_path, login_model, ele_model, cls_model, driver):
    '''
    Dynamic analysis to find CRP
    '''

    # get url
    orig_url = url
    successful = False # reach CRP or not?
    # path to save redirected URL
    new_screenshot_path = screenshot_path.replace('shot.png', 'new_shot.png')
    new_html_path = new_screenshot_path.replace('new_shot.png', 'new_html.txt')
    new_info_path = new_screenshot_path.replace('new_shot.png', 'new_info.txt')

    visit_success, driver = visit_url(driver, orig_url)
    if not visit_success:
        return url, screenshot_path, successful, 0

    start_time = time.time()
    print("Getting url")
    page_text = get_page_text(driver).split('\n')  # tokenize by \n

    # HTML heuristic based login finder
    reach_crp, time_deduct_html = keyword_heuristic(driver=driver, orig_url=orig_url, page_text=page_text,
                                                      new_screenshot_path=new_screenshot_path, new_html_path=new_html_path,
                                                      new_info_path=new_info_path,
                                                    ele_model=ele_model,
                                                    cls_model=cls_model)

    print('After HTML keyword finder:', reach_crp)
    total_time = time.time() - start_time - time_deduct_html

    # If HTML login finder did not find CRP, call CV-based login finder
    if not reach_crp:
        # Ensure that it goes back to the original URL
        visit_success, driver = visit_url(driver, orig_url)
        if not visit_success:
            return url, screenshot_path, successful, total_time  # load URL unsucessful
        try:
            driver.save_screenshot(screenshot_path.replace('shot.png', 'shot4cv.png'))
        except Exception as e:
            return url, screenshot_path, successful, total_time  # save updated screenshot unsucessful

        start_time = time.time()
        reach_crp, time_deduct_cv = cv_heuristic(driver=driver,
                                                 orig_url=orig_url, old_screenshot_path=screenshot_path.replace('shot.png', 'shot4cv.png'),
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

    return current_url, current_ss, reach_crp, total_time
