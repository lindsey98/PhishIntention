import os
from phishintention.src.util.chrome import *
import re
import time
from tqdm import tqdm
from selenium.common.exceptions import TimeoutException, NoAlertPresentException
import json
from phishintention.src.crp_locator import login_config, login_recognition, dynamic_analysis
from phishintention.src.AWL_detector import vis
import cv2
from phishintention.src.AWL_detector import element_config
from phishintention.src.crp_classifier import credential_config
import numpy as np
from bs4 import BeautifulSoup as Soup


def temporal_driver(lang_txt:str):
    '''
    initialize chrome settings
    :return:
    '''
    # enable translation
    white_lists = {}

    with open(lang_txt) as langf:
        for i in langf.readlines():
            i = i.strip()
            text = i.split(' ')
            print(text)
            white_lists[text[1]] = 'en'
    prefs = {
        "translate": {"enabled": "true"},
        "translate_whitelists": white_lists
    }

    options = webdriver.ChromeOptions()

    options.add_experimental_option("prefs", prefs)
    options.add_argument('--ignore-certificate-errors') # ignore errors
    options.add_argument('--ignore-ssl-errors')
    # options.add_argument("--headless") # disable browser (have some issues: https://github.com/mherrmann/selenium-python-helium/issues/47)
    options.add_argument('--no-proxy-server')
    options.add_argument("--proxy-server='direct://'")
    options.add_argument("--proxy-bypass-list=*")

    options.add_argument("--start-maximized")
    options.add_argument('--window-size=1920,1080') # fix screenshot size
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
    options.set_capability('unhandledPromptBehavior', 'dismiss') # dismiss
    options.add_argument("disable-notifications")
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\Users\ruofan\Downloads",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })


    return options

def keyword_heuristic_debug(driver, orig_url, page_text, obfuscate=False):
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
    top3_urls = []

    if obfuscate:
        login_regex = re.compile(r"log in|login|log on|logon|signin|sign in|authenticat(e|ion)|(user|account|profile|dashboard)", re.I)
        page_text = login_regex.sub("l0gin", str(page_text))
        register_regex = re.compile(r"sign up|signup|regist(er|ration)|(create|new).*account", re.I)
        page_text = register_regex.sub("new_acc0unt", str(page_text))

    for i in page_text:
        # looking for keyword
        keyword_finder = re.findall('(login)|(log in)|(signup)|(sign up)|(sign in)|(submit)|(register)|(create.*account)|(join now)|(new user)|(my account)|(come in)|(check in)|(personal area)|(登入)|(登录)|(登錄)|(注册)|(Anmeldung)|(iniciar sesión)|(s\'identifier)|(ログインする)|(サインアップ)|(ログイン)|(로그인)|(가입하기)|(시작하기)|(регистрация)|(войти)|(вход)|(accedered)|(gabung)|(daftar)|(masuk)|(girişi)|(üye ol)|(وارد)|(عضویت)|(regístrate)|(acceso)|(acessar)|(entrar)|(giriş)สมัครสม|(Přihlásit)',
                                    i, re.IGNORECASE)
        if len(keyword_finder) > 0:
            ct += 1
            print("found")
            found_kw = [x for x in keyword_finder[0] if len(x) > 0][0]
            print(found_kw)
            if len(i) <= 20:
                click_text(i)
            else:
                click_text(found_kw)
            print('Successfully click')
            # save redirected url
            try:
                current_url = driver.current_url
                top3_urls.append(current_url)
                ct += 1 # count +1
            except TimeoutException as e:
                print(e)
                pass

            try:
                driver.get(orig_url)
                time.sleep(2)
                # click_popup()
                alert_msg = driver.switch_to.alert.text
                driver.switch_to.alert.dismiss()
            except TimeoutException as e:
                print(str(e))
                break
            except Exception as e:
                print(str(e))
                print("no alert")

        # Only check Top 3
        if ct >= 3:
            break

    return top3_urls

def cv_heuristic_debug(driver, orig_url, old_screenshot_path):
    '''
    CV based login finder
    :param driver: chrome driver
    :param orig_url: original URL
    :param old_screenshot_path: original screenshot path
    :return:
    '''
    # CV-based login finder
    # predict elements
    pred_classes, pred_boxes, _ = login_recognition(img=old_screenshot_path, model=login_model)

    #### TODO: remove this
    os.makedirs('debug', exist_ok=True)
    check = vis(old_screenshot_path, pred_boxes, pred_classes)
    cv2.imwrite(os.path.join('debug', '{}.png'.format(os.path.basename(os.path.dirname(old_screenshot_path)))), check)
    ############################

    # # visualize elements
    top3_urls = []
    # if no prediction at all
    if len(pred_boxes) == 0:
        return []

    for bbox in pred_boxes.detach().cpu().numpy()[:min(3, len(pred_boxes))]:  # only for top3 boxes
        x1, y1, x2, y2 = bbox
        print(bbox)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)  # click the center point
        click_point(center[0], center[1])  # redirect to that page

        # save redirected url
        try:
            current_url = driver.current_url
            top3_urls.append(current_url)
        except TimeoutException as e:
            pass


        try:
            driver.get(orig_url)
            time.sleep(2)
            # click_popup()
            alert_msg = driver.switch_to.alert.text
            driver.switch_to.alert.dismiss()
        except TimeoutException as e:
            print(str(e))
            break
        except Exception as e:
            print(str(e))
            print("no alert")

    return top3_urls

def rel2abs(html_path):
    '''
        Replace relative URLs in html to absolute URLs
        :param html_path: path to .html file
    '''
    soup = Soup(open(html_path, encoding='iso-8859-1').read(), features="lxml")
    if not soup.find('base'):
        head = soup.find('head')
        if head is not None:
            head = next(head.children, None)
            if head is not None:
                base = soup.new_tag('base')
                base['href'] = os.path.basename(html_path).split('.html')[0]
                head.insert_before(base)
    return soup.prettify()

if __name__ == '__main__':

    ############################ Temporal scripts ################################################################################################################
    from seleniumwire import webdriver
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
    from webdriver_manager.chrome import ChromeDriverManager
    import helium

    # load driver
    options = temporal_driver(lang_txt='./src/util/lang.txt')
    capabilities = DesiredCapabilities.CHROME
    capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
    capabilities["unexpectedAlertBehaviour"] = "dismiss"  # handle alert
    capabilities["pageLoadStrategy"] = "eager" # FIXME: eager load

    driver = webdriver.Chrome(ChromeDriverManager().install(),
                              desired_capabilities=capabilities,
                              chrome_options=options)
    driver.set_page_load_timeout(30)  # set timeout to avoid wasting time
    driver.set_script_timeout(30)  # set timeout to avoid wasting time
    driver.implicitly_wait(30)
    helium.set_driver(driver)
    ##################################################################################################################################################################
    # element recognition model
    ele_cfg, ele_model = element_config(
        rcnn_weights_path='./src/AWL_detector_utils/output/website_lr0.001/model_final.pth',
        rcnn_cfg_path='./src/AWL_detector_utils/configs/faster_rcnn_web.yaml')

    cls_model = credential_config(
        checkpoint='./src/crp_classifier_utils/output/Increase_resolution_lr0.005/BiT-M-R50x1V2_0.005.pth.tar',
        model_type='mixed')

    login_cfg, login_model = login_config(
        rcnn_weights_path='./src/crp_locator_utils/login_finder/output/lr0.001_finetune/model_final.pth',
        rcnn_cfg_path='./src/crp_locator_utils/login_finder/configs/faster_rcnn_login_lr0.001_finetune.yaml')

    # 600 URLs
    legitimate_folder = 'D:/ruofan/xdriver3-open/1003_legitimate_loginbutton_labelled/460_legitimate'
    urldict = {}
    if os.path.exists('./datasets/460_legitimate_detectedURL_eager_obfuscate.json'):
        with open('./datasets/460_legitimate_detectedURL_eager_obfuscate.json', 'rt', encoding='utf-8') as handle:
            urldict = json.load(handle)
    print(urldict)

    '''Runtime investigation'''
    # for kk, folder in tqdm(enumerate(os.listdir(legitimate_folder))):
    #     if kk < 462:
    #         continue
    #     print(folder)
    #     screenshot_path = os.path.join(legitimate_folder, folder, 'shot.png')
    #     info_path = screenshot_path.replace('shot.png', 'info.txt')
    #     url = open(info_path, encoding='utf-8').read()
    #     start_time = time.time()
    #     _, _, successful, process_time = dynamic_analysis(url=url, screenshot_path=screenshot_path,
    #                                                       cls_model=cls_model, ele_model=ele_model,
    #                                                       login_model=login_model,
    #                                                       driver=driver)
    #     total_time = time.time() - start_time
    #     if folder not in open('./datasets/460_legitimate_runtime.txt').read():
    #         with open('./datasets/460_legitimate_runtime.txt', 'a+') as f:
    #             f.write(folder+'\t'+str(process_time)+'\t'+str(total_time)+'\n')
    #
    # dynamic_total = [float(x.split('\t')[-1]) for x in open('./datasets/460_legitimate_runtime.txt').readlines()] + \
    #     [float(x.split('\t')[-1]) for x in open('./datasets/600_legitimate_runtime.txt').readlines()]
    # dynamic_partial = [float(x.split('\t')[-2]) for x in open('./datasets/460_legitimate_runtime.txt').readlines()] + \
    #     [float(x.split('\t')[-2]) for x in open('./datasets/600_legitimate_runtime.txt').readlines()]
    # print(np.min(dynamic_total), np.median(dynamic_total), np.mean(dynamic_total), np.max(dynamic_total))
    # print(np.min(dynamic_partial), np.median(dynamic_partial),  np.mean(dynamic_partial), np.max(dynamic_partial))

        # break

    html_obfuscate = True
    for kk, folder in tqdm(enumerate(os.listdir(legitimate_folder))):

        # if kk<=434: continue

        #### TODO: remote this, this is only for single url testing
        if folder != 'mediamarkt.de': continue
        #########

        old_screenshot_path = os.path.join(legitimate_folder, folder, 'shot.png')
        old_html_path = old_screenshot_path.replace('shot.png', 'html.txt')
        old_info_path = old_screenshot_path.replace('shot.png', 'info.txt')

        # if not crawled properly
        if not os.path.exists(old_info_path): continue

        # convert .txt to .html
        if os.path.exists(old_html_path):
            soup = rel2abs(old_html_path)
            with open(os.path.join(old_html_path.replace('html.txt', '{}.html'.format(folder))), "w", encoding='utf-8') as f:
                f.write(str(soup))
            soup = Soup(open(old_html_path, encoding='iso-8859-1').read(), features="lxml")
            soup = soup.prettify()
            with open(os.path.join(old_html_path.replace('html.txt', '{}.html'.format(folder))), "w", encoding='utf-8') as f:
                f.write(str(soup))
            old_html_path = old_html_path.replace('html.txt', '{}.html'.format(folder))

        # check whether have been visited before
        orig_url = open(old_info_path, encoding='utf-8').read()
        print('Current URL:', orig_url)
        domain_name = orig_url.split('//')[-1]
        # if domain_name in urldict.keys():
        #     if len(urldict[domain_name]) > 0:
        #         continue

        # initial visit
        start_time = time.time()
        try:
            driver.get(orig_url)
            # driver.get(os.path.join('file://', old_html_path))
            time.sleep(2)
            # click_popup()
            alert_msg = driver.switch_to.alert.text
            driver.switch_to.alert.dismiss()
        except TimeoutException as e:
            print(str(e))
            clean_up_window(driver)  # clean up the windows
            continue
        except Exception as e:
            print(str(e))
            print("no alert")
        print('Finish loading URL twice {:.4f}'.format(time.time() - start_time))

        print("getting url")
        start_time = time.time()
        page_text = get_page_text(driver).split('\n')  # tokenize by space
        print('Finish getting HTML text {:.4f}'.format(time.time() - start_time))

        # HTML heuristic first FIXME: check CRP for original URL first
        start_time = time.time()
        top3_urls_html = keyword_heuristic_debug(driver=driver, orig_url=orig_url, page_text=page_text, obfuscate=html_obfuscate)
        # top3_urls_html = keyword_heuristic_debug(driver=driver,
        #                                          orig_url=os.path.join('file://', old_html_path),
        #                                          page_text=page_text,
        #                                          obfuscate=True)
        print('After HTML keyword finder:', top3_urls_html); print('Finish HTML login finder {:.4f}'.format(time.time() - start_time))
        clean_up_window(driver)

        # go back to the original site to run CV model
        start_time = time.time()
        try:
            driver.get(orig_url) # FIXME: still load the original URL instead
            # driver.get(os.path.join('file://', old_html_path))
            time.sleep(2)
            # click_popup()
            alert_msg = driver.switch_to.alert.text
            driver.switch_to.alert.dismiss()
        except TimeoutException as e:
            print(str(e))
            clean_up_window(driver)  # clean up the windows
            continue
        except Exception as e:
            print(str(e))
            print("no alert")
        print('Finish loading original URL again {:.4f}'.format(time.time() - start_time))

        # save new screenshot
        driver.save_screenshot(os.path.join(legitimate_folder, folder, 'new.png'))
        start_time = time.time()
        top3_urls_cv = cv_heuristic_debug(driver=driver, orig_url=orig_url, old_screenshot_path=os.path.join(legitimate_folder, folder, 'new.png'))
        # top3_urls_cv = cv_heuristic_debug(driver=driver, orig_url=os.path.join('file://', old_html_path),
        #                                   old_screenshot_path=os.path.join(legitimate_folder, folder, 'new.png'))
        print('After CV finder', top3_urls_cv); print('Finish CV login finder {:.4f}'.format(time.time() - start_time))

        # append urls
        urldict[domain_name] = []
        for url in top3_urls_html:
            if legitimate_folder in url:
                url = ''.join(url.split(legitimate_folder +'/'+ folder+'/'+folder+'.html'))
                if not url.startswith('http'):
                    url = 'https://' + url
            url = url.split('.html')[0]
            urldict[domain_name].append(url)

        for url in top3_urls_cv:
            if legitimate_folder in url:
                url = ''.join(url.split(legitimate_folder +'/'+ folder+'/'+folder+'.html'))
                if not url.startswith('http'):
                    url = 'https://' + url
            url = url.split('.html')[0]
            urldict[domain_name].append(url)

        # write
        # with open('./datasets/600_legitimate_detectedURL_eager.json', 'wt', encoding='utf-8') as handle:
        #     json.dump(urldict, handle)
        # print(urldict)
        # with open('./datasets/460_legitimate_detectedURL_eager_obfuscate.json', 'wt', encoding='utf-8') as handle:
        #     json.dump(urldict, handle)

        clean_up_window(driver)
        driver.quit()
        exit()
