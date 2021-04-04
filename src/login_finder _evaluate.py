import os
from src.util.chrome import *
import re
import time
from tqdm import tqdm
from selenium.common.exceptions import TimeoutException, NoAlertPresentException
import json
from src.login_finder import login_config, login_recognition

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
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
    options.set_capability('unhandledPromptBehavior', 'dismiss') # dismiss

    return  options

def keyword_heuristic_debug(driver, orig_url, page_text):
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

    for i in page_text:
        # looking for keyword
        keyword_finder = re.findall('(login)|(log in)|(signup)|(sign up)|(sign in)|(submit)|(register)|(create.*account)|(join now)|(new user)|(my account)|(entrance)|(come in)|(登入)|(登录)|(登錄)|(注册)|(Anmeldung)|(iniciar sesión)|(identifier)|(ログインする)|(サインアップ)|(ログイン)|(로그인)|(가입하기)|(시작하기)|(регистрация)|(Войти)|(вход)|(accedered)|(gabung)|(daftar)|(masuk)|(girişi)|(üye ol)|(وارد)|(عضویت)|(regístrate)|(acceso)|(acessar)|(entrar)|(giriş yap)|(เข้าสู่ระบบ)|(สมัครสมาชิก)|(Přihlási)',
                                    i.lower())
        if len(keyword_finder) > 0:
            print("found")
            click_text(i) # click that text
            # save redirected url
            try:
                current_url = driver.current_url
                top3_urls.append(current_url)
                ct += 1 # count +1
            except TimeoutException as e:
                pass

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
                continue
            except NoAlertPresentException as e:
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
    _, pred_boxes, _ = login_recognition(img=old_screenshot_path, model=login_model)
    # # visualize elements
    top3_urls = []
    # if no prediction at all
    if len(pred_boxes) == 0:
        return []

    # print(pred_boxes.detach().cpu().numpy())
    for bbox in pred_boxes.detach().cpu().numpy()[:min(3, len(pred_boxes))]: # only for top3 boxes
        x1, y1, x2, y2 = bbox
        print(bbox)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        click_point(center[0], center[1])  # redirect to that page

        # save redirected url
        try:
            current_url = driver.current_url
            top3_urls.append(current_url)
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
            continue
        except Exception as e:
            print(str(e))
            # print("no alert")

    return top3_urls

if __name__ == '__main__':

    ############################ Temporal scripts ################################################################################################################
    from seleniumwire import webdriver
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
    from webdriver_manager.chrome import ChromeDriverManager
    import helium

    login_cfg, login_model = login_config(rcnn_weights_path='./src/dynamic/login_finder/output/lr0.001_finetune/model_final.pth',
                                          rcnn_cfg_path='./src/dynamic/login_finder/configs/faster_rcnn_login_lr0.001_finetune.yaml')

    # 600 URLs
    legitimate_folder = './datasets/600_legitimate'
    # urldict = {}
    with open('./datasets/600_legitimate_detectedURL.json', 'rt', encoding='utf-8') as handle:
        urldict = json.load( handle)
    print(urldict)
    # debug_folder = './datasets/debug'
    # os.makedirs(debug_folder, exist_ok=True)

    for folder in tqdm(os.listdir(legitimate_folder)):

        old_screenshot_path = os.path.join(legitimate_folder, folder, 'shot.png')
        old_html_path = old_screenshot_path.replace('shot.png', 'html.txt')
        old_info_path = old_screenshot_path.replace('shot.png', 'info.txt')

        # get url
        if not os.path.exists(old_info_path):
            continue

        if folder not in open('./datasets/fail_login_finder.txt').read():
            continue

        orig_url = open(old_info_path, encoding='utf-8').read()
        print(orig_url)
        domain_name = orig_url.split('//')[-1]
        if domain_name in urldict.keys():
            if len(urldict[domain_name]) > 0:
                continue

        # load driver
        options = temporal_driver(lang_txt='./src/util/lang.txt')
        capabilities = DesiredCapabilities.CHROME
        capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
        capabilities["unexpectedAlertBehaviour"] = "dismiss"  # handle alert

        driver = webdriver.Chrome(ChromeDriverManager().install(), desired_capabilities=capabilities,
                                  chrome_options=options)
        driver.set_page_load_timeout(60)  # set timeout to avoid wasting time
        driver.set_script_timeout(60)  # set timeout to avoid wasting time
        helium.set_driver(driver)

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
            continue
        except Exception as e:
            print(str(e))
            print("no alert")

        print("getting url")
        page_text = get_page_text(driver).split()  # tokenize by \n or space
        page_text.sort(key=len)  # sort text according to length
        print('Num token in HTML: ', len(page_text))

        # debug screenshot
        # driver.save_screenshot(os.path.join(debug_folder, folder+'.png'))

        # FIXME: check CRP for original URL first
        top3_urls_html = keyword_heuristic_debug(driver=driver, orig_url=orig_url, page_text=page_text)
        print('After HTML keyword finder:',top3_urls_html)

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
            continue
        except Exception as e:
            print(str(e))
            print("no alert")

        top3_urls_cv = cv_heuristic_debug(driver=driver, orig_url=orig_url, old_screenshot_path=old_screenshot_path)
        print('After CV finder', top3_urls_cv)
        urldict[domain_name] = []
        urldict[domain_name].extend(top3_urls_html)
        urldict[domain_name].extend(top3_urls_cv)
        #
        with open('./datasets/600_legitimate_detectedURL.json', 'wt', encoding='utf-8') as handle:
            json.dump(urldict, handle)

        driver.quit()