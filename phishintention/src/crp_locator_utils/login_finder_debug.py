from phishintention.src.crp_locator import login_config, login_recognition, vis
from phishintention.src.crp_locator_utils.login_finder_evaluate import keyword_heuristic_debug, temporal_driver
from selenium.common.exceptions import *
from phishintention.src.util.chrome import *
import cv2
import os
import numpy as np
from tqdm import tqdm

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
        except WebDriverException:
            pass

        return_success, driver = visit_url(driver, orig_url)
        if not return_success:
            break  # FIXME: TIMEOUT Error

    return top3_urls

########################### Temporal scripts ################################################################################################################
# load driver ONCE
from seleniumwire import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from webdriver_manager.chrome import ChromeDriverManager
import helium

# load driver
options = temporal_driver(lang_txt='./src/util/lang.txt', enable_translate=False)
options.add_argument('--headless') # FIXME
capabilities = DesiredCapabilities.CHROME
capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
capabilities["pageLoadStrategy"] = "eager"  # FIXME: eager load

driver = webdriver.Chrome(ChromeDriverManager().install(), desired_capabilities=capabilities,
                          chrome_options=options)
driver.set_page_load_timeout(30)  # set timeout to avoid wasting time
driver.set_script_timeout(30)  # set timeout to avoid wasting time
driver.implicitly_wait(5)
helium.set_driver(driver)

login_cfg, login_model = login_config(
    rcnn_weights_path='./src/dynamic/login_finder/output/lr0.001_finetune/model_final.pth',
    rcnn_cfg_path='./src/dynamic/login_finder/configs/faster_rcnn_login_lr0.001_finetune.yaml')

# FIXME:  If webdriver not working:  (1) enable/disable translation (2) enable click popup (3) headless

# 600 URLs
# legitimate_folder = './datasets/460_legitimate'
legitimate_folder = './datasets/600_legitimate'
ct = 0

# for folder in [x.strip() for x in open('./datasets/fail_login_finder_eager_460_HTML.txt').readlines()]:
# for folder in [x.strip() for x in open('./datasets/fail_login_finder_eager_600_HTML.txt').readlines()]:
for folder in tqdm(os.listdir(legitimate_folder)):

    ct += 1
    # if folder != 'books.com.tw': # HTML cannot, cv can in 460 legitimate examples: ccs-trick, https://ksl.com, https://avnori.com, https://teletica.com, https://scmp.com, https://tvn24.pl, https://allkpop.com
    #     continue

    old_screenshot_path = os.path.join(legitimate_folder, folder, 'shot.png')
    old_html_path = old_screenshot_path.replace('shot.png', 'html.txt')
    old_info_path = old_screenshot_path.replace('shot.png', 'info.txt')

    # get url
    init_time = time.time()
    orig_url = open(old_info_path, encoding='utf-8').read()
    print(folder)
    print(orig_url)
    domain_name = orig_url.split('//')[-1]

    #TODO: overall time including page loading time, remove page loading time
    #TODO: eager mode/normal mode
    start_time = time.time()
    visit_success, driver = visit_url(driver, orig_url)
    if not visit_success:
        continue
    time.sleep(5)
    visit_success, driver = visit_url(driver, orig_url)
    if not visit_success:
        continue

    print('Finish loading URL twice {:.4f}'.format(time.time() - start_time))

    print("getting url")
    time.sleep(5)
    start_time = time.time()
    page_text = get_page_text(driver).split('\n')  # tokenize by space
    print(page_text)
    print('Num token in HTML: ', len(page_text))
    print('Finish getting HTML text {:.4f}'.format(time.time() - start_time))

    # FIXME: check CRP for original URL first
    start_time = time.time()
    top3_urls_html = keyword_heuristic_debug(driver=driver, orig_url=orig_url, page_text=page_text)
    print('After HTML keyword finder:', top3_urls_html)
    print('Finish HTML login finder {:.4f}'.format(time.time() - start_time))

    start_time = time.time()
    visit_success, driver = visit_url(driver, orig_url)
    if not visit_success:
        continue
    print('Finish loading original URL again {:.4f}'.format(time.time() - start_time))

    # FIXME: update the screenshots
    driver.save_screenshot(old_screenshot_path)
    #
    start_time = time.time()
    top3_urls_cv = cv_heuristic_debug(driver=driver, orig_url=orig_url, old_screenshot_path=old_screenshot_path)
    print('After CV finder', top3_urls_cv)
    print('Finish CV login finder {:.4f}'.format(time.time() - start_time))

    # if (np.sum(['login' in x for x in top3_urls_html]) > 0) or (np.sum(['signup' in x for x in top3_urls_html]) > 0): # HTML is correct for sure
    #     if len(top3_urls_cv) == 0 or np.sum([x==orig_url for x in top3_urls_cv]) == len(top3_urls_cv): # CV cannot find
    #         print('Found! HTML correct, CV wrong', folder)
    #         break

    if len(top3_urls_html) == 0 or np.sum([x==orig_url for x in top3_urls_html]) == len(top3_urls_html): # HTML cannot find
        if (np.sum(['login' in x for x in top3_urls_cv]) > 0) or (np.sum(['signup' in x for x in top3_urls_cv]) > 0):
            print('Found! HTML wrong, CV correct', folder)
            break
    clean_up_window(driver)

driver.quit()


'''Getting gt URLS'''
#################################################################################

legitimate_folder = './datasets/600_legitimate'
xml_folder = './datasets/600_legitimate_xml'
write_txt_path = './datasets/gt_loginurl_for600.txt'

# with open(write_txt_path, 'w+') as f:
#     f.write('folder\turl\n')

for folder in tqdm(os.listdir(legitimate_folder)):

    url = open(os.path.join(legitimate_folder, folder, 'info.txt'), encoding='utf-8').read()
    domain_name = url.split('//')[-1]

    if domain_name in open(write_txt_path).read(): # already written
        continue

    if domain_name + '.xml' in os.listdir(xml_folder):

        # get url
        orig_url = url
        try:
            print("getting url")
            driver.get(orig_url)
            alert_msg = driver.switch_to.alert.text
            driver.switch_to.alert.dismiss()
            time.sleep(1)
        except TimeoutException as e:
            print(str(e))
            continue # cannot visit
        except Exception as e:
            print(str(e))
            print("no alert")

        # read labelled ground-truth
        _, gt_coords = read_xml(os.path.join(xml_folder, domain_name + '.xml'))
        for bbox in gt_coords:
            x1, y1, x2, y2 = bbox
            # print(bbox)
            center = ((x1+x2)/2, (y1+y2)/2)
            click_point(center[0], center[1])
            try:
                current_url = driver.current_url
                with open(write_txt_path, 'a+') as f:
                    f.write(domain_name + '\t' + current_url + '\n')
            except TimeoutException as e:
                pass

            try:
                print("getting url")
                driver.get(orig_url) # go back
                alert_msg = driver.switch_to.alert.text
                driver.switch_to.alert.dismiss()
                time.sleep(1)
            except TimeoutException as e:
                print(str(e))
                # continue
                break # cannot go back
            except NoAlertPresentException as e:
                print("no alert")

    clean_up_window(driver)
