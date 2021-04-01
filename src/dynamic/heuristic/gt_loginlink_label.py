

from src.util.chrome import *
from src.credential_classifier.bit_pytorch.utils import read_xml
import os
import time
from tqdm import tqdm
from selenium.common.exceptions import TimeoutException, NoAlertPresentException

######################### Temporal scripts ######################################
import helium
from seleniumwire import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from webdriver_manager.chrome import ChromeDriverManager

def initialize_chrome_settings(lang_txt:str):
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
    options.add_argument("--headless") # disable browser

    options.add_argument("--start-maximized")
    options.add_argument('--window-size=1920,1080') # fix screenshot size
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
    options.set_capability('unhandledPromptBehavior', 'dismiss') # dismiss

    return  options
# load driver ONCE
options = initialize_chrome_settings(lang_txt='./src/util/lang.txt')
capabilities = DesiredCapabilities.CHROME
capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
capabilities["unexpectedAlertBehaviour"] = "dismiss"  # handle alert

driver = webdriver.Chrome(ChromeDriverManager().install(), desired_capabilities=capabilities, chrome_options=options)
helium.set_driver(driver)

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
