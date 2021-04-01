# Global configuration
from src.layout import *
from src.siamese import *
from src.element_detector import *
from src.credential import *
from src.util.chrome import *
from src.login_finder import *
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




# element recognition model
ele_cfg, ele_model = element_config(rcnn_weights_path = './src/element_detector/output/website_lr0.001/model_final.pth',
                                    rcnn_cfg_path='./src/element_detector/configs/faster_rcnn_web.yaml')

# CRP classifier -- mixed version
cls_model = credential_config(checkpoint='./src/credential_classifier/output/hybrid/hybrid_lr0.005/BiT-M-R50x1V2_0.005.pth.tar',
                              model_type='mixed')

login_cfg, login_model = login_config(
    rcnn_weights_path='./src/dynamic/login_finder/output/lr0.001_finetune/model_final.pth',
    rcnn_cfg_path='./src/dynamic/login_finder/configs/faster_rcnn_login_lr0.001_finetune.yaml')

# siamese model
print('Load protected logo list')
pedia_model, logo_feat_list, file_name_list = phishpedia_config(num_classes=277, 
                                                weights_path='./src/phishpedia/resnetv2_rgb_new.pth.tar',
                                                targetlist_path='./src/phishpedia/expand_targetlist/')

siamese_ts = 0.87 # FIXME: threshold is 0.87 in phish-discovery?

# brand-domain dictionary
domain_map_path = './src/phishpedia/domain_map.pkl'

# load driver ONCE
options = initialize_chrome_settings(lang_txt='./src/util/lang.txt')
capabilities = DesiredCapabilities.CHROME
capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
capabilities["unexpectedAlertBehaviour"] = "dismiss"  # handle alert

driver = webdriver.Chrome(ChromeDriverManager().install(), desired_capabilities=capabilities, chrome_options=options)
driver.set_page_load_timeout(60)
helium.set_driver(driver)

