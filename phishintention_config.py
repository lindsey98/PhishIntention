# Global configuration
from src.siamese import *
from src.element_detector import *
from src.credential import *
from src.util.chrome import *
from src.login_finder import *
import helium
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from webdriver_manager.chrome import ChromeDriverManager

def driver_loader():
    '''
    load chrome driver
    :return:
    '''

    options = initialize_chrome_settings(lang_txt='./src/util/lang.txt')
    capabilities = DesiredCapabilities.CHROME
    capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
    capabilities["unexpectedAlertBehaviour"] = "dismiss"  # handle alert
    capabilities["pageLoadStrategy"] = "eager"  # eager mode #FIXME: set eager mode, may load partial webpage

    driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(),
                              desired_capabilities=capabilities,
                              chrome_options=options)
    driver.set_page_load_timeout(60)  # set timeout to avoid wasting time
    driver.set_script_timeout(60)  # set timeout to avoid wasting time
    helium.set_driver(driver)
    return driver

# element recognition model
ele_cfg, ele_model = element_config(rcnn_weights_path = './src/element_detector/output/website_lr0.001/model_final.pth',
                                    rcnn_cfg_path='./src/element_detector/configs/faster_rcnn_web.yaml')

# CRP classifier -- mixed version
# cls_model = credential_config(checkpoint='./src/credential_classifier/output/hybrid/hybrid_lr0.005/BiT-M-R50x1V2_0.005.pth.tar',
#                               model_type='mixed')

cls_model = credential_config(checkpoint='./src/credential_classifier/output/Increase_resolution_lr0.005/BiT-M-R50x1V2_0.005.pth.tar',model_type='mixed')

login_cfg, login_model = login_config(
    rcnn_weights_path='./src/dynamic/login_finder/output/lr0.001_finetune/model_final.pth',
    rcnn_cfg_path='./src/dynamic/login_finder/configs/faster_rcnn_login_lr0.001_finetune.yaml')

# siamese model
print('Load protected logo list')
# pedia_model, logo_feat_list, file_name_list = phishpedia_config(num_classes=277,
#                                                 weights_path='./src/siamese_pedia/resnetv2_rgb_new.pth.tar',
#                                                 targetlist_path='./src/siamese_pedia/expand_targetlist/')

pedia_model, ocr_model, logo_feat_list, file_name_list = phishpedia_config_OCR(num_classes=277,
                                                weights_path='./src/siamese_OCR/output/targetlist_lr0.01/bit.pth.tar',
                                                ocr_weights_path='./src/siamese_OCR/demo_downgrade.pth.tar',
                                                targetlist_path='./src/siamese_pedia/expand_targetlist/')
print('Finish loading protected logo list')

siamese_ts = 0.87 # FIXME: threshold is 0.87 in phish-discovery?

# brand-domain dictionary
domain_map_path = './src/siamese_pedia/domain_map.pkl'

