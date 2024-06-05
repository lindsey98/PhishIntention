# Global configuration
import phishintention
from phishintention.src.OCR_aided_siamese import *
from phishintention.src.AWL_detector import *
from phishintention.src.crp_classifier import *
from phishintention.src.util.chrome import *
from phishintention.src.crp_locator import *
import helium
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from webdriver_manager.chrome import ChromeDriverManager
from typing import Union
import yaml
import subprocess


def driver_loader():
    '''
    load chrome driver
    :return:
    '''

    options = initialize_chrome_settings(lang_txt=os.path.join(os.path.dirname(__file__), 'src/util/lang.txt'))
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



def load_config(cfg_path: Union[str, None] = None, reload_targetlist=False, device='cuda'):

    #################### '''Default''' ####################
    if cfg_path is None:
        with open(os.path.join(os.path.dirname(__file__), 'configs.yaml')) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
    else:
        with open(cfg_path) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

    # element recognition model
    AWL_CFG_PATH = configs['AWL_MODEL']['CFG_PATH']
    AWL_WEIGHTS_PATH = configs['AWL_MODEL']['WEIGHTS_PATH']
    AWL_CONFIG, AWL_MODEL = element_config(rcnn_weights_path=AWL_WEIGHTS_PATH,
                                           rcnn_cfg_path=AWL_CFG_PATH, device=device)

    CRP_CLASSIFIER = credential_config(
        checkpoint=configs['CRP_CLASSIFIER']['WEIGHTS_PATH'],
        model_type=configs['CRP_CLASSIFIER']['MODEL_TYPE'])

    CRP_LOCATOR_CONFIG, CRP_LOCATOR_MODEL = login_config(
        rcnn_weights_path=configs['CRP_LOCATOR']['WEIGHTS_PATH'],
        rcnn_cfg_path=configs['CRP_LOCATOR']['CFG_PATH'],
        device=device)

    # siamese model
    print('Load protected logo list')
    if configs['SIAMESE_MODEL']['TARGETLIST_PATH'].endswith('.zip') \
            and not os.path.isdir('{}'.format(configs['SIAMESE_MODEL']['TARGETLIST_PATH'].split('.zip')[0])):
        subprocess.run('cd {} && unzip expand_targetlist.zip -d .'.format(os.path.dirname(configs['SIAMESE_MODEL']['TARGETLIST_PATH'])), shell=True)
        #subprocess.run(
        #     "unzip {} -d {}/".format(configs['SIAMESE_MODEL']['TARGETLIST_PATH'],
        #                              configs['SIAMESE_MODEL']['TARGETLIST_PATH'].split('.zip')[0] ),
        #     shell=True,
        #)
        #subprocess.run(
        #    "unzip {}".format(configs['SIAMESE_MODEL']['TARGETLIST_PATH']),
        #    shell=True,
        #)

    if os.path.exists(os.path.join(os.path.dirname(configs['SIAMESE_MODEL']['TARGETLIST_PATH']), 'LOGO_FEATS.npy')) and reload_targetlist == False:
        SIAMESE_MODEL, OCR_MODEL = phishpedia_config_OCR_easy(
            num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
            weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'],
            ocr_weights_path=configs['SIAMESE_MODEL']['OCR_WEIGHTS_PATH'],
            )
        LOGO_FEATS = np.load(os.path.join(os.path.dirname(configs['SIAMESE_MODEL']['TARGETLIST_PATH']), 'LOGO_FEATS.npy'))
        LOGO_FILES = np.load(os.path.join(os.path.dirname(configs['SIAMESE_MODEL']['TARGETLIST_PATH']), 'LOGO_FILES.npy'))

    else:
        SIAMESE_MODEL, OCR_MODEL, LOGO_FEATS, LOGO_FILES = phishpedia_config_OCR(
            num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
            weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'],
            ocr_weights_path=configs['SIAMESE_MODEL']['OCR_WEIGHTS_PATH'],
            targetlist_path=configs['SIAMESE_MODEL']['TARGETLIST_PATH'].split('.zip')[0])
        np.save(os.path.join(os.path.dirname(configs['SIAMESE_MODEL']['TARGETLIST_PATH']), 'LOGO_FEATS'), LOGO_FEATS)
        np.save(os.path.join(os.path.dirname(configs['SIAMESE_MODEL']['TARGETLIST_PATH']), 'LOGO_FILES'), LOGO_FILES)

    print('Finish loading protected logo list')

    SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']  # FIXME: threshold is 0.87 in phish-discovery?

    # brand-domain dictionary
    DOMAIN_MAP_PATH = configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH']

    return AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH

