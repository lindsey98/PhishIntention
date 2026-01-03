# Global configuration
import yaml
from modules.awl_detector import config_rcnn
from modules.crp_classifier import credential_config
from modules.logo_matching import siamese_model_config, ocr_model_config, cache_reference_list
import os

def get_absolute_path(relative_path):
    base_path = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base_path, relative_path))

def load_config(reload_targetlist=False):

    with open(os.path.join(os.path.dirname(__file__), 'configs/configs.yaml')) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    # Iterate through the configuration and update paths
    # Need to test on github actions, do not use absolute path
    '''
    for section, settings in configs.items():
        for key, value in settings.items():
            if 'PATH' in key and isinstance(value, str):  # Check if the key indicates a path
                absolute_path = get_absolute_path(value)
                configs[section][key] = absolute_path
    '''

    AWL_MODEL = config_rcnn(cfg_path=configs['AWL_MODEL']['CFG_PATH'],
                                        weights_path=configs['AWL_MODEL']['WEIGHTS_PATH'],
                                        conf_threshold=configs['AWL_MODEL']['DETECT_THRE'])

    CRP_CLASSIFIER = credential_config(
                                    checkpoint=configs['CRP_CLASSIFIER']['WEIGHTS_PATH'],
                                    model_type=configs['CRP_CLASSIFIER']['MODEL_TYPE'])

    CRP_LOCATOR_MODEL = config_rcnn(
                                cfg_path=configs['CRP_LOCATOR']['CFG_PATH'],
                                weights_path=configs['CRP_LOCATOR']['WEIGHTS_PATH'],
                                conf_threshold=configs['CRP_LOCATOR']['DETECT_THRE'])

    # siamese model
    SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']

    import logging
    logger = logging.getLogger(__name__)
    logger.info('Loading protected logo list')
    targetlist_zip_path = configs['SIAMESE_MODEL']['TARGETLIST_PATH']
    targetlist_dir = os.path.dirname(targetlist_zip_path)
    zip_file_name = os.path.basename(targetlist_zip_path)
    targetlist_folder = zip_file_name.split('.zip')[0]
    full_targetlist_folder_dir = os.path.join(targetlist_dir, targetlist_folder)

    SIAMESE_MODEL = siamese_model_config(num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
                                         weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'])

    OCR_MODEL = ocr_model_config(weights_path = configs['SIAMESE_MODEL']['OCR_WEIGHTS_PATH'])

    LOGO_FEATS, LOGO_FILES = cache_reference_list(model=SIAMESE_MODEL,
                                                      ocr_model=OCR_MODEL,
                                                      targetlist_path=full_targetlist_folder_dir,
                                                      reload_targetlist=reload_targetlist)    
        
    logger.info('Finished loading protected logo list')

    DOMAIN_MAP_PATH = configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH']

    return AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH