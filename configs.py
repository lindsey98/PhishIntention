# Global configuration
import os
import subprocess

import numpy as np
import yaml

from modules.awl_detector import config_rcnn
from modules.crp_classifier import credential_config
from modules.logo_matching import (
    cache_reference_list,
    ocr_model_config,
    siamese_model_config,
)
from dataclasses import dataclass  
from typing import Optional  
import numpy as np  
  
@dataclass  
class ModelConfig:  
    """模型配置类，封装所有模型和参数"""  
    awl_model: any  
    crp_classifier: any  
    crp_locator_model: any  
    siamese_model: any  
    ocr_model: any  
    siamese_threshold: float  
    logo_features: np.ndarray  
    logo_files: np.ndarray  
    domain_map_path: str  
      
    @classmethod  
    def from_file(cls, config_path: str, reload_targetlist: bool = False):  
        """从配置文件创建ModelConfig实例"""  
        return cls(_load_from_config(config_path, reload_targetlist))

def get_absolute_path(relative_path):
    base_path = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base_path, relative_path))

def _load_yaml_config(config_path: str) -> dict:  
    """加载YAML配置文件"""  
    with open(config_path) as file:  
        configs = yaml.load(file, Loader=yaml.FullLoader)  
      
    # 转换相对路径为绝对路径  
    for section, settings in configs.items():  
        for key, value in settings.items():  
            if 'PATH' in key and isinstance(value, str):  
                configs[section][key] = get_absolute_path(value)  
      
    return configs  

def _load_or_cache_logo_features(configs: dict, siamese_model, ocr_model, reload_targetlist: bool = False):  
    """加载或缓存logo特征"""  
    targetlist_zip_path = configs['SIAMEE_MODEL']['TARGETLIST_PATH']  
    targetlist_dir = os.path.dirname(targetlist_zip_path)  
    zip_file_name = os.path.basename(targetlist_zip_path)  
    targetlist_folder = zip_file_name.split('.zip')[0]  
    full_targetlist_folder_dir = os.path.join(targetlist_dir, targetlist_folder)  
      
    if reload_targetlist or (not os.path.exists(os.path.join(os.path.dirname(__file__), 'LOGO_FEATS.npy'))):  
        LOGO_FEATS, LOGO_FILES = cache_reference_list(model=siamese_model,  
                                                      ocr_model=ocr_model,  
                                                      targetlist_path=full_targetlist_folder_dir)  
        print('Finish loading protected logo list')  
        np.save(os.path.join(os.path.dirname(__file__),'LOGO_FEATS.npy'), LOGO_FEATS)  
        np.save(os.path.join(os.path.dirname(__file__),'LOGO_FILES.npy'), LOGO_FILES)  
    else:  
        LOGO_FEATS, LOGO_FILES = np.load(os.path.join(os.path.dirname(__file__),'LOGO_FEATS.npy')), np.load(os.path.join(os.path.dirname(__file__),'LOGO_FILES.npy'))  
      
    return LOGO_FEATS, LOGO_FILES

def _initialize_models(configs: dict, reload_targetlist: bool = False) -> dict:  
    """初始化所有模型"""  
    # AWL模型  
    awl_model = config_rcnn(  
        cfg_path=configs['AWL_MODEL']['CFG_PATH'],  
        weights_path=configs['AWL_MODEL']['WEIGHTS_PATH'],  
        conf_threshold=configs['AWL_MODEL']['DETECT_THRE']  
    )  
      
    # CRP分类器  
    crp_classifier = credential_config(  
        checkpoint=configs['CRP_CLASSIFIER']['WEIGHTS_PATH'],  
        model_type=configs['CRP_CLASSIFIER']['MODEL_TYPE']  
    )  
      
    # CRP定位器  
    crp_locator_model = config_rcnn(  
        cfg_path=configs['CRP_LOCATOR']['CFG_PATH'],  
        weights_path=configs['CRP_LOCATOR']['WEIGHTS_PATH'],  
        conf_threshold=configs['CRP_LOCATOR']['DETECT_THRE']  
    )  
      
    # Siamese模型  
    siamese_threshold = configs['SIAMEE_MODEL']['MATCH_THRE']  
    siamese_model = siamese_model_config(  
        num_classes=configs['SIAMEE_MODEL']['NUM_CLASSES'],  
        weights_path=configs['SIAMEE_MODEL']['WEIGHTS_PATH']  
    )  
      
    # OCR模型  
    ocr_model = ocr_model_config(  
        weights_path=configs['SIAMEE_MODEL']['OCR_WEIGHTS_PATH']  
    )  
      
    # Logo特征缓存  
    logo_features, logo_files = _load_or_cache_logo_features(  
        configs, siamese_model, ocr_model, reload_targetlist  
    )  
      
    return {  
        'awl_model': awl_model,  
        'crp_classifier': crp_classifier,  
        'crp_locator_model': crp_locator_model,  
        'siamese_model': siamese_model,  
        'ocr_model': ocr_model,  
        'siamese_threshold': siamese_threshold,  
        'logo_features': logo_features,  
        'logo_files': logo_files,  
        'domain_map_path': configs['SIAMEE_MODEL']['DOMAIN_MAP_PATH']  
    }  
  
def _load_from_config(config_path: str, reload_targetlist: bool = False) -> dict:  
    """完整的配置加载流程"""  
    configs = _load_yaml_config(config_path)  
    return _initialize_models(configs, reload_targetlist)

def load_config(reload_targetlist: bool = False):  
    """保持原有接口，内部使用新的配置类"""  
    config_path = os.path.join(os.path.dirname(__file__), 'configs/configs.yaml')  
    model_config = ModelConfig.from_file(config_path, reload_targetlist)  
      
    # 返回原有格式的元组，保持兼容性  
    return (  
        model_config.awl_model,  
        model_config.crp_classifier,  
        model_config.crp_locator_model,  
        model_config.siamese_model,  
        model_config.ocr_model,  
        model_config.siamese_threshold,  
        model_config.logo_features,  
        model_config.logo_files,  
        model_config.domain_map_path  
    )

"""def load_config(reload_targetlist=False):

    with open(os.path.join(os.path.dirname(__file__), "configs/configs.yaml")) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    # Iterate through the configuration and update paths
    for section, settings in configs.items():
        for key, value in settings.items():
            if "PATH" in key and isinstance(
                value, str
            ):  # Check if the key indicates a path
                absolute_path = get_absolute_path(value)
                configs[section][key] = absolute_path

    AWL_MODEL = config_rcnn(
        cfg_path=configs["AWL_MODEL"]["CFG_PATH"],
        weights_path=configs["AWL_MODEL"]["WEIGHTS_PATH"],
        conf_threshold=configs["AWL_MODEL"]["DETECT_THRE"],
    )

    CRP_CLASSIFIER = credential_config(
        checkpoint=configs["CRP_CLASSIFIER"]["WEIGHTS_PATH"],
        model_type=configs["CRP_CLASSIFIER"]["MODEL_TYPE"],
    )

    CRP_LOCATOR_MODEL = config_rcnn(
        cfg_path=configs["CRP_LOCATOR"]["CFG_PATH"],
        weights_path=configs["CRP_LOCATOR"]["WEIGHTS_PATH"],
        conf_threshold=configs["CRP_LOCATOR"]["DETECT_THRE"],
    )

    # siamese model
    SIAMESE_THRE = configs["SIAMESE_MODEL"]["MATCH_THRE"]

    print("Load protected logo list")
    targetlist_zip_path = configs["SIAMESE_MODEL"]["TARGETLIST_PATH"]
    targetlist_dir = os.path.dirname(targetlist_zip_path)
    zip_file_name = os.path.basename(targetlist_zip_path)
    targetlist_folder = zip_file_name.split(".zip")[0]
    full_targetlist_folder_dir = os.path.join(targetlist_dir, targetlist_folder)

    SIAMESE_MODEL = siamese_model_config(
        num_classes=configs["SIAMESE_MODEL"]["NUM_CLASSES"],
        weights_path=configs["SIAMESE_MODEL"]["WEIGHTS_PATH"],
    )

    OCR_MODEL = ocr_model_config(
        weights_path=configs["SIAMESE_MODEL"]["OCR_WEIGHTS_PATH"]
    )

    if reload_targetlist or (
        not os.path.exists(os.path.join(os.path.dirname(__file__), "LOGO_FEATS.npy"))
    ):
        LOGO_FEATS, LOGO_FILES = cache_reference_list(
            model=SIAMESE_MODEL,
            ocr_model=OCR_MODEL,
            targetlist_path=full_targetlist_folder_dir,
        )
        print("Finish loading protected logo list")
        np.save(os.path.join(os.path.dirname(__file__), "LOGO_FEATS.npy"), LOGO_FEATS)
        np.save(os.path.join(os.path.dirname(__file__), "LOGO_FILES.npy"), LOGO_FILES)

    else:
        LOGO_FEATS, LOGO_FILES = np.load(
            os.path.join(os.path.dirname(__file__), "LOGO_FEATS.npy")
        ), np.load(os.path.join(os.path.dirname(__file__), "LOGO_FILES.npy"))

    DOMAIN_MAP_PATH = configs["SIAMESE_MODEL"]["DOMAIN_MAP_PATH"]

    return (
        AWL_MODEL,
        CRP_CLASSIFIER,
        CRP_LOCATOR_MODEL,
        SIAMESE_MODEL,
        OCR_MODEL,
        SIAMESE_THRE,
        LOGO_FEATS,
        LOGO_FILES,
        DOMAIN_MAP_PATH,
    )"""
