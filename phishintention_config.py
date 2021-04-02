# Global configuration
# from src.layout import *
from src.siamese import *
from src.element_detector import *
from src.credential import *
from src.util.chrome import *
from src.login_finder import *
import helium
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from webdriver_manager.chrome import ChromeDriverManager

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


