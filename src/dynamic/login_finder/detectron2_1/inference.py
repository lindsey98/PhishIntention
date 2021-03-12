from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import numpy as np

def login_element_config(rcnn_weights_path: str, rcnn_cfg_path: str):
    '''
    Load element detector configurations
    :param rcnn_weights_path: path to rcnn weights
    :param rcnn_cfg_path: path to configuration file
    :return cfg: rcnn cfg
    :return model: rcnn model
    '''
    # merge configuration
    cfg = get_cfg()
    cfg.merge_from_file(rcnn_cfg_path)
    cfg.MODEL.WEIGHTS = rcnn_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 # lower this threshold to report more boxes
    
    # initialize model
    model = DefaultPredictor(cfg)
    return cfg, model

def login_element_recognition(img, model):
    '''
    Recognize elements from a screenshot
    :param img: [str|np.ndarray]
    :param model: rcnn model
    :return pred_classes: torch.Tensor of shape Nx1 0 for logo, 1 for input, 2 for button, 3 for label(text near input), 4 for block
    :return pred_boxes: torch.Tensor of shape Nx4, bounding box coordinates in (x1, y1, x2, y2)
    :return pred_scores: torch.Tensor of shape Nx1, prediction confidence of bounding boxes
    '''
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
    else:
        img = img
        
    pred = model(img)
    pred_i = pred["instances"].to("cpu")
    pred_classes = pred_i.pred_classes # Boxes types
    pred_boxes = pred_i.pred_boxes.tensor # Boxes coords
    pred_scores = pred_i.scores # Boxes prediction scores

    return pred_classes, pred_boxes, pred_scores


