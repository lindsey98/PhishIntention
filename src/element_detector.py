from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import numpy as np

# global dict
class_dict = {0: 'logo', 1: 'input', 2:'button', 3:'label', 4:'block'}
inv_class_dict = {v: k for k, v in class_dict.items()}

def element_config(rcnn_weights_path: str, rcnn_cfg_path: str):
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

def element_recognition(img, model):
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

def find_element_type(pred_boxes, pred_classes, bbox_type='button'):
    '''
    Filter bboxes by type
    :param pred_boxes: torch.Tensor of shape Nx4, bounding box coordinates in (x1, y1, x2, y2)
    :param pred_classes: torch.Tensor of shape Nx1 0 for logo, 1 for input, 2 for button, 3 for label(text near input), 4 for block
    :param bbox_type: the type of box we want to find
    :return pred_boxes_after: pred_boxes after filtering
    :return pred_classes_after: pred_classes after filtering
    '''    
    assert bbox_type in ['logo', 'input', 'button', 'label', 'block']
    pred_boxes_after = pred_boxes[pred_classes == inv_class_dict[bbox_type]]
    pred_classes_after = pred_classes[pred_classes == inv_class_dict[bbox_type]]
    return pred_boxes_after, pred_classes_after
    
    
def vis(img_path, pred_boxes, pred_classes):
    '''
    Visualize rcnn predictions
    :param img_path: str
    :param pred_boxes: torch.Tensor of shape Nx4, bounding box coordinates in (x1, y1, x2, y2)
    :param pred_classes: torch.Tensor of shape Nx1 0 for logo, 1 for input, 2 for button, 3 for label(text near input), 4 for block
    :return None
    '''
    
    check = cv2.imread(img_path)
    if len(pred_boxes) == 0: # no element
        return check
    
    pred_boxes = pred_boxes.numpy() if not isinstance(pred_boxes, np.ndarray) else pred_boxes
    pred_classes = pred_classes.numpy() if not isinstance(pred_classes, np.ndarray) else pred_classes
    
    # draw rectangles
    for j, box in enumerate(pred_boxes):
        cv2.rectangle(check, (box[0], box[1]), (box[2], box[3]), (36, 255, 12), 2)
        cv2.putText(check, class_dict[pred_classes[j].item()], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return check
#     cv2.imwrite(img_path.split('.png')[0]+'_vis.png', check)

if __name__ == '__main__':
    # change this to your image path
    img_path = 'test.png'
    
    # load configurations
    ele_cfg, ele_model = element_config(rcnn_weights_path = 'model_final.pth', 
                                        rcnn_cfg_path='configs/faster_rcnn_web.yaml')
    # predict elements
    pred_classes, pred_boxes, pred_scores = element_recognition(img=img_path, model=ele_model)
#     print(pred_boxes)
    
    # visualize elements 
    vis(img_path, pred_boxes, pred_classes)
    
    # only get buttons
    pred_buttons, _ = find_element_type(pred_boxes, pred_classes, bbox_type='button')
#     print(pred_buttons)
    
