from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import numpy as np
import torch


def cv_imread(filePath):
    '''
    When image path contains nonenglish characters, normal cv2.imread will have error
    :param filePath:
    :return:
    '''
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


def config_rcnn(cfg_path, weights_path, conf_threshold):
    '''
    Configure weights and confidence threshold
    :param cfg_path:
    :param weights_path:
    :param conf_threshold:
    :return:
    '''
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    # uncomment if you installed detectron2 cpu version
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'

    # Initialize model
    predictor = DefaultPredictor(cfg)
    return predictor

def pred_rcnn(im, predictor):
    '''
    Perform inference for RCNN
    :param im:
    :param predictor:
    :return:
    '''
    im = cv2.imread(im)

    if im is not None:
        if im.shape[-1] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    else:
        return None, None, None

    outputs = predictor(im)

    instances = outputs['instances']
    pred_classes = instances.pred_classes.detach().cpu()  # tensor
    pred_boxes = instances.pred_boxes.tensor.detach().cpu()  # Boxes object
    pred_scores = instances.scores  # tensor

    return pred_boxes, pred_classes, pred_scores


def find_element_type(pred_boxes, pred_classes, bbox_type='button'):
    '''
    Filter bboxes by type
    :param pred_boxes: torch.Tensor of shape Nx4, bounding box coordinates in (x1, y1, x2, y2)
    :param pred_classes: torch.Tensor of shape Nx1 0 for logo, 1 for input, 2 for button, 3 for label(text near input), 4 for block
    :param bbox_type: the type of box we want to find
    :return pred_boxes_after: pred_boxes after filtering
    :return pred_classes_after: pred_classes after filtering
    '''
    # global dict
    class_dict = {0: 'logo', 1: 'input', 2: 'button', 3: 'label', 4: 'block'}
    inv_class_dict = {v: k for k, v in class_dict.items()}
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
    class_dict = {0: 'logo', 1: 'input', 2: 'button', 3: 'label', 4: 'block'}
    check = cv2.imread(img_path)
    if pred_boxes is None or len(pred_boxes) == 0:
        return check
    pred_boxes = pred_boxes.numpy() if not isinstance(pred_boxes, np.ndarray) else pred_boxes
    pred_classes = pred_classes.numpy() if not isinstance(pred_classes, np.ndarray) else pred_classes

    # draw rectangles
    for j, box in enumerate(pred_boxes):
        cv2.rectangle(check, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (36, 255, 12), 2)
        cv2.putText(check, class_dict[pred_classes[j].item()], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

    return check



