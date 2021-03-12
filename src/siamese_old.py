from phishpedia.models import KNOWN_MODELS
from phishpedia.utils import brand_converter
from phishpedia.inference import siamese_inference, pred_siamese, siamese_inference_debug
from phishpedia.utils import brand_converter
import torch
import os
import numpy as np
from collections import OrderedDict


def phishpedia_classifier_old(pred_classes, pred_boxes, 
                          domain_map_path:str,
                          model, logo_feat_list, file_name_list, shot_path:str, 
                          url:str, 
                          ts:float):
    '''
    Run siamese
    :param pred_classes: torch.Tensor/np.ndarray Nx1 predicted box types
    :param pred_boxes: torch.Tensor/np.ndarray Nx4 predicted box coords
    :param domain_map_path: path to domain map dict
    :param model: siamese model
    :param logo_feat_list: targetlist embeddings
    :param file_name_list: targetlist paths
    :param shot_path: path to image
    :param url: url
    :param ts: siamese threshold
    :return pred_target
    '''
    # targetlist domain list
    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)
        
    # look at boxes for logo class only
    logo_boxes = pred_boxes[pred_classes==1] 
    
    # run logo matcher
    pred_target = None
    if len(logo_boxes) > 0:
        # siamese prediction for logo box
        for i, coord in enumerate(logo_boxes):
            min_x, min_y, max_x, max_y = coord
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            target_this, domain_this = siamese_inference(model, domain_map, 
                                                         logo_feat_list, file_name_list,
                                                         shot_path, bbox, t_s=ts, grayscale=False)
            
            # domain matcher to avoid FP
            if not target_this is None and tldextract.extract(url).domain not in domain_this: 
                pred_target = target_this 
                break # break if target is matched
    
    return brand_converter(pred_target)
        