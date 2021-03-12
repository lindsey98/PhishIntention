import sys
sys.path.append("..") 

from configs import get_cfg
from modelling import *
from datasets import WebMapper
import json
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, default_setup
import detectron2.data.transforms as T
from detectron2.structures.image_list import ImageList

import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import funcy

import cv2
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def setup(args):
    """
    Create AL configs and perform basic setups.
    """

    # Initialize the configurations
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Ensure it uses appropriate names and architecture  
    cfg.MODEL.ROI_HEADS.NAME = 'ROIHeadsAL'
    cfg.MODEL.META_ARCHITECTURE = 'ActiveLearningRCNN'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 # lower this threshold to report more boxes

    cfg.freeze()
    default_setup(cfg, args)
    return cfg



class Inference:
    
    def __init__(self, cfg, data_path, write_path):

        # Initialize model
        self.cfg = cfg
        self.model = build_model(cfg)
        self.model.eval()

        # Build_model will not load weights, have to load weights explicitly
        checkpointer = DetectionCheckpointer(self.model)  
        checkpointer.load(cfg.MODEL.WEIGHTS)
        # Augmentation at test time
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        # Test image folder
        self.data_path = data_path

        self.write_path = write_path
        
    def _feature_embed(self, features):
        '''Obtain feature embeddings from backbone'''
        # Avgpool over channels
        features_heatmaps = [torch.mean(features[k], dim=1) for k in features.keys()]
        
        # Adaptive pooling 2D --> Tensor(shape 5x8x8)]
        pooled_heatmaps = torch.cat([F.adaptive_avg_pool2d(f_map[None, ...], output_size=(8, 8)) \
                                     .view(1, 8, 8).detach().cpu() for f_map in features_heatmaps], dim=0)
        pooled_heatmaps = pooled_heatmaps.view(-1).numpy()
        
        return pooled_heatmaps
    
    def _create_instance_dicts(self, outputs, file_name, image_id):
        '''Create dict from output'''
        instance_dicts = []

        instances = outputs["instances"]
        pred_classes = instances.pred_classes.cpu().numpy()
        pred_boxes = instances.pred_boxes
        scores = instances.scores.cpu().numpy()
        scores_al = instances.scores_al.cpu().numpy() if 'scores_al' in instances.get_fields().keys() else None
        
        # For each bounding box
        for i, box in enumerate(pred_boxes):
            box = box.cpu().numpy()
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            width = x2 - x1
            height = y2 - y1

            # HACK Only specific for this dataset
            category_id = int(pred_classes[i] + 1)
            # category_id = self.contiguous_id_to_thing_id[pred_classes[i]]
            score = float(scores[i])
            score_al = float(scores_al[i]) if (scores_al is not None) else None
            
            i_dict = {
                "image_id": image_id,
                "file_name": file_name,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "score": score,
                "score_al": score_al,
            }

            instance_dicts.append(i_dict)

        return instance_dicts
    
    def _save_inference_dicts(self, **kwargs):
        raise Exception ("Not Implemented")
        
    def run(self, **kwargs):
        raise Exception ("Not Implemented")        
        
        
class ALInference(Inference):
    '''Save inference dict for any dataset'''

    def _save_inference_dicts(self, feature_refs):

        # Save predictions in coco format
        coco_instances_results = []
        
        for i, file in tqdm(enumerate(os.listdir(self.data_path))):
            
            original_image = cv2.imread(os.path.join(self.data_path, file))
            if self.cfg.INPUT.FORMAT == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}

            
            with torch.no_grad():
                # Predict 
                outputs = self.model.forward_al([inputs], feature_refs)
                
                # Convert to coco predictions format
                instance_dicts = self._create_instance_dicts(outputs[0], file_name=file, image_id=i)
                coco_instances_results.extend(instance_dicts)
                               
            if i % 100 == 0:
                # Write intermediate results
                with open(self.write_path, "w") as f:
                    json.dump(coco_instances_results, f)

        with open(self.write_path, "w") as f:
            json.dump(coco_instances_results, f)   

    def run(self, feature_refs=None):
        self._save_inference_dicts(feature_refs)

class ALInference_emb(Inference):
    '''Save feature embeddings for any dataset'''

    def _save_inference_dicts(self):

        # Save predictions in coco format
        coco_instances_results = []
        pooled_heatmaps = []
        
        for i, file in tqdm(enumerate(os.listdir(self.data_path))):
            
            original_image = cv2.imread(os.path.join(self.data_path, file))
            height, width = original_image.shape[:2]
            if self.cfg.INPUT.FORMAT == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]

            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]
            images = self.model.preprocess_image(inputs)

            with torch.no_grad():
                # Get features, Get heatmap of shape (320,)
                features = self.model.backbone(images.tensor)
                pooled_heatmap = self._feature_embed(features)
                pooled_heatmaps.append({'file_name': file,
                                        'feature': pooled_heatmap})
                               
            if i % 100 == 0:
                # Write intermediate results
#                 print(pooled_heatmaps[0]['feature'].shape)
                with open(self.write_path, 'wb') as handle:
                    pickle.dump(pooled_heatmaps, handle)

        with open(self.write_path, 'wb') as handle:
            pickle.dump(pooled_heatmaps, handle)

    def run(self):
        self._save_inference_dicts()


if __name__ == '__main__':
    
    parser = default_argument_parser()
    parser.add_argument("--write_path", required=True, help="Where to save inference pkl/json")
    parser.add_argument("--data_path", required=True, help="Test image folder")
    
    args = parser.parse_args()
    
    print("Command Line Args:", args)
    cfg = setup(args)

#     AL_inference = ALInference(cfg, args.data_path, args.write_path)
    AL_inference = ALInference_emb(cfg, args.data_path, args.write_path)
    AL_inference.run()
    
    
#     python AL_inference.py --write_path ../../output/entropy_al.json --data_path ../../datasets/AL_pool_imgs/ --config-file ../../configs/faster_rcnn_web_lr0.001_entropy.yaml MODEL.WEIGHTS ../../output/website_lr0.001/model_final.pth