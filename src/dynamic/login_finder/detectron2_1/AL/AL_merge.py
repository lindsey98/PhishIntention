
import funcy
import random
import json
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..") 

from configs import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.data import MetadataCatalog, DatasetCatalog



def setup(args):
    """
    Create configs and perform basic setups.
    """

    # Initialize the configurations
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Ensure it uses appropriate names and architecture  
    cfg.MODEL.ROI_HEADS.NAME = 'ROIHeadsAL'
    cfg.MODEL.META_ARCHITECTURE = 'ActiveLearningRCNN'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # lower this threshold to get more boxes

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

class MergeDataset:
    
    def __init__(self, cfg, orig_dataset_json, 
                 al_dataset_json, al_pred_dict, 
                 merge_dataset_path):
        
        self.cfg = cfg
        # Read training data gt json file: {"images":[], "annotations":[]}
        with open(orig_dataset_json, 'rt', encoding='UTF-8') as f:
            self.orig_data_dict = json.load(f)
            
        # Read AL data gt json file: {"images":[], "annotations":[]}
        with open(al_dataset_json, 'rt', encoding='UTF-8') as f:
            self.al_data_dict = json.load(f)           
            
        # Read AL data prediction file: [{"image_id":, "bbox":, "score":, "category_id":, "score_al":}, 
        #                                {"image_id":, "bbox":, "score":, "category_id":, "score_al":}]
        with open(al_pred_dict, 'rt', encoding='UTF-8') as f:
            self.al_pred_dict = json.load(f)
            
        # Merge dataset json path
        self.merge_dataset_path = merge_dataset_path
        
    def _merge_datadict(self, **kwargs):
        raise Exception ("Not Implemented")
            
    def run(self, **kwargs):
        raise Exception ("Not Implemented")
        
        
class SelectALDataset(MergeDataset):
    '''Merge original datadict with selected AL datadict'''
    def __init__(self, cfg, orig_dataset_json, 
                 al_dataset_json, al_pred_dict, 
                 merge_dataset_path):
        
        # Inherit all 
        super(SelectALDataset, self).__init__(cfg, orig_dataset_json, 
                 al_dataset_json, al_pred_dict, 
                 merge_dataset_path)
    
    def _aggregate_score(self):
        '''Aggregate proposals scores for single image'''
        count = 0
        all_image_ids = set([x['image_id'] for x in self.al_pred_dict])

        for image_id in all_image_ids:
        
            values = funcy.lfilter(lambda a: a['image_id'] == image_id, self.al_pred_dict)
            if self.cfg.AL.IMAGE_SCORE_AGGREGATION == 'avg':
                agg_score = np.mean(funcy.lmap(lambda a: a['score_al'], values))
            elif self.cfg.AL.IMAGE_SCORE_AGGREGATION == 'max':
                agg_score = np.max(funcy.lmap(lambda a: a['score_al'], values))
            elif self.cfg.AL.IMAGE_SCORE_AGGREGATION == 'sum':
                agg_score = np.sum(funcy.lmap(lambda a: a['score_al'], values))
            elif self.cfg.AL.IMAGE_SCORE_AGGREGATION == 'random':
                agg_score = random.sample(funcy.lmap(lambda a: a['score_al'], values), 1)[0]

            for r in values:
                r['image_score'] = agg_score
                
            count += 1
            if count % 1000 == 0:
                print(count)
        

    def _select_topn(self, n):
        '''Get topn images with topn uncertainty score'''
        
        print('Start score aggregation') 
        if self.cfg.AL.OBJECT_SCORING == 'feature_emb': # for feature embedding method is already aggregted
            pass
        else:
            self._aggregate_score()
        print('Score aggregation finished')
        
        if self.cfg.AL.OBJECT_SCORING == 'feature_emb':
            image_scores = [fs['score_al'] for fs in self.al_pred_dict]
        else:
            image_scores = [fs['image_score'] for fs in self.al_pred_dict]
            
        image_ids = [fs['image_id'] for fs in self.al_pred_dict]
        
        # Remove duplicates: we only want to look at image scores 
        image_scores = list(dict.fromkeys(image_scores))
        image_ids = list(dict.fromkeys(image_ids))
        
        assert len(image_scores) == len(image_ids)
        # Select TopN uncertain image ids
        print('Start finding topN image ids')
        sorted_image_scores = np.argsort(image_scores)[::-1] 
        print(sorted_image_scores)
        selected_image_ids = np.asarray(image_ids)[sorted_image_scores[:n]]
        
        al_select_images, al_select_annotations = self._select_dict_byid(selected_image_ids)
        return al_select_images, al_select_annotations
    
    
    def _select_dict_byid(self, selected_image_ids):
        '''Select dictionary by ID'''
        
        print('Number of AL instances before filtering:', len(self.al_data_dict["images"]))
        al_select_images = funcy.lfilter(lambda a: a['id'] in selected_image_ids.tolist(), 
                                       self.al_data_dict["images"])
        
        al_select_annotations = funcy.lfilter(lambda a: a['image_id'] in selected_image_ids.tolist(), 
                                       self.al_data_dict["annotations"])     
        
        print('Number of AL instances after filtering:', len(al_select_images))
        print('Number of AL bboxes after filtering:', len(al_select_annotations))

        return al_select_images, al_select_annotations

        
    def _merge_datadict(self):
        '''Merging two datadict'''
        
        datadict_orig = self.orig_data_dict
        al_select_images, al_select_annotations = self._select_topn(n=len(self.al_data_dict["images"])//4) # Select 1/4 of AL data
        
        # Directly merge because there is no image id conflict
        merged_dict = datadict_orig
        merged_dict['images'].extend(al_select_images)
        merged_dict['annotations'].extend(al_select_annotations)

        return merged_dict
    
    def run(self):
        merged_dict = self._merge_datadict()
        with open(self.merge_dataset_path, 'wt', encoding='UTF-8') as f:
            json.dump(merged_dict, f)
            
        return merged_dict
    
    
    
    
class PseudoALDataset(MergeDataset):
    '''Merge training dataset with pseudo labelled AL dataset'''
    def __init__(self, cfg, orig_dataset_json, 
                 al_dataset_json, al_pred_dict, 
                 merge_dataset_path):
        
        super(PseudoALDataset, self).__init__(cfg, orig_dataset_json, 
                 al_dataset_json, al_pred_dict, 
                 merge_dataset_path)


    def _generate_pseudo_dict(self):
        coco_images = []
        coco_annotations = []

        for image_dict in tqdm(self.al_pred_dict):

            instance_gt = funcy.lfilter(lambda a: a["id"] == image_dict["image_id"], 
                                        self.al_data_dict["images"])[0]
            
            instance_width = instance_gt['width']
            instance_height = instance_gt['height']
            instance_file_name = instance_gt['file_name']

            if image_dict["image_id"] not in [x["id"] for x in coco_images]:
                coco_image = {
                    "id": image_dict["image_id"],
                    "width": instance_width,
                    "height": instance_height,
                    "file_name": instance_file_name,
                }
                coco_images.append(coco_image)

            x1, y1, width, height = image_dict['bbox']
            category_id = image_dict['category_id']
            ann = {
                "area": width * height,
                "image_id": image_dict["image_id"],
                "bbox": [x1, y1, width, height],
                "category_id": category_id,
                "id": len(coco_annotations) + 1, # id for box, need to be continuous
                "iscrowd": 0
                }

            coco_annotations.append(ann)
        return coco_images, coco_annotations

    def _annot_reorder(self, merged_dict):
        '''Reorder annotations'''
        ct = 0
        for annot in merged_dict["annotations"]:
            annot['id'] = ct + 1
            ct += 1
        
    def _merge_datadict(self):
        '''Reindex datadict when merging two datadict'''
        datadict_orig = self.orig_data_dict
        coco_images, coco_annotations = self._generate_pseudo_dict() 
        
        # Directly merge because there is no image id conflict
        merged_dict = datadict_orig
        merged_dict['images'].extend(coco_images)
        merged_dict['annotations'].extend(coco_annotations)
        
        merged_dict = self._annot_reorder(merged_dict)
        
                
        metadata = MetadataCatalog.get('coco_2017_train')

        # unmap the category mapping ids for COCO
        if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
            reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
        else:
            reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

        categories = [
            {"id": reverse_id_mapper(id), "name": name}
            for id, name in enumerate(metadata.thing_classes)
        ]
        merged_dict['categories'] = categories
        
        return merged_dict

    
    
    def run(self):
        merged_dict = self._merge_datadict()
        with open(self.merge_dataset_path, 'wt', encoding='UTF-8') as f:
            json.dump(merged_dict, f)
            
        return merged_dict
    
    
    
if __name__ == '__main__':
    parser = default_argument_parser()

    # Extra Configurations for dataset names and paths
    parser.add_argument("--json_annotation_train", required=True, help="The path to the training set JSON annotation")
    
    parser.add_argument("--json_annotation_gt_AL", required=True, help="The path to the AL set gt JSON annotation")
    parser.add_argument("--json_annotation_pred_AL", required=True, help="The path to the AL set prediction JSON annotation")
    
    parser.add_argument("--json_annotation_merge", required=True, help="The path to the merged set JSON annotation")
    parser.add_argument("--select", action="store_true", help="Select AL/Generate Pseudo-labelled AL")
    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    
    if args.select:
        
        al_dataset = SelectALDataset(cfg, 
                                     orig_dataset_json=args.json_annotation_train,
                                     al_dataset_json=args.json_annotation_gt_AL,
                                     al_pred_dict=args.json_annotation_pred_AL,
                                     merge_dataset_path=args.json_annotation_merge)
        
    else:
    
        al_dataset = PseudoALDataset(cfg, 
                                     orig_dataset_json=args.json_annotation_train,
                                     al_dataset_json=args.json_annotation_gt_AL,
                                     al_pred_dict=args.json_annotation_pred_AL,
                                     merge_dataset_path=args.json_annotation_merge)
        
        
    al_dataset.run()
    



    