from typing import List, Set, Dict, Tuple, Optional
import os
import json
import numpy

def construct_dict(
        filename: str,
        height: int,
        weight: int,
        image_id: int,
        gt_boxes: numpy.ndarray,
        gt_classes: numpy.ndarray,
        json_file: str
    ) -> None:
    
        """Construct/Append coco-style json annotation

        Parameters
        ----------
        filename : str
            path to instance
        height: int
        weight： int
        image_id: int
            unique identifier to instance
        gt_boxes: numpy.ndarray
            array of groundtruth boxes, each is of (min_x, min_y, max_x, max_y)
        gt_classes: numpy.ndarray
            groundtruth classes for groundtruth boxes
        json_file: str
            path to json

        Returns
        -------
            None
        """
        if os.path.exists(json_file):
            with open(json_file, 'r') as handle:
                json_dict = json.load(handle)
        else:
            json_dict = {"images": [],  "annotations": [], "categories": [{"id": 1, "name": "box"}, {"id": 2, "name": "logo"}]}
        
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        
        ## get gt_box annotations
        for i, b in enumerate(gt_boxes):
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            width = x2 - x1
            height = y2 - y1
            
            category_id = gt_classes[i]
            id_annot = json_dict['annotations'][-1]['id']+1 if len(json_dict["annotations"])!=0 else 0
                
            ann = {
                "area": width * height,
                "image_id": image_id,
                "bbox": [x1, y1, width, height],
                "category_id": category_id,
                "id": id_annot, # id for box, need to be continuous
                "iscrowd": 0
                }
            json_dict["annotations"].append(ann)
            
        ## write to json file
        with open(json_file, "w") as f:
            json.dump(json_dict, f)