
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2_1.datasets import *
from detectron2.data import build_detection_test_loader
from tqdm import tqdm
import json


if __name__ == '__main__':
    
    # Modify config
    cfg = get_cfg()
    cfg.merge_from_file('configs/faster_rcnn_login_lr0.001.yaml')
    cfg = cfg.clone()  # cfg can be modified by model


    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    input_format = cfg.INPUT.FORMAT
    
    # Init dataloader on test dataset
    dataset_mapper = LoginMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(
        cfg, cfg.DATASETS.TEST[0], mapper=dataset_mapper
    )
    
    category_dict = [{"id": 1, "name":"login"}]

    datadict = {"images":[], "annotations":[], "categories": category_dict}
    

    for i, batch in tqdm(enumerate(data_loader)):
        ## Save gt box after transformation
        instances = batch[0]["instances"]
        gt_boxes = instances.gt_boxes
        gt_classes = instances.gt_classes

        imgpath = batch[0]["file_name"].split('/')[-1]
        img_height, img_width = batch[0]["height"], batch[0]["width"]
        image_id = batch[0]["image_id"]

        image = {
            "file_name": imgpath,
            "height": img_height,
            "width": img_width,
            "id": image_id,
        }

        datadict["images"].append(image)

        for k, box in enumerate(gt_boxes.tensor.numpy()):
            x1, y1, x2, y2 = list(map(int, box))
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            # find corresponding category id
            category_id = int(gt_classes.numpy()[k]) + 1
            id_annot = len(datadict["annotations"]) + 1 #id field must start with 1

            ann = {
                "area": width * height,
                "image_id": image_id,
                "bbox": [x1, y1, width, height],
                "category_id": category_id,
                "id": id_annot, # id for box, need to be continuous
                "iscrowd": 0
                }

            datadict["annotations"].append(ann)
            
        if i % 100 == 0:
            with open('../../../datasets/login_finder_dataset/val_coco2_transform.json', 'wt', encoding='UTF-8') as f:
                json.dump(datadict, f)

    with open('../../../datasets/login_finder_dataset/val_coco2_transform.json', 'wt', encoding='UTF-8') as f:
        json.dump(datadict, f)