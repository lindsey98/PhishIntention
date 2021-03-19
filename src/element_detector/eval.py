from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(gt_coco_path, results_coco_path):

    coco_gt = COCO(gt_coco_path)
    coco_dt = coco_gt.loadRes(results_coco_path)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    
if __name__ == '__main__':
    gt_coco_path = '../../datasets/val_coco.json' # here the ground-truth is for the original-sized image
    results_coco_path =  'output/website_lr0.001/coco_instances_results.json' # here the prediction is for the resized image
    
    evaluate(gt_coco_path, results_coco_path)

