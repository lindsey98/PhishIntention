import torch
from typing import List

def dice_coefficient(area1, area2, inter_area):
    return 2*inter_area / (area1 + area2)

def iou(area1, area2, inter_area):
    return inter_area / (area1 + area2 - inter_area)

def overlap_coefficient(area1, area2, inter_area):
    return inter_area / torch.min(area1, area2)

def _elementwise_scoring_template(boxes1, boxes2, formula):
    
    area1, area2 = boxes1.area(), boxes2.area()
    inter_area = elementwise_intersect_area(boxes1, boxes2)

    scores = torch.where(
            inter_area > 0,
            formula(area1, area2, inter_area),
            torch.zeros(1, dtype=inter_area.dtype, device=inter_area.device),
        )
    return scores

def _pairwise_scoring_template(boxes1, boxes2, formula):
    
    area1, area2 = boxes1.area(), boxes2.area()
    inter_area = pairwise_intersect_area(boxes1, boxes2)

    scores = torch.where(
            inter_area > 0,
            formula(area1[:, None], area2, inter_area),
            torch.zeros(1, dtype=inter_area.dtype, device=inter_area.device),
        )
    return scores

def elementwise_intersect_area(boxes1, boxes2):

    # Modified based on 
    # https://detectron2.readthedocs.io/_modules/detectron2/structures/boxes.html#pairwise_iou
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    width_height = \
        torch.min(boxes1[:, 2:], boxes2[:, 2:]) - \
        torch.max(boxes1[:, :2], boxes2[:, :2])  
        # [N,M,2]

    width_height.clamp_(min=0)  # [N,2]
    inter = width_height.prod(dim=-1)  # [N]
    del width_height
    return inter

def pairwise_intersect_area(boxes1, boxes2):

    # Modified based on 
    # https://detectron2.readthedocs.io/_modules/detectron2/structures/boxes.html#pairwise_iou
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    width_height = \
        torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - \
        torch.max(boxes1[:, None, :2], boxes2[:, :2])  
        # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height
    return inter

def elementwise_iou(boxes1, boxes2):

    return _elementwise_scoring_template(boxes1, boxes2, iou)

def elementwise_dice_coefficient(boxes1, boxes2):

    return _elementwise_scoring_template(boxes1, boxes2, dice_coefficient)

def elementwise_overlap_coefficient(boxes1, boxes2):

    return _elementwise_scoring_template(boxes1, boxes2, overlap_coefficient)

def pairwise_iou(boxes1, boxes2):

    return _pairwise_scoring_template(boxes1, boxes2, iou)

def pairwise_dice_coefficient(boxes1, boxes2):

    return _pairwise_scoring_template(boxes1, boxes2, dice_coefficient)

def pairwise_overlap_coefficient(boxes1, boxes2):

    return _pairwise_scoring_template(boxes1, boxes2, overlap_coefficient)




# def select_top(gt, overlappings):
#     max_idx = np.argmax(overlappings)
#     if overlappings[max_idx]<=0.0: 
#         return []
#     else:
#         return [gt[max_idx]]

# def select_nonzero(gt, overlappings):
#     return [box for (box, overlap) in zip(gt, overlappings) if overlap>0.0]



def select_top(overlapping_scores) -> List[List[int]]: 
    """
    Obtain the gt box ids of the top overlapping score 
    for each prediction box. Drop the gt box the if all 
    the overlapping scores are smaller than zero for a
    prediction box.
    """

    max_scores = overlapping_scores.max(dim=-1)
    return [[idx.item()] if val>0 else [] \
            for idx, val in zip(max_scores.indices, max_scores.values)]

def select_above(overlapping_scores, threshold=0) -> List[List[int]]:
    """
    Obtain the gt box ids of the overlapping score more than 
    some threshold for each prediction box.
    """

    return_indices = [[] for _ in range(overlapping_scores.shape[0])]
    for pred_idx, gt_idx in zip(*torch.where(overlapping_scores>threshold)):
        return_indices[pred_idx].append(gt_idx.item())
    
    return return_indices

def select_nonzero(overlapping_scores) -> List[List[int]]:
    """
    Obtain the gt box ids of the overlapping score more than 
    0 for each prediction box.
    """

    return select_above(overlapping_scores, 0)