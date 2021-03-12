from detectron2.config.defaults import _C
from detectron2.config.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Additional Configuration for Active Learning
# ---------------------------------------------------------------------------- #

_C.AL = CN()
_C.AL.MODE = 'object' # {'image', 'object'}
# Perform active learning on whether image-level or object-level 
_C.AL.OBJECT_SCORING = '1vs2' # {'1vs2, 'least_confidence', 'random', 'perturbation'}
# The method to compute the individual object scores  
_C.AL.IMAGE_SCORE_AGGREGATION = 'avg' # {'avg', 'max', 'sum'}
# The method to aggregate the individual object scores to the whole image score 

_C.AL.PERTURBATION = CN()
# Configurations for perturbation scoring method
_C.AL.PERTURBATION.VERSION = 1 # 1 to 5 
# 1 - CE SCORING 
# 2 - KL SCORING 
# 3 - IOU SCORING 
# 4 - IOU + CE 
# 5 - IOU + KL * 3
_C.AL.PERTURBATION.ALPHAS = [0.08, 0.12] 
# Horizontal translation ratio
_C.AL.PERTURBATION.BETAS = [0.04, 0.16] 
# Vertical translation ratio
_C.AL.PERTURBATION.RANDOM = False 
# whether generate random perturbation at different iterations
_C.AL.PERTURBATION.LAMBDA = 1.
# the weighting constant 

_C.AL.DATASET = CN()
# Specifies the configs for creating new datasets 
# It will also combines configs from DATASETS and DATALOADER 
# when creating the DynamicDataset for training.
_C.AL.DATASET.NAME = ''
_C.AL.DATASET.IMG_ROOT = ''
_C.AL.DATASET.ANNO_PATH = ''
# Extra meta information for the dataset, only supports COCO Dataset
# This is somehow ugly, and shall be updated in the future.
_C.AL.DATASET.CACHE_DIR = 'al_datasets'
# The created dataset will be saved in during the active learning 
# training process 
_C.AL.DATASET.NAME_PREFIX = 'r'
# The created dataset will be named as '{NAME}-{NAME_PREFIX}{idx}' in the 
# Detectron2 system
_C.AL.DATASET.BUDGET_STYLE = 'object' # {'image', 'object'}
# Depericated. Now the budget style is the same as AL.MODE
_C.AL.DATASET.IMAGE_BUDGET = 20
_C.AL.DATASET.OBJECT_BUDGET = 2000
# Specifies the way to calculate the budget 
# If specify the BUDGET_STYLE as image, while using the object-level
# Active Learning, we will convert the image_budget to object budget 
# by OBJECT_BUDGET = IMAGE_BUDGET * AVG_OBJ_IN_TRAINING. 
# Similarly, we have 
# IMAGE_BUDGET = OBJECT_BUDGET // AVG_OBJ_IN_TRAINING.
_C.AL.DATASET.BUDGET_ALLOCATION = 'linear'
_C.AL.DATASET.SAMPLE_METHOD = 'top' # {'top', 'kmeans'}
# The method to sample images when labeling 

_C.AL.OBJECT_FUSION = CN()
# Specifies the configs to fuse model prediction and ground-truth (gt)
_C.AL.OBJECT_FUSION.OVERLAPPING_METRIC = 'iou' # {'iou', 'dice_coefficient', 'overlap_coefficient'}
# The function to calculate the overlapping between model pred and gt
_C.AL.OBJECT_FUSION.OVERLAPPING_TH = 0.25 # Optional
# The threshold for selecting the boxes 
_C.AL.OBJECT_FUSION.SELECTION_METHOD = 'top' # {'top', 'above', 'nonzero'}
# For gt boxes with non-zero overlapping with the pred box, specify the 
# way to choose the gt boxes. 
# top: choose the one with the highest overlapping
# above: choose the ones has the overlapping above the threshold specified above
# nonzero: choose the gt boxes with non-zero overlapping
_C.AL.OBJECT_FUSION.REMOVE_DUPLICATES = True
_C.AL.OBJECT_FUSION.REMOVE_DUPLICATES_TH = 0.15
# For the fused dataset, remove duplicated boxes with overlapping more than 
# the given threshold
_C.AL.OBJECT_FUSION.RECOVER_MISSING_OBJECTS = True
# If true, we recover the mis-identified objects during the process
_C.AL.OBJECT_FUSION.INITIAL_RATIO = 0.85
_C.AL.OBJECT_FUSION.LAST_RATIO = 0.25
_C.AL.OBJECT_FUSION.DECAY = 'linear'
# During the object fusion process, we decay the number of objects selected for inspection
# as training goes on. 
_C.AL.OBJECT_FUSION.PRESELECTION_RAIO = 1.5
_C.AL.OBJECT_FUSION.ENDSELECTION_RAIO = 1.25
_C.AL.OBJECT_FUSION.SELECTION_RAIO_DECAY = 'linear'
# During the object fusion process, we take the top x number of objects out of predictions. 
# The x is calculate as x = avg_object_per_image * SELECTION_RAIO
# The purpose is not to bring too much useless model predictions in the fusion procedure. 
_C.AL.OBJECT_FUSION.RECOVER_ALMOST_CORRECT_PRED = True
_C.AL.OBJECT_FUSION.BUDGET_ETA = 0.2


_C.AL.TRAINING = CN()
_C.AL.TRAINING.ROUNDS = 5 
# The number of rounds for performing AL dataset update
_C.AL.TRAINING.EPOCHS_PER_ROUND_INITIAL = 500
# The numbers of epochs for training during each round. 
# As Detectron2 does not support epochs natively, we will use the 
# following formula to convert the epochs to iterations after creating 
# the new dataset:
# iterations = total_imgs / batch_size * epochs_per_round
_C.AL.TRAINING.EPOCHS_PER_ROUND_DECAY = 'linear'
_C.AL.TRAINING.EPOCHS_PER_ROUND_LAST = 50

################Add output dir to log traning loss #############
_C.OUTPUT_DIR_LOSS = _C.OUTPUT_DIR + '/training_loss.json'
#############################################################

##############################################################
### Provide config support for newer version of Detectron2 ###
### ------------------------------------------------------ ###
### Note: this is only for compatibility when loading      ### 
### model configs and weights, and the actual config       ###
### won't be used                                          ### 
##############################################################

_C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0
_C.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
_C.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
_C.SOLVER.NESTEROV = False
_C.SOLVER.REFERENCE_WORLD_SIZE = 0

################ Add checkpoint period for loss #############
_C.SOLVER.CHECKPOINT_PERIOD_LOSS = 1000
###############################################################