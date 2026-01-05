from PIL import Image
import numpy as np
from collections import OrderedDict
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from ocr_lib.models.attention_recognition_head import AttentionRecognitionHead
from ocr_lib.loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss
from ocr_lib.models.tps_spatial_transformer import TPSSpatialTransformer
from ocr_lib.models.stn_head import STNHead

from ocr_lib.models.resnet_aster import ResNet_ASTER

__factory = {
  'ResNet_ASTER': ResNet_ASTER,
}


def names():
  return sorted(__factory.keys())


def create(name, *args, **kwargs):
  """Create a model instance.

  Parameters
  ----------
  name: str
    Model name. One of __factory
  pretrained: bool, optional
    If True, will use ImageNet pretrained model. Default: True
  num_classes: int, optional
    If positive, will change the original classifier the fit the new classifier with num_classes. Default: True
  with_words: bool, optional
    If True, the input of this model is the combination of image and word. Default: False
  """
  if name not in __factory:
    raise KeyError('Unknown model:', name)
  return __factory[name](*args, **kwargs)

# from config import get_args
# global_args = get_args(sys.argv[1:])


class ModelBuilder(nn.Module):
  """
  This is the integrated model.
  """
  def __init__(self, arch, rec_num_classes, sDim, attDim, max_len_labels, eos, STN_ON=False):
    super(ModelBuilder, self).__init__()

    self.arch = arch
    self.rec_num_classes = rec_num_classes
    self.sDim = sDim
    self.attDim = attDim
    self.max_len_labels = max_len_labels
    self.eos = eos
    self.STN_ON = STN_ON
    self.tps_inputsize = [32, 64]

    self.encoder = create(self.arch,
                      with_lstm=True,
                      n_group=1)
    encoder_out_planes = self.encoder.out_planes

    self.decoder = AttentionRecognitionHead(
                      num_classes=rec_num_classes,
                      in_planes=encoder_out_planes,
                      sDim=sDim,
                      attDim=attDim,
                      max_len_labels=max_len_labels)
    self.rec_crit = SequenceCrossEntropyLoss()

    if self.STN_ON:
      self.tps = TPSSpatialTransformer(
        output_image_size=tuple([32, 100]),
        num_control_points=20,
        margins=tuple([0.05,0.05]))
    
      self.stn_head = STNHead(
        in_planes=3,
        num_ctrlpoints=20,
        activation='none')

  def forward(self, input_dict):
    return_dict = {}
    return_dict['losses'] = {}
    return_dict['output'] = {}

    x, rec_targets, rec_lengths = input_dict['images'], \
                                  input_dict['rec_targets'], \
                                  input_dict['rec_lengths']

    # rectification
    if self.STN_ON:
      # input images are downsampled before being fed into stn_head.
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      stn_img_feat, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)
      if not self.training:
        # save for visualization
        return_dict['output']['ctrl_points'] = ctrl_points
        return_dict['output']['rectified_images'] = x

    encoder_feats = self.encoder(x)
    encoder_feats = encoder_feats.contiguous()

    if self.training:
      rec_pred = self.decoder([encoder_feats, rec_targets, rec_lengths])
      loss_rec = self.rec_crit(rec_pred, rec_targets, rec_lengths)
      return_dict['losses']['loss_rec'] = loss_rec
    else:
      rec_pred, rec_pred_scores = self.decoder.beam_search(encoder_feats, 5, self.eos)
      rec_pred_ = self.decoder([encoder_feats, rec_targets, rec_lengths])
      loss_rec = self.rec_crit(rec_pred_, rec_targets, rec_lengths)
      return_dict['losses']['loss_rec'] = loss_rec
      return_dict['output']['pred_rec'] = rec_pred
      return_dict['output']['pred_rec_score'] = rec_pred_scores

    # pytorch0.4 bug on gathering scalar(0-dim) tensors
    for k, v in return_dict['losses'].items():
      return_dict['losses'][k] = v.unsqueeze(0)

    return return_dict

  def features(self, input_dict):

    x, rec_targets, rec_lengths = input_dict['images'], \
                                  input_dict['rec_targets'], \
                                  input_dict['rec_lengths']
    
    # rectification
    if self.STN_ON:
      # input images are downsampled before being fed into stn_head.
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      stn_img_feat, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)

    encoder_feats = self.encoder(x)
    encoder_feats = encoder_feats.contiguous()
    
    rec_pred_ = self.decoder([encoder_feats, rec_targets, rec_lengths])

    return encoder_feats, rec_pred_
        