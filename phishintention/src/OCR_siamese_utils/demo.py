from __future__ import absolute_import
import sys
# sys.path.append('./')

import argparse
import os
import os.path as osp
import numpy as np
import math
import time
from PIL import Image, ImageFile

import torch
from torch.backends import cudnn
from torchvision import transforms

from .lib.models.model_builder import ModelBuilder
from .lib.utils.labelmaps import get_vocabulary, labels2strs


def image_process(image_path, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    
    img = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path.convert('RGB')

    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img

class DataInfo(object):
    """
    Save the info about the dataset.
    This a code snippet from dataset.py
    """
    def __init__(self, voc_type):
        super(DataInfo, self).__init__()
        self.voc_type = voc_type

        assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.rec_num_classes = len(self.voc)


def ocr_model_config(checkpoint, height=None, width=None):
    
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('using cuda.')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
  
    # Create data loaders
    if height is None or width is None:
        height, width = (32, 100)

    dataset_info = DataInfo('ALLCASES_SYMBOLS')

    # Create model
    model = ModelBuilder(arch='ResNet_ASTER', rec_num_classes=dataset_info.rec_num_classes,
                         sDim=512, attDim=512, max_len_labels=100,
                         eos=dataset_info.char2id[dataset_info.EOS], STN_ON=True)

    # Load from checkpoint
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if device == 'cuda':
        model = model.to(device)
        
    return model

def ocr_main(image_path, model, height=None, width=None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Evaluation
    model.eval()
    
    img = image_process(image_path)
    with torch.no_grad():
        img = img.to(device)
    input_dict = {}
    input_dict['images'] = img.unsqueeze(0)
    
    # TODO: testing should be more clean. to be compatible with the lmdb-based testing, need to construct some meaningless variables.
    dataset_info = DataInfo('ALLCASES_SYMBOLS')
    rec_targets = torch.IntTensor(1, 100).fill_(1)
    rec_targets[:,100-1] = dataset_info.char2id[dataset_info.EOS]
    input_dict['rec_targets'] = rec_targets.to(device)
    input_dict['rec_lengths'] = [100]
    
    with torch.no_grad():
        features, decoder_feat = model.features(input_dict)
    features = features.detach().cpu()
    decoder_feat = decoder_feat.detach().cpu()
    features = torch.mean(features, dim=1)
    
    return features

