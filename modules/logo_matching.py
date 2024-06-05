from PIL import Image, ImageOps
from torchvision import transforms
from utils.utils import brand_converter, resolution_alignment, l2_norm
from modules.models2 import KNOWN_MODELS
from ocr_lib.models.model_builder import ModelBuilder
from ocr_lib.utils.labelmaps import get_vocabulary
import torch
from torch.backends import cudnn
import os
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from tldextract import tldextract
import pickle

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

def ocr_model_config(weights_path, height=None, width=None):
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
    weights_path = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights_path['state_dict'])

    if device == 'cuda':
        model = model.to(device)

    return model

def siamese_model_config(num_classes: int, weights_path: str):
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)

    # Load weights
    weights = torch.load(weights_path, map_location='cpu')
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        if k.startswith('module'):
            name = k.split('module.')[1]
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model


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


def ocr_main(image_path, model, height=None, width=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Evaluation
    model.eval()

    img = image_process(image_path)
    with torch.no_grad():
        img = img.to(device)
    input_dict = {}
    input_dict['images'] = img.unsqueeze(0)

    dataset_info = DataInfo('ALLCASES_SYMBOLS')
    rec_targets = torch.IntTensor(1, 100).fill_(1)
    rec_targets[:, 100 - 1] = dataset_info.char2id[dataset_info.EOS]
    input_dict['rec_targets'] = rec_targets.to(device)
    input_dict['rec_lengths'] = [100]

    with torch.no_grad():
        features, decoder_feat = model.features(input_dict)
    features = features.detach().cpu()
    decoder_feat = decoder_feat.detach().cpu()
    features = torch.mean(features, dim=1)

    return features

@torch.no_grad()
def get_ocr_aided_siamese_embedding(img, model, ocr_model, grayscale=False):
    '''
    Inference for a single image
    :param img: image path in str or image in PIL.Image
    :param model: Siamese model to make inference
    :param ocr_model: OCR model
    :param imshow: enable display of image or not
    :param title: title of displayed image
    :param grayscale: convert image to grayscale or not
    :return feature embedding of shape (2048,)
    '''
    img_size = 224
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std),
     ])

    img = Image.open(img) if isinstance(img, str) else img
    img = img.convert("RGBA").convert("L").convert("RGB") if grayscale else img.convert("RGBA").convert("RGB")

    ## Resize the image while keeping the original aspect ratio
    pad_color = 255 if grayscale else (255, 255, 255)
    img = ImageOps.expand(img, (
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=pad_color)

    img = img.resize((img_size, img_size))

    # Predict the embedding
    # get ocr embedding from pretrained paddleOCR
    with torch.no_grad():
        ocr_emb = ocr_main(image_path=img, model=ocr_model, height=None, width=None)
        ocr_emb = ocr_emb[0]
        ocr_emb = ocr_emb[None, ...].to(device)  # remove batch dimension

    # Predict the embedding
    with torch.no_grad():
        img = img_transforms(img)
        img = img[None, ...].to(device)
        logo_feat = model.features(img, ocr_emb)
        logo_feat = l2_norm(logo_feat).squeeze(0).cpu().numpy()  # L2-normalization final shape is (2560,)

    return logo_feat

def pred_brand(model, ocr_model, domain_map, logo_feat_list, file_name_list, shot_path: str, pred_bbox, t_s, grayscale=False):
    '''
    Return predicted brand for one cropped image
    :param model: model to use
    :param domain_map: brand-domain dictionary
    :param logo_feat_list: reference logo feature embeddings
    :param file_name_list: reference logo paths
    :param shot_path: path to the screenshot
    :param pred_bbox: 1x4 np.ndarray/list/tensor bounding box coords
    :param t_s: similarity threshold for siamese
    :param grayscale: convert image(cropped) to grayscale or not
    :return: predicted target, predicted target's domain
    '''

    try:
        img = Image.open(shot_path)
    except OSError:  # if the image cannot be identified, return nothing
        print('Screenshot cannot be open')
        return None, None, None

    ## get predicted box --> crop from screenshot
    cropped = img.crop((pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]))
    img_feat = get_ocr_aided_siamese_embedding(cropped, model, ocr_model, grayscale=grayscale)

    ## get cosine similarity with every protected logo
    sim_list = logo_feat_list @ img_feat.T  # take dot product for every pair of embeddings (Cosine Similarity)
    pred_brand_list = file_name_list

    assert len(sim_list) == len(pred_brand_list)

    ## get top 3 brands
    idx = np.argsort(sim_list)[::-1][:3]
    pred_brand_list = np.array(pred_brand_list)[idx]
    sim_list = np.array(sim_list)[idx]

    # top1,2,3 candidate logos
    top3_logolist = [Image.open(x) for x in pred_brand_list]
    top3_brandlist = [brand_converter(os.path.basename(os.path.dirname(x))) for x in pred_brand_list]
    top3_domainlist = [domain_map[x] for x in top3_brandlist]
    top3_simlist = sim_list

    for j in range(3):
        predicted_brand, predicted_domain = None, None

        ## If we are trying those lower rank logo, the predicted brand of them should be the same as top1 logo, otherwise might be false positive
        if top3_brandlist[j] != top3_brandlist[0]:
            continue

        ## If the largest similarity exceeds threshold
        if top3_simlist[j] >= t_s:
            predicted_brand = top3_brandlist[j]
            predicted_domain = top3_domainlist[j]
            final_sim = top3_simlist[j]

        ## Else if not exceed, try resolution alignment, see if can improve
        else:
            cropped, candidate_logo = resolution_alignment(cropped, top3_logolist[j])
            img_feat = get_ocr_aided_siamese_embedding(cropped, model, ocr_model, grayscale=grayscale)
            logo_feat = get_ocr_aided_siamese_embedding(candidate_logo, model, ocr_model, grayscale=grayscale)
            final_sim = logo_feat.dot(img_feat)
            if final_sim >= t_s:
                predicted_brand = top3_brandlist[j]
                predicted_domain = top3_domainlist[j]
            else:
                break  # no hope, do not try other lower rank logos

        ## If there is a prediction, do aspect ratio check
        if predicted_brand is not None:
            ratio_crop = cropped.size[0] / cropped.size[1]
            ratio_logo = top3_logolist[j].size[0] / top3_logolist[j].size[1]
            # aspect ratios of matched pair must not deviate by more than factor of 2.5
            if max(ratio_crop, ratio_logo) / min(ratio_crop, ratio_logo) > 2.5:
                continue  # did not pass aspect ratio check, try other
            # If pass aspect ratio check, report a match
            else:
                return predicted_brand, predicted_domain, final_sim

    return None, None, top3_simlist[0]

def cache_reference_list(model, ocr_model, targetlist_path: str, grayscale=False):
    '''
    cache the embeddings of the reference list
    '''

    #  Prediction for targetlists
    logo_feat_list = []
    file_name_list = []

    for target in tqdm(os.listdir(targetlist_path)):
        if target.startswith('.'):  # skip hidden files
            continue
        for logo_path in os.listdir(os.path.join(targetlist_path, target)):
            if logo_path.endswith('.png') or logo_path.endswith('.jpeg') or logo_path.endswith('.jpg') or logo_path.endswith('.PNG') \
                    or logo_path.endswith('.JPG') or logo_path.endswith('.JPEG'):
                if logo_path.startswith('loginpage') or logo_path.startswith('homepage'):  # skip homepage/loginpage
                    continue
                logo_feat_list.append(get_ocr_aided_siamese_embedding(img=os.path.join(targetlist_path, target, logo_path),
                                                                      model=model,
                                                                      ocr_model=ocr_model,
                                                                      grayscale=grayscale))
                file_name_list.append(str(os.path.join(targetlist_path, target, logo_path)))

    return np.asarray(logo_feat_list), np.asarray(file_name_list)

def check_domain_brand_inconsistency(logo_boxes,
                                     domain_map_path: str,
                                     model,
                                     ocr_model,
                                     logo_feat_list,
                                     file_name_list,
                                     shot_path: str,
                                     url: str,
                                     ts: float):

    # targetlist domain list
    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)

    # look at boxes for logo class only
    print('number of logo boxes:', len(logo_boxes))
    matched_target, matched_domain, matched_coord, this_conf = None, None, None, None

    # run logo matcher
    if len(logo_boxes) > 0:
        # siamese prediction for logo box
        for i, coord in enumerate(logo_boxes):
            min_x, min_y, max_x, max_y = coord
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            matched_target, matched_domain, this_conf = pred_brand(model, ocr_model, domain_map,
                                                                  logo_feat_list, file_name_list,
                                                                  shot_path, bbox, t_s=ts, grayscale=False)

            # domain matcher to avoid FP
            if matched_target is not None:
                matched_coord = coord
                # if tldextract.extract(url).domain+ '.'+tldextract.extract(url).suffix not in matched_domain:
                if tldextract.extract(url).domain not in matched_domain:
                    # avoid fp due to godaddy domain parking, ignore webmail provider (ambiguous)
                    if matched_target == 'GoDaddy' or matched_target == "Webmail Provider" or matched_target == "Government of the United Kingdom":
                        matched_target = None  # ignore the prediction
                        matched_domain = None  # ignore the prediction
                else:  # benign, real target
                    matched_target = None  # ignore the prediction
                    matched_domain = None  # ignore the prediction
                break  # break if target is matched
            break  # only look at 1st logo

    return brand_converter(matched_target), matched_domain, matched_coord, this_conf



