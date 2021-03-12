from .credential_classifier.bit_pytorch.models import FCMaxPoolV2, KNOWN_MODELS
from .credential_classifier.bit_pytorch.grid_divider import read_img_reverse, coord2pixel_reverse
from .layout_matcher.heuristic import layout_heuristic
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transform
from collections import OrderedDict

def credential_config(checkpoint, model_type='mixed'):
    '''
    Load credential classifier configurations
    :param checkpoint: classifier weights
    :return model: classifier
    '''
    # load weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'screenshot':
        model = KNOWN_MODELS['BiT-M-R50x1'](head_size=2)
    elif model_type == 'layout':
        model = KNOWN_MODELS['FCMaxV2'](head_size=2)
    else:
        model = KNOWN_MODELS['BiT-M-R50x1V2'](head_size=2)
        
    checkpoint = torch.load(checkpoint, map_location="cpu")
    checkpoint = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if not k.startswith('module'):
            new_state_dict[k]=v
            continue
        name = k.split('module.')[1]
        new_state_dict[name]=v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def credential_classifier_mixed(img:str, coords, types, model):
    # process it into grid_array
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = Image.open(img).convert('RGB')
    
    # transform to tensor
    transformation = transform.Compose([transform.Resize((256, 256)), 
                                        transform.ToTensor()])
    image = transformation(image)
    
    # append class channels
    # class grid tensor is of shape 5xHxW
    grid_tensor = coord2pixel_reverse(img_path=img,
                              coords=coords, 
                              types=types)

    image = torch.cat((image.double(), grid_tensor), dim=0)
    assert image.shape == (8, 256, 256) # ensure correct shape

    # inference
    with torch.no_grad():
        pred_orig = model(image[None,...].to(device, dtype=torch.float))
        assert pred_orig.shape[-1] == 2 ## assert correct shape
        pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
        
    return pred



def credential_classifier(img:str, coords, types, model):
    '''
    Run credential classifier
    :param img: path to image
    :param coords: torch.Tensor/np.ndarray Nx4 bbox coords
    :param types: torch.Tensor/np.ndarray Nx4 bbox types
    :param model: classifier 
    :return pred: predicted class 'credential': 0, 'noncredential': 1
    :return conf: prediction confidence
    '''

    # process it into grid_array
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid_arr = read_img_reverse(img, coords, types)
    assert grid_arr.shape == (9, 10, 10) # ensure correct shape

    # inference
    with torch.no_grad():
        pred_orig = model(grid_arr.type(torch.float).to(device))
        pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
#             conf, _ = torch.max(F.softmax(pred_orig, dim=-1), dim=-1)
#             conf = conf.item()
        
#     return pred, conf
    return pred

def credential_classifier_screenshot(img:str, model):
    '''
    Run credential classifier on screenshot
    :param img: path to image
    :param coords: torch.Tensor/np.ndarray Nx4 bbox coords
    :param types: torch.Tensor/np.ndarray Nx4 bbox types
    :param model: classifier 
    :return pred: predicted class 'credential': 0, 'noncredential': 1
    :return conf: prediction confidence
    '''

    # process it into grid_array
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = Image.open(img).convert('RGB')
    
    # transform to tensor
    transformation = transform.Compose([transform.Resize((256, 256)), 
                                        transform.ToTensor()])
    image = transformation(img)

    # inference
    with torch.no_grad():
        pred_orig = model(image[None,...].to(device, dtype=torch.float))
        assert pred_orig.shape[-1] == 2 ## assert correct shape
        pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
        
    return pred


def credential_classifier_mixed_al(img:str, coords, types, model):
    '''
    Run credential classifier for AL dataset
    :param img: path to image
    :param coords: torch.Tensor/np.ndarray Nx4 bbox coords
    :param types: torch.Tensor/np.ndarray Nx4 bbox types
    :param model: classifier 
    :return pred: predicted class 'credential': 0, 'noncredential': 1
    :return conf: torch.Tensor NxC prediction confidence
    '''
    # process it into grid_array
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = Image.open(img).convert('RGB')
    
    # transform to tensor
    transformation = transform.Compose([transform.Resize((256, 256)), 
                                        transform.ToTensor()])
    image = transformation(image)
    
    # append class channels
    # class grid tensor is of shape 5xHxW
    grid_tensor = coord2pixel_reverse(img_path=img,
                              coords=coords, 
                              types=types)

    image = torch.cat((image.double(), grid_tensor), dim=0)
    assert image.shape == (8, 256, 256) # ensure correct shape

    # inference
    with torch.no_grad():
       
        pred_features = model.features(image[None,...].to(device, dtype=torch.float))
        pred_orig = model(image[None,...].to(device, dtype=torch.float))
        pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
        conf= F.softmax(pred_orig, dim=-1).detach().cpu()
        
    return pred, conf, pred_features


def credential_classifier_al(img:str, coords, types, model):
    '''
    Run credential classifier for AL dataset
    :param img: path to image
    :param coords: torch.Tensor/np.ndarray Nx4 bbox coords
    :param types: torch.Tensor/np.ndarray Nx4 bbox types
    :param model: classifier 
    :return pred: predicted class 'credential': 0, 'noncredential': 1
    :return conf: torch.Tensor NxC prediction confidence
    '''
    # process it into grid_array
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid_arr = read_img_reverse(img, coords, types)
    assert grid_arr.shape == (9, 10, 10) # ensure correct shape

    # inference
    with torch.no_grad():
        pred_features = model.features(grid_arr.type(torch.float).to(device))
        pred_orig = model(grid_arr.type(torch.float).to(device))
        pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
        conf= F.softmax(pred_orig, dim=-1).detach().cpu()
        
    return pred, conf, pred_features


def credential_classifier_al_screenshot(img:str, model):
    '''
    Run credential classifier for AL dataset
    :param img: path to image
    :param coords: torch.Tensor/np.ndarray Nx4 bbox coords
    :param types: torch.Tensor/np.ndarray Nx4 bbox types
    :param model: classifier 
    :return pred: predicted class 'credential': 0, 'noncredential': 1
    :return conf: torch.Tensor NxC prediction confidence
    '''
    # process it into grid_array
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = Image.open(img).convert('RGB')
    
    # transform to tensor
    transformation = transform.Compose([transform.Resize((256, 256)), 
                                        transform.ToTensor()])
    image = transformation(img)
    
    # inference
    with torch.no_grad():
        pred_features = model.features(image[None,...].to(device, dtype=torch.float))
        pred_orig = model(image[None,...].to(device, dtype=torch.float))
        pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
        conf= F.softmax(pred_orig, dim=-1).detach().cpu()
        
    return pred, conf, pred_features


def credential_overall(img_path, cls_model, pred_boxes, pred_classes):

    # Credential heuristic module
    pattern_ct, len_input = layout_heuristic(pred_boxes, pred_classes)
    if len_input == 0:
        cre_pred = 1
    elif pattern_ct >= 2:
        cre_pred = 0
    else:
        # Credential classifier module
        cre_pred, _, _ = credential_classifier_al(img=img_path, coords=pred_boxes, 
                                                  types=pred_classes, model=cls_model)

    return cre_pred