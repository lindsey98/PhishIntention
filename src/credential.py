from .credential_classifier.bit_pytorch.models import KNOWN_MODELS
from .credential_classifier.bit_pytorch.grid_divider import read_img_reverse, coord2pixel_reverse, topo2pixel
from .credential_classifier.HTML_heuristic.post_form import *

# from .layout_matcher.heuristic import layout_heuristic
# from .layout_matcher.topology import knn_matrix
# from .layout_matcher.misc import preprocess

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
    elif model_type == 'mixed':
        model = KNOWN_MODELS['BiT-M-R50x1V2'](head_size=2)
    elif model_type == 'topo':
        model = KNOWN_MODELS['BiT-M-R50x1V3'](head_size=2)
    else:
        raise NotImplementedError
        
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


# def credential_classifier_topo(img:str, coords, types, model):
#     # process it into grid_array
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     image = Image.open(img).convert('RGB')
    
#     # transform to tensor
#     transformation = transform.Compose([transform.Resize((256, 256)), 
#                                         transform.ToTensor()])
#     image_T = transformation(image)
    
#     # append class channels
#     # class grid tensor is of shape 5xHxW
#     grid_tensor = coord2pixel_reverse(img_path=img,
#                               coords=coords, 
#                               types=types)

#     image_T = torch.cat((image_T.double(), grid_tensor), dim=0)
#     assert image_T.shape == (8, 256, 256) # ensure correct shape
    
#     # get topological tensor
#     compos = coords[types != 4] # remove block
#     resize_compos = preprocess(image.size[:2][::-1], coords.numpy()) # Rescale all coords to be [0, 100], used to compute KNN matrix
#     if len(compos) > 0:
#         box_matrix, _ = knn_matrix(compos=resize_compos, k=3, norm_method='log') # layout extraction -- KNN matrix computation
#         box_matrix = box_matrix.reshape(box_matrix.shape[0], -1) # reshape to be Nx(KxZ)

#         topo_tensor = topo2pixel(img_path=img, 
#                                  coords=compos, 
#                                  knn_matrix=box_matrix).double()
#     else:
#         topo_tensor = torch.zeros((12, 256, 256)).double() # no component


#     # inference
#     with torch.no_grad():
#         pred_orig = model(image_T[None,...].to(device, dtype=torch.float), topo_tensor[None, ...].to(device, dtype=torch.float))
#         assert pred_orig.shape[-1] == 2 ## assert correct shape
#         pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
        
#     return pred

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



# def credential_classifier(img:str, coords, types, model):
#     '''
#     Run credential classifier
#     :param img: path to image
#     :param coords: torch.Tensor/np.ndarray Nx4 bbox coords
#     :param types: torch.Tensor/np.ndarray Nx4 bbox types
#     :param model: classifier 
#     :return pred: predicted class 'credential': 0, 'noncredential': 1
#     :return conf: prediction confidence
#     '''

#     # process it into grid_array
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     grid_arr = read_img_reverse(img, coords, types)
#     assert grid_arr.shape == (9, 10, 10) # ensure correct shape

#     # inference
#     with torch.no_grad():
#         pred_orig = model(grid_arr.type(torch.float).to(device))
#         pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
# #             conf, _ = torch.max(F.softmax(pred_orig, dim=-1), dim=-1)
# #             conf = conf.item()
        
# #     return pred, conf
#     return pred

# def credential_classifier_screenshot(img:str, model):
#     '''
#     Run credential classifier on screenshot
#     :param img: path to image
#     :param coords: torch.Tensor/np.ndarray Nx4 bbox coords
#     :param types: torch.Tensor/np.ndarray Nx4 bbox types
#     :param model: classifier 
#     :return pred: predicted class 'credential': 0, 'noncredential': 1
#     :return conf: prediction confidence
#     '''

#     # process it into grid_array
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     img = Image.open(img).convert('RGB')
    
#     # transform to tensor
#     transformation = transform.Compose([transform.Resize((256, 256)), 
#                                         transform.ToTensor()])
#     image = transformation(img)

#     # inference
#     with torch.no_grad():
#         pred_orig = model(image[None,...].to(device, dtype=torch.float))
#         assert pred_orig.shape[-1] == 2 ## assert correct shape
#         pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
        
#     return pred


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
    # class grid tensor is of shape 8xHxW
    grid_tensor = coord2pixel_reverse(img_path=img, coords=coords, types=types)

    image = torch.cat((image.double(), grid_tensor), dim=0)
    assert image.shape == (8, 256, 256) # ensure correct shape

    # inference
    with torch.no_grad():
       
        pred_features = model.features(image[None,...].to(device, dtype=torch.float))
        pred_orig = model(image[None,...].to(device, dtype=torch.float))
        pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
        conf= F.softmax(pred_orig, dim=-1).detach().cpu()
        
    return pred, conf, pred_features


# def credential_classifier_al(img:str, coords, types, model):
#     '''
#     Run credential classifier for AL dataset
#     :param img: path to image
#     :param coords: torch.Tensor/np.ndarray Nx4 bbox coords
#     :param types: torch.Tensor/np.ndarray Nx4 bbox types
#     :param model: classifier 
#     :return pred: predicted class 'credential': 0, 'noncredential': 1
#     :return conf: torch.Tensor NxC prediction confidence
#     '''
#     # process it into grid_array
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     grid_arr = read_img_reverse(img, coords, types)
#     assert grid_arr.shape == (9, 10, 10) # ensure correct shape

#     # inference
#     with torch.no_grad():
#         pred_features = model.features(grid_arr.type(torch.float).to(device))
#         pred_orig = model(grid_arr.type(torch.float).to(device))
#         pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
#         conf= F.softmax(pred_orig, dim=-1).detach().cpu()
        
#     return pred, conf, pred_features


# def credential_classifier_al_screenshot(img:str, model):
#     '''
#     Run credential classifier for AL dataset
#     :param img: path to image
#     :param coords: torch.Tensor/np.ndarray Nx4 bbox coords
#     :param types: torch.Tensor/np.ndarray Nx4 bbox types
#     :param model: classifier 
#     :return pred: predicted class 'credential': 0, 'noncredential': 1
#     :return conf: torch.Tensor NxC prediction confidence
#     '''
#     # process it into grid_array
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     img = Image.open(img).convert('RGB')
    
#     # transform to tensor
#     transformation = transform.Compose([transform.Resize((256, 256)), 
#                                         transform.ToTensor()])
#     image = transformation(img)
    
#     # inference
#     with torch.no_grad():
#         pred_features = model.features(image[None,...].to(device, dtype=torch.float))
#         pred_orig = model(image[None,...].to(device, dtype=torch.float))
#         pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
#         conf= F.softmax(pred_orig, dim=-1).detach().cpu()
        
#     return pred, conf, pred_features


############################################ For HTML heuristic ##########################################################

def html_heuristic(html_path):
    tree = read_html(html_path)
    proc_data = proc_tree(tree)
    return check_post(proc_data, version=2)




