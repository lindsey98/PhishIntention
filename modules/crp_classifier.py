
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transform
from collections import OrderedDict
from lxml import html
import io
import os
import numpy as np
from utils.utils import read_img_reverse, coord2pixel_reverse, topo2pixel
from collections import OrderedDict  # pylint: disable=g-importing-member
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.models import KNOWN_MODELS

def credential_config(checkpoint, model_type='mixed'):
    '''
    Load credential classifier configurations
    :param checkpoint: classifier weights
    :param model_type: layout|screenshot|mixed|topo
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
        raise ValueError('CRP Model type not supported, please use one of the following [screenshot|layout|mixed|topo]')

    checkpoint = torch.load(checkpoint, map_location="cpu")
    checkpoint = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if not k.startswith('module'):
            new_state_dict[k] = v
            continue
        name = k.split('module.')[1]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def credential_classifier_mixed(img: str, coords, types, model):
    '''
    Call mixed CRP classifier
    :param img: image path
    :param coords: prediction from layout detector
    :param types: prediction from layout detector
    :param model: CRP model
    :return: CRP = 0 or nonCRP = 1
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = Image.open(img).convert('RGB')

    # transform to tensor
    transformation = transform.Compose([transform.Resize((256, 512)),
                                        transform.ToTensor()])
    image = transformation(image)

    # append class channels
    # class grid tensor is of shape 5xHxW
    grid_tensor = coord2pixel_reverse(img_path=img,
                                      coords=coords,
                                      types=types,
                                      reshaped_size=(256, 512))

    image = torch.cat((image.double(), grid_tensor), dim=0)
    assert image.shape == (8, 256, 512)  # ensure correct shape

    # inference
    with torch.no_grad():
        pred_orig = model(image[None, ...].to(device, dtype=torch.float))
        assert pred_orig.shape[-1] == 2  ## assert correct shape
        pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item()  # 'credential': 0, 'noncredential': 1

    return pred


############################################ For HTML heuristic ##########################################################

def read_html(html_path):
    '''
    read html and parse into tree
    :param html_path: path to html.txt
    '''
    done = False
    tree_list = None

    # check if html path exist
    if not os.path.exists(html_path):
        print('Path not exists: {}'.format(html_path))
        return tree_list

    # parse html text
    try:
        with io.open(html_path, 'r', encoding='ISO-8859-1') as f:
            page = f.read()
            tree = html.fromstring(page)
            tree_list = tree
            done = True
    except Exception as e:
        pass

    # try another encoding
    if not done:
        try:
            with open(html_path, 'r', encoding="utf-8") as f:
                page = f.read()
                tree = html.fromstring(page)
                tree_list = tree
                done = True
        except Exception as e:
            pass

    # try another encoding
    if not done:
        try:
            with open(html_path, 'r', encoding='ANSI') as f:
                page = f.read()
                tree = html.fromstring(page)
                tree_list = tree
                done = True
        except Exception as e:
            pass

    return tree_list


def proc_tree(tree, obfuscate=False):
    '''
    returns number of forms, type of forms in a list, number of inputs in each form, number of password field in each form
    :param tree: Element html object
    '''

    if tree is None:  # parsing into tree failed
        return 0, [], [], [], []
    forms = tree.xpath('.//form')  # find form
    if len(forms) == 0:  # no form
        return 0, [], [], [], []
    else:
        if obfuscate:
            for form in forms:
                inputs = form.xpath('.//input')
                for input in inputs:
                    try:
                        if input.get('type') == "password":
                            input.attrib['type'] = "passw0rd"
                    except:
                        pass

        methods = []
        count_inputs = []
        count_password = []
        count_username = []

        for form in forms:
            count = 0
            methods.append(form.get('method'))  # get method of form "post"/"get"

            inputs = form.xpath('.//input')
            count_inputs.append(len(inputs))  # get number if inputs
            inputs = form.xpath('.//input[@type="password"]')  # get number of password fields
            inputs2 = form.xpath(
                './/input[@name="password" and @type!="hidden" and @type!="search" and not(contains(@placeholder, "search")) and @aria-label!="search" and @title!="search"]')
            count_password.append(len(inputs) + len(inputs2))

            usernames = form.xpath('.//input[@type="username"]')  # get number of username fields
            usernames2 = form.xpath(
                './/input[@name="username" and @type!="hidden" and @type!="search" and not(contains(@placeholder, "search")) and @aria-label!="search" and @title!="search"]')  # get number of username fields
            count_username.append(len(usernames) + len(usernames2))

        return len(forms), methods, count_inputs, count_password, count_username


def check_post(x, version=1):
    '''
    check whether html contains postform/user name input field/ password input field
    :param x: Tuple object (len(forms):int, methods:List[str|float], count_inputs:List[int], count_password:List[int], count_username:List[int])
    :return:
    '''

    num_form, methods, num_inputs, num_password, num_username = x
    #     print(num_password, num_username)

    if len(methods) == 0:
        have_postform = 0
    else:
        have_postform = (len([y for y in [x for x in methods if x is not None] if y.lower() == 'post']) > 0)

    if len(num_password) == 0:
        have_password = 0
    else:
        have_password = (np.sum(num_password) > 0)

    if len(num_username) == 0:
        have_username = 0
    else:
        have_username = (np.sum(num_username) > 0)

    # CRP = 0, nonCRP = 1
    if version == 1:
        return 0 if (have_postform) else 1
    elif version == 2:
        return 0 if (have_password | have_username) else 1
    elif version == 3:
        return 0 if (have_postform | have_password | have_username) else 1


def html_heuristic(html_path):
    '''
    Call HTML heuristic
    :param html_path: path to html file
    :return: CRP = 0 or nonCRP = 1
    '''
    tree = read_html(html_path)
    proc_data = proc_tree(tree)
    return check_post(proc_data, version=2)


