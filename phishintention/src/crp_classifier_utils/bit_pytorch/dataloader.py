import torch.utils.data as data
import numpy as np
from phishintention.src.crp_classifier_utils.bit_pytorch.grid_divider import read_img, coord2pixel, topo2pixel
from phishintention.src.crp_classifier_utils.bit_pytorch.utils import read_txt, read_txt_screenshot
import torchvision.transforms as transform
import os
import torch

import matplotlib.pyplot as plt
from PIL import Image
import cv2

class HybridLoader(data.Dataset):
    '''
    Dataloader for mixed classifier
    '''
    def __init__(self, img_folder: str, annot_path: str):
        self.img_folder = img_folder
        self.annot_path = annot_path
        self.num_imgs, self.labels, self.paths, self.preprocess_coordinates, self.img_classes = read_txt(annot_path)

        self.classes = {'credential': 0, 'noncredential': 1}
        self.transform = transform.Compose([transform.Resize((256, 512)),
                                            transform.ToTensor(),
                                            ])

    def __getitem__(self, item: int):

        image_file = list(set(self.paths))[item] # image path
        img_label = self.classes[np.asarray(self.labels)[np.asarray(self.paths) == image_file][0]] # credential/non-credential

        img_coords = np.asarray(self.preprocess_coordinates)[np.asarray(self.paths) == image_file] # box coordinates
        img_classes = np.asarray(self.img_classes)[np.asarray(self.paths) == image_file] # box types
        if len(img_coords) == 0:
            raise IndexError('list index out of range')
        
        # original image shape is 3xHxW
        image = Image.open(os.path.join(self.img_folder, image_file+'.png')).convert('RGB')
        image = self.transform(image) 
        
        # append class channels
        # class grid tensor is of shape 5xHxW
        grid_tensor = coord2pixel(img_path=os.path.join(self.img_folder, image_file+'.png'),
                                  coords=img_coords, 
                                  types=img_classes)
    
        image = torch.cat((image.double(), grid_tensor), dim=0)

        return image, img_label, image_file

    def __len__(self):
        return self.num_imgs

# class HybridLoaderDebug(data.Dataset):
#     '''
#         Dataloader for mixed classifier, return the image file as well
#         '''
#     def __init__(self, img_folder: str, annot_path: str):
#         self.img_folder = img_folder
#         self.annot_path = annot_path
#         self.num_imgs, self.labels, self.paths, self.preprocess_coordinates, self.img_classes = read_txt(annot_path)
#
#         self.classes = {'credential': 0, 'noncredential': 1}
#         self.transform = transform.Compose([transform.Resize((256, 512)),
#                                             transform.ToTensor()])
#
#     def __getitem__(self, item: int):
#         image_file = list(set(self.paths))[item]  # image path
#         img_label = self.classes[
#             np.asarray(self.labels)[np.asarray(self.paths) == image_file][0]]  # credential/non-credential
#
#         img_coords = np.asarray(self.preprocess_coordinates)[np.asarray(self.paths) == image_file]  # box coordinates
#         img_classes = np.asarray(self.img_classes)[np.asarray(self.paths) == image_file]  # box types
#         if len(img_coords) == 0:
#             raise IndexError('list index out of range')
#
#         # original image shape is 3xHxW
#         image = Image.open(os.path.join(self.img_folder, image_file + '.png')).convert('RGB')
#         image = self.transform(image)
#
#         # append class channels
#         # class grid tensor is of shape 5xHxW
#         grid_tensor = coord2pixel(img_path=os.path.join(self.img_folder, image_file + '.png'),
#                                   coords=img_coords,
#                                   types=img_classes,
#                                   reshaped_size=(256, 512))
#
#         image = torch.cat((image.double(), grid_tensor), dim=0)
#
#         return image, img_label, image_file
#
#     def __len__(self):
#         return self.num_imgs

class LayoutLoader(data.Dataset):
    '''
    Data loader for layout-only classifier
    '''

    def __init__(self, img_folder: str, annot_path: str):
        self.img_folder = img_folder
        self.annot_path = annot_path
        self.num_imgs, self.labels, self.paths, self.preprocess_coordinates, self.img_classes = read_txt(annot_path)
        self.classes = {'credential': 0, 'noncredential': 1}

    def __getitem__(self, item: int):

        image_file = list(set(self.paths))[item] # image path
        img_coords = np.asarray(self.preprocess_coordinates)[np.asarray(self.paths) == image_file] # box coordinates
        img_classes = np.asarray(self.img_classes)[np.asarray(self.paths) == image_file] # box types
        if len(img_coords) == 0:
            raise IndexError('list index out of range')

        img_label = self.classes[np.asarray(self.labels)[np.asarray(self.paths) == image_file][0]] # credential/non-credential

        grid_arr = read_img(img_path=os.path.join(self.img_folder, image_file+'.png'),
                            coords=img_coords, types=img_classes, grid_num=10)

        return grid_arr, img_label, image_file

    def __len__(self):
        return self.num_imgs

# class LayoutLoaderDebug(data.Dataset):
#     '''
#     Data loader for layout-only classifier, return image file as well
#     '''
#
#     def __init__(self, img_folder: str, annot_path: str):
#         self.img_folder = img_folder
#         self.annot_path = annot_path
#         self.num_imgs, self.labels, self.paths, self.preprocess_coordinates, self.img_classes = read_txt(annot_path)
#         self.classes = {'credential': 0, 'noncredential': 1}
#
#     def __getitem__(self, item: int):
#
#         image_file = list(set(self.paths))[item] # image path
#         img_coords = np.asarray(self.preprocess_coordinates)[np.asarray(self.paths) == image_file] # box coordinates
#         img_classes = np.asarray(self.img_classes)[np.asarray(self.paths) == image_file] # box types
#         if len(img_coords) == 0:
#             raise IndexError('list index out of range')
#
#         img_label = self.classes[np.asarray(self.labels)[np.asarray(self.paths) == image_file][0]] # credential/non-credential
#
#         grid_arr = read_img(img_path=os.path.join(self.img_folder, image_file+'.png'),
#                             coords=img_coords, types=img_classes, grid_num=10)
#
#         return grid_arr, img_label, image_file
#
#     def __len__(self):
#         return self.num_imgs

class ScreenshotLoader(data.Dataset):
    '''
    Dataloader for screenshot-only classifier
    '''
    def __init__(self, img_folder: str, annot_path: str):
        self.img_folder = img_folder
        self.annot_path = annot_path
        self.num_imgs, self.labels, self.paths = read_txt_screenshot(annot_path)
        self.classes = {'credential': 0, 'noncredential': 1}
        self.transform = transform.Compose([transform.Resize((256, 512)),
                                            transform.ToTensor(),
                                            ])

    def __getitem__(self, item: int):

        image_file = list(set(self.paths))[item] # image path
        img_label = self.classes[np.asarray(self.labels)[np.asarray(self.paths) == image_file][0]] # credential/non-credential
        image = Image.open(os.path.join(self.img_folder, image_file+'.png')).convert('RGB')
        image = self.transform(image)
        return image, img_label, image_file

    def __len__(self):
        return self.num_imgs

# class ScreenshotLoaderDebug(data.Dataset):
#
#     def __init__(self, img_folder: str, annot_path: str):
#         self.img_folder = img_folder
#         self.annot_path = annot_path
#         self.num_imgs, self.labels, self.paths = read_txt_screenshot(annot_path)
#         self.classes = {'credential': 0, 'noncredential': 1}
#         self.transform = transform.Compose([transform.Resize((256, 256)),
#                                             transform.ToTensor()])
#
#     def __getitem__(self, item: int):
#
#         image_file = list(set(self.paths))[item] # image path
#
#         img_label = self.classes[np.asarray(self.labels)[np.asarray(self.paths) == image_file][0]] # credential/non-credential
#
#         image = Image.open(os.path.join(self.img_folder, image_file+'.png')).convert('RGB')
#
#         image = self.transform(image)
#
#         return image, img_label, image_file
#
#     def __len__(self):
#         return self.num_imgs


if __name__ == '__main__':
    
    train_set_orig = ScreenshotLoader(img_folder='../datasets/train_imgs',
                                      annot_path='../datasets/train_coords.txt')
        
    train_set = ScreenshotLoader(img_folder='../datasets/train_merge_imgs',
                                 annot_path='../datasets/train_merge_coords.txt')

    val_set = ScreenshotLoader(img_folder='../datasets/val_imgs',
                               annot_path='../datasets/val_coords.txt')
    
    print(len(train_set_orig))
    print(len(train_set))
    print(len(val_set))
#     train_loader = torch.utils.data.DataLoader(
#         train_set, batch_size=32, drop_last=False, shuffle=False)

#     val_loader = torch.utils.data.DataLoader(
#         test_set, batch_size=32, drop_last=False, shuffle=False)

