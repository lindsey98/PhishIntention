import torch.utils.data as data
from PIL import Image, ImageOps
import pickle
import numpy as np
import os
import torch
from phishintention.src.OCR_siamese_utils.demo import *


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, label_dict, ocr_model,
                 transform=None, grayscale=False):

        self.ocr_model = ocr_model
        self.transform = transform
        self.data_root = data_root
        self.grayscale = grayscale
        data_list = [x.strip('\n') for x in open(data_list).readlines()]

        with open(label_dict, 'rb') as handle:
            self.label_dict = pickle.load(handle)

        self.classes = list(self.label_dict.keys())

        self.n_data = len(data_list)

        self.img_paths = []
        self.labels = []

        for data in data_list:
            image_path = data
            label = image_path.split('/')[0]
            self.img_paths.append(image_path)
            self.labels.append(label)

    def __getitem__(self, item):

        img_path, label= self.img_paths[item], self.labels[item]
        img_path_full = os.path.join(self.data_root, img_path)
        if self.grayscale:
            img = Image.open(img_path_full).convert('L').convert('RGB')
        else:
            img = Image.open(img_path_full).convert('RGB')

        img = ImageOps.expand(img, (
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))

        # label = np.array(label,dtype='float32')
        label = self.label_dict[label]
        if self.transform is not None:
            img = self.transform(img)

        # get ocr embedding from pretrained paddleOCR
        ocr_emb = ocr_main(image_path=img_path_full, model=self.ocr_model, height=None, width=None)
        ocr_emb = ocr_emb[0] # remove batch dimension
        return img, label, ocr_emb

    def __len__(self):
        return self.n_data


if __name__ == '__main__':
    import torchvision as tv
    import siamese_unified.bit_hyperrule as bit_hyperrule
    

    # load OCR model
    ocr_model = ocr_model_config(checkpoint='/home/l/liny/ruofan/PhishIntention/src/OCR/demo.pth.tar')

    precrop, crop = bit_hyperrule.get_resolution_from_dataset('logo_2k')

    train_tx = tv.transforms.Compose([
            tv.transforms.Resize((precrop, precrop)),
            tv.transforms.RandomCrop((crop, crop)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = GetLoader(data_root='/home/l/liny/ruofan/PhishIntention/src/siamese_retrain/logo2k/Logo-2K+',
                          data_list='/home/l/liny/ruofan/PhishIntention/src/siamese_retrain/logo2k/train.txt',
                          label_dict='/home/l/liny/ruofan/PhishIntention/src/siamese_retrain/logo2k/logo2k_labeldict.pkl',
                          transform=train_tx,
                          ocr_model=ocr_model,
                          )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=12, shuffle=False)

    for b, (x, y, emb) in enumerate(train_loader):
        print(x.shape)
        print(y.shape)
        print(emb.shape)
        break

