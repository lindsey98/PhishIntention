from .FGSM import fgsm
from .JSMA import jsma
from .DeepFool import deepfool
from .CWL2 import cw
from src.OCR.demo import *

import os
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any


class adversarial_attack():
    '''
    Perform adversarial attack
    '''
    def __init__(self, method, model, 
                 dataloader, device, num_classes=10, save_data=False, 
                 use_ocr: bool = False, ocr_model: Any = None):
        '''
        :param method: Which attack method to use
        :param model: subject model to attack
        :param dataloader: dataloader
        :param device: cuda/cpu
        :param num_classes: number of classes for classification model
        :param save_data: save data or not
        '''
        self.method = method
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.save_data = save_data
        self.use_ocr = use_ocr
        self.ocr_model = ocr_model

        
    def batch_attack(self):
        '''
        Run attack on a batch if using ocr_model, batch size should be 1 
        '''
        # Accuracy counter
        correct = 0
        total = 0
        adv_examples = []
        ct_save = 0

        # Loop over all examples in test set
        if self.use_ocr:
            for ct, (data, label, data_path) in tqdm(enumerate(self.dataloader)):
                data = data.to(self.device, dtype=torch.float) 
                label = label.to(self.device, dtype=torch.long)
                data.requires_grad = True

                ocr_emb = ocr_main(image_path=data_path[0], 
                                   model=self.ocr_model, 
                                   height=None, width=None)
                ocr_emb = ocr_emb[0]
                ocr_emb = ocr_emb[None, ...]
                ocr_emb = ocr_emb.to(self.device)
                ocr_emb.requires_grad = False

                # Forward pass the data through the model
                output = self.model(data, ocr_emb)
                self.model.zero_grad()
                init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                if init_pred.item() != label.item():  # initially was incorrect --> no need to generate adversary
                    total += 1
                    print(ct)
                    continue

                # Call Attack
                if self.method in ['fgsm', 'stepll']:
                    criterion = nn.CrossEntropyLoss()
                    perturbed_data = fgsm(self.model, self.method, data, label, criterion, use_ocr=True, ocr_emb=ocr_emb)

                elif self.method == 'jsma':
                    # randomly select a target class
                    target_class = init_pred
                    while target_class == init_pred:
                        target_class = torch.randint(0, self.num_classes, (1,)).to(self.device)
                    print(target_class)
                    perturbed_data = jsma(self.model, self.num_classes, data, target_class, use_ocr=True, ocr_emb=ocr_emb)

                elif self.method == 'deepfool':
                    f_image = output.detach().cpu().numpy().flatten()
                    I = (np.array(f_image)).flatten().argsort()[::-1]
                    perturbed_data = deepfool(self.model, self.num_classes, data, label, I, use_ocr=True, ocr_emb=ocr_emb)

                elif self.method == 'cw':
                    target_class = init_pred
                    while target_class == init_pred:
                        target_class = torch.randint(0, self.num_classes, (1,)).to(self.device)
                    print(target_class)
                    perturbed_data = cw(self.model, self.device, data, label, target_class, use_ocr=True, ocr_emb=ocr_emb)

                else:
                    print('Attack method is not supported， please choose your attack from [fgsm|stepll|jsma|deepfool|cw]')

                # Re-classify the perturbed image
                self.model.zero_grad()
                self.model.eval()
                with torch.no_grad():
                    output = self.model(perturbed_data, ocr_emb)

                # Check for success
                final_pred = output.max(1, keepdim=True)[1]
                if final_pred.item() == init_pred.item():
                    correct += 1  # still correct
                else:# save successful attack
                    print(final_pred)
                    print(init_pred)

                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
                total += 1
                print(ct)
                print("Test Accuracy = {}".format(correct/float(total)))
                
                
        else:
            for ct, (data, label) in tqdm(enumerate(self.dataloader)):
                data = data.to(self.device, dtype=torch.float) 
                label = label.to(self.device, dtype=torch.long)

                # Forward pass the data through the model
                output = self.model(data)
                self.model.zero_grad()
                init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                if init_pred.item() != label.item():  # initially was incorrect --> no need to generate adversary
                    total += 1
                    print(ct)
                    continue

                # Call Attack
                if self.method in ['fgsm', 'stepll']:
                    criterion = nn.CrossEntropyLoss()
                    perturbed_data = fgsm(self.model, self.method, data, label, criterion)

                elif self.method == 'jsma':
                    # randomly select a target class
                    target_class = init_pred
                    while target_class == init_pred:
                        target_class = torch.randint(0, self.num_classes, (1,)).to(self.device)
                    print(target_class)
                    perturbed_data = jsma(self.model, self.num_classes, data, target_class)

                elif self.method == 'deepfool':
                    f_image = output.detach().cpu().numpy().flatten()
                    I = (np.array(f_image)).flatten().argsort()[::-1]
                    perturbed_data = deepfool(self.model, self.num_classes, data, label, I)

                elif self.method == 'cw':
                    target_class = init_pred
                    while target_class == init_pred:
                        target_class = torch.randint(0, self.num_classes, (1,)).to(self.device)
                    print(target_class)
                    perturbed_data = cw(self.model, self.device, data, label, target_class)

                else:
                    print('Attack method is not supported， please choose your attack from [fgsm|stepll|jsma|deepfool|cw]')

                # Re-classify the perturbed image
                self.model.zero_grad()
                self.model.eval()
                with torch.no_grad():
                    output = self.model(perturbed_data)

                # Check for success
                final_pred = output.max(1, keepdim=True)[1]
                if final_pred.item() == init_pred.item():
                    correct += 1  # still correct
                else:# save successful attack
                    print(final_pred)
                    print(init_pred)

                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
                total += 1
                print(ct)
                print("Test Accuracy = {}".format(correct/float(total)))


        # Calculate final accuracy
        final_acc = correct / float(len(self.dataloader))
        print("Test Accuracy = {} / {} = {}".format(correct, total, final_acc))

        # Return the accuracy and an adversarial example
        return final_acc, adv_examples
            
