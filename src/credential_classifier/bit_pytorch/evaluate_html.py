import torch
import numpy as np
import credential_classifier.bit_pytorch.models as models
from credential_classifier.bit_pytorch.dataloader_debug import *
from credential_classifier.HTML_heuristic.post_form import *
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def html_heuristic(html_path):
    tree = read_html(html_path)
    proc_data = proc_tree(tree)
    return check_post(proc_data, version=2)


def evaluate(model, train_loader, all_folder):
    '''
    :param model: model to be evaluated
    :param train_loader: dataloader to be evaluated
    :return: classification acc
    '''

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b, (x, y, file) in tqdm(enumerate(train_loader)):

            x = x.to(device, non_blocking=True, dtype=torch.float)
            y = y.to(device, non_blocking=True, dtype=torch.long)
            total += len(x)

            html_path = os.path.join(all_folder, file[0], 'html.txt')
            if not os.path.exists(html_path):
                print(html_path, ' not exist')
                preds = 1
            else:
                preds = html_heuristic(html_path)

            ############ if use HTML heuristic only: HTML heuristic is correct
            #             if preds == y.item():
            #                 correct += 1
            #                 print("\nGT:", y.item())
            #                 print("Pred by HTML:", preds)
            #                 print(correct, total)
            #######################################################

            ############ if use HTML+CV ########################

            if preds == 0 and preds == y.item():  # if HTML heuristic says it has post form
                correct += 1
                print("\nGT:", y.item())
                print("Pred by HTML:", preds)
                print(correct, total)

            else:  # inconclusive, use CRP classifier
                # Compute output, measure accuracy
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(y).sum().item()
                print("\nGT:", y.item())
                print("Pred by classifier:", preds.item())
                print(correct, total)
            ########################################################

    return float(correct / total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--weights_path", required=True, help="Where to load pretrained weights")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.KNOWN_MODELS['BiT-M-R50x1V2'](head_size=2)
#     model = models.KNOWN_MODELS['FCMaxV2'](head_size=2)

    # load weights
    checkpoint = torch.load(args.weights_path, map_location="cpu")
    checkpoint = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if not k.startswith('module'):
            new_state_dict[k] = v
            continue
        name = k.split('module.')[1]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = torch.nn.DataParallel(model)
    model.to(device)

    # evaluate
    train_img_folder = '/home/l/liny/ruofan/PhishIntention/datasets/train_imgs'
    train_all_folder = '/home/l/liny/ruofan/PhishIntention/datasets/train_merge_folder'

    test_img_folder = '/home/l/liny/ruofan/PhishIntention/datasets/val_merge_imgs'
    test_all_folder = '/home/l/liny/ruofan/PhishIntention/datasets/val_merge_folder'

    train_set = HybridLoaderDebug(img_folder=train_img_folder,
                          annot_path='/home/l/liny/ruofan/PhishIntention/datasets/train_coords.txt')

    val_set = HybridLoaderDebug(img_folder=test_img_folder,
                          annot_path='/home/l/liny/ruofan/PhishIntention/datasets/val_merge_coords.txt')

#     train_set = GetLoaderDebug(img_folder=train_img_folder,
#                                annot_path='/home/l/liny/ruofan/PhishIntention/datasets/train_coords.txt')

#     val_set = GetLoaderDebug(img_folder=test_img_folder,
#                              annot_path='/home/l/liny/ruofan/PhishIntention/datasets/val_merge_coords.txt')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, drop_last=False, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, drop_last=False, shuffle=False)

    acc = evaluate(model, train_loader, train_all_folder)
    print('Training Acc : {:.4f}'.format(acc))
    #
    acc = evaluate(model, val_loader, test_all_folder)
    print('Validation Acc : {:.4f}'.format(acc))




