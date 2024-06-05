import torch
import numpy as np
import phishintention.src.crp_classifier_utils.bit_pytorch.models as models
from phishintention.src.crp_classifier_utils.bit_pytorch.dataloader import *
from phishintention.src.crp_classifier_utils.HTML_heuristic.post_form import *
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def html_heuristic(html_path, obfuscate=False):
    tree = read_html(html_path)
    proc_data = proc_tree(tree, obfuscate=obfuscate)
    return check_post(proc_data, version=2)

def evaluate(model, train_loader, all_folder, obfuscate=False, method='cv_html'):
    '''
    :param model: model to be evaluated
    :param train_loader: dataloader to be evaluated
    :return: classification acc
    '''

    model.eval()
    correct = 0
    total = 0
    predicted_results = []
    gt_results = []
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
                preds = html_heuristic(html_path, obfuscate)

            ############ if use HTML heuristic only: HTML heuristic is correct
            if method == 'html':
                if preds == y.item():
                    correct += 1
                    print("\nGT:", y.item())
                    print("Pred by HTML:", preds)
                    print(correct, total)
            #######################################################

            ############ if use HTML+CV ########################
            elif method == 'cv_html':
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

            ############ if use CV only ########################
            # Compute output, measure accuracy
            elif method == 'cv':
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(y).sum().item()
                print("\nGT:", y.item())
                print("Pred by classifier:", preds.item())
                print(correct, total)
            ########################################################

            predicted_results.append(preds)
            gt_results.append(y.item())

    return float(correct / total), predicted_results, gt_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    # parser.add_argument("--weights_path", default='./src/credential_classifier/output/Increase_resolution_lr0.005/BiT-M-R50x1V2_0.005.pth.tar',
    #                         help="Where to load pretrained weights")
    # parser.add_argument("--weights_path", default='./src/credential_classifier/output/screenshot/screenshot/BiT-M-R50x1_0.01.pth.tar',
    #                     help="Where to load pretrained weights")
    parser.add_argument("--weights_path", default='./src/credential_classifier/output/website_finetune/websiteV2/FCMaxV2_0.005.pth.tar',
                        help="Where to load pretrained weights")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = models.KNOWN_MODELS['BiT-M-R50x1V2'](head_size=2)
    # model = models.KNOWN_MODELS['BiT-M-R50x1'](head_size=2)
    model = models.KNOWN_MODELS['FCMaxV2'](head_size=2)

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
    train_img_folder = './datasets/train_imgs'
    train_all_folder = './datasets/train_merge_folder'

    test_img_folder = './datasets/val_merge_imgs'
    test_all_folder = './datasets/val_merge_folder'

    # train_set = HybridLoader(img_folder=train_img_folder,
    #                       annot_path='./datasets/train_coords.txt')
    # # #
    # val_set = HybridLoader(img_folder=test_img_folder,
    #                       annot_path='./datasets/val_merge_coords.txt')
    #
    # train_set = ScreenshotLoader(img_folder=train_img_folder,
    #                            annot_path='./datasets/train_coords.txt')
    # #
    # val_set = ScreenshotLoader(img_folder=test_img_folder,
    #                          annot_path='./datasets/val_merge_coords.txt')

    train_set = LayoutLoader(img_folder=train_img_folder,
                               annot_path='./datasets/train_coords.txt')

    val_set = LayoutLoader(img_folder=test_img_folder,
                             annot_path='./datasets/val_merge_coords.txt')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, drop_last=False, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, drop_last=False, shuffle=False)

    # acc, predicted_results, gt_results = evaluate(model, train_loader, train_all_folder, obfuscate=False, method='cv_html')
    # print('Training Acc : {:.4f}'.format(acc))
    #
    acc, predicted_results, gt_results = evaluate(model, val_loader, test_all_folder, obfuscate=False, method='cv_html')
    print('Acc : {:.4f}'.format(acc))
    print('Recall : {:.4f}'.format(np.sum((np.asarray(gt_results) == np.asarray(predicted_results)) & (np.asarray(gt_results) == 0)) / np.sum(np.asarray(gt_results) == 0) ))
    print('Precision : {:.4f}'.format(np.sum((np.asarray(gt_results) == np.asarray(predicted_results)) & (np.asarray(gt_results) == 0)) / np.sum(np.asarray(predicted_results) == 0)))





