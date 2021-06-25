
import torch
import numpy as np
import credential_classifier.bit_pytorch.models as models
from credential_classifier.bit_pytorch.dataloader import GetLoader, ImageLoader, HybridLoader
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
from tqdm import tqdm
import logging
os.environ["CUDA_VISIBLE_DEVICES"]="1,0"

def vis_helper(x):
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot("331")
    ax.set_title("X", fontsize=15)
    ax.imshow(x[0].numpy() if not isinstance(x[0], np.ndarray) else x[0], cmap='gray')
    
    ax = plt.subplot("332")
    ax.set_title("Y", fontsize=15)
    ax.imshow(x[1].numpy() if not isinstance(x[1], np.ndarray) else x[1], cmap='gray')
    
    ax = plt.subplot("333")
    ax.set_title("W", fontsize=15)
    ax.imshow(x[2].numpy() if not isinstance(x[2], np.ndarray) else x[2], cmap='gray')
    
    ax = plt.subplot("334")
    ax.set_title("H", fontsize=15)
    ax.imshow(x[3].numpy() if not isinstance(x[3], np.ndarray) else x[3], cmap='gray')
    
    ax = plt.subplot("335")
    ax.set_title("C1(logo)", fontsize=15)
    ax.imshow(x[4].numpy() if not isinstance(x[4], np.ndarray) else x[4], cmap='gray')
    
    ax = plt.subplot("336")
    ax.set_title("C2(input)", fontsize=15)
    ax.imshow(x[5].numpy() if not isinstance(x[5], np.ndarray) else x[5], cmap='gray')
    
    ax = plt.subplot("337")
    ax.set_title("C3(button)", fontsize=15)
    ax.imshow(x[6].numpy() if not isinstance(x[6], np.ndarray) else x[6], cmap='gray')

    ax = plt.subplot("338")
    ax.set_title("C4(label)", fontsize=15)
    ax.imshow(x[7].numpy() if not isinstance(x[7], np.ndarray) else x[7], cmap='gray')

    ax = plt.subplot("339")
    ax.set_title("C5(block)", fontsize=15)
    ax.imshow(x[8].numpy() if not isinstance(x[8], np.ndarray) else x[8], cmap='gray')
    
    plt.show()
    
def evaluate(model, train_loader):
    '''
    :param model: model to be evaluated
    :param train_loader: dataloader to be evaluated
    :return: classification acc
    '''

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b, (x, y) in tqdm(enumerate(train_loader)):
            x = x.to(device, non_blocking=True, dtype=torch.float)
            y = y.to(device, non_blocking=True, dtype=torch.long)

            # Compute output, measure accuracy
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(y).sum().item()
            total += len(logits)
            print("GT:", y)
            print("Pred:", preds)
            print(correct, total)

    return float(correct/total)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", choices=list(models.KNOWN_MODELS.keys()),required=True,
                      help="Which variant to use; BiT-M gives best results.")
    parser.add_argument("--weights_path", required=True, help="Where to load pretrained weights")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.KNOWN_MODELS[args.model](head_size=2)
    
    logging.basicConfig(filename='credential_classifier/evaluate_acc.log', level=logging.INFO)
    logger = logging.getLogger('trace')
    
    # load weights
    checkpoint = torch.load(args.weights_path, map_location="cpu")
    checkpoint = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if not k.startswith('module'):
            new_state_dict[k]=v
            continue
        name = k.split('module.')[1]
        new_state_dict[name]=v
        
    model.load_state_dict(new_state_dict)
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    # evaluate
    if args.model.startswith('FCMax'):
        train_set = GetLoader(img_folder='../../datasets/train_merge_imgs_grid2',
                          annot_path='../../datasets/train_al_grid_merge_coords2.txt')
        val_set = GetLoader(img_folder='../../datasets/val_merge_imgs',
                          annot_path='../../datasets/val_merge_coords.txt')    
        
    elif args.model == 'BiT-M-R50x1':
        train_set = ImageLoader(img_folder='../datasets/train_merge_imgs',
                          annot_path='../datasets/train_al_merge_coords2.txt')
        val_set = ImageLoader(img_folder='../datasets/val_merge_imgs',
                          annot_path='../datasets/val_merge_coords.txt')
        
    else:
        train_set = HybridLoader(img_folder='../datasets/train_imgs',
                          annot_path='../datasets/train_coords.txt')
        val_set = HybridLoader(img_folder='../datasets/val_merge_imgs',
                          annot_path='../datasets/val_merge_coords.txt')
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, drop_last=False, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, drop_last=False, shuffle=False)
    
    acc = evaluate(model, train_loader)
    print('Training Acc : {:.4f}'.format(acc))
    logger.info('Training Acc : {:.4f}'.format(acc))
    
    acc = evaluate(model, val_loader)
    print('Validation Acc : {:.4f}'.format(acc))
    logger.info('Validation Acc : {:.4f}'.format(acc))


