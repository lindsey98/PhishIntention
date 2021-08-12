
from siamese_unified.bit_pytorch.train import run_eval, mktrainval
import siamese_unified.bit_common as bit_common
import logging
import siamese_unified.bit_pytorch.models as models
import torch
from os.path import join as pjoin



if __name__ == '__main__':
    
    parser = bit_common.argparser(models.KNOWN_MODELS.keys())
    args = parser.parse_args()
    logger = bit_common.setup_logger(args)
    
    train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)

    logger = logging.getLogger('evaluate_{}'.format(args.name))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model from {}.npz".format(args.model))
    model = models.KNOWN_MODELS[args.model](head_size=len(valid_set.classes), zero_head=True)

    savename = pjoin(args.logdir, args.name, "bit.pth.tar")
    checkpoint = torch.load(savename, map_location="cpu")
    logger.info("Found saved model to resume from at '{}'".format(savename))
    model.load_state_dict(checkpoint["model"])

    model = model.to(device)
    run_eval(model, valid_loader, device, logger, step='eval')


