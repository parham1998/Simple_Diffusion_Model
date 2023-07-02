# =============================================================================
# Import required libraries
# =============================================================================
import argparse
import numpy as np

import torch

from dataset import make_dataloader
from models import ContextUnetSprite
from engine import Engine
from utils import *

# checking the availability of GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Define hyperparameters
# =============================================================================
parser = argparse.ArgumentParser(
    description='PyTorch Training for Automatic Image Annotation')
parser.add_argument('--seed', default=20, type=int,
                    help='seed for initializing training')
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--num_workers', default=2, type=int,
                    help='number of data loading workers (default: 2)')
parser.add_argument('--learning-rate', default=0.001, type=float)
parser.add_argument('--context', dest='context', action='store_true')
parser.add_argument('--sampling', dest='sampling', action='store_true')
parser.add_argument('--ddim', dest='ddim', action='store_true')
parser.add_argument(
    '--save_dir', default='./checkpoints/', type=str, help='save path')


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    dataloader = make_dataloader(args)

    n_cfeat = 5
    # height = 16
    model = ContextUnetSprite(in_channels=3,
                              n_feat=64,
                              n_cfeat=n_cfeat)

    engine = Engine(args,
                    model,
                    dataloader,
                    n_cfeat)

    if not args.sampling:
        engine.initialization()
        engine.train_iteration()
    else:
        engine.initialization()
        if args.context:
            path = args.save_dir + "Context_Sprite_" + \
                str(args.epochs) + ".pth"
            print(path)
            engine.load_model(path)
            ctx = torch.tensor([
                # hero, non-hero, food, spell, side-facing
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0.6, 0, 0],
                [0, 0, 0.6, 0.4, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 1, 0, 1]
            ]).float().to(device)
            if args.ddim:
                samples, intermediate = engine.sample_ddim(n_sample=ctx.shape[0],
                                                           context=ctx)
            else:
                samples, intermediate, i_list = engine.sample_ddpm(n_sample=ctx.shape[0],
                                                                   context=ctx)
        else:
            path = args.save_dir + "Sprite_" + str(args.epochs) + ".pth"
            print(path)
            engine.load_model(path)
            if args.ddim:
                samples, intermediate = engine.sample_ddim(n_sample=16)
            else:
                samples, intermediate, i_list = engine.sample_ddpm(n_sample=16)
        #
        if args.ddim:
            for i in range(len(intermediate)):
                grid_imshow(intermediate[i])
        else:
            for i in range(len(intermediate)):
                grid_imshow(intermediate[i], i_list[i])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
