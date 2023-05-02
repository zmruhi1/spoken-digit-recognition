import argparse


import numpy as np
import torch

# utils
from tqdm import tqdm

from models.sdr import train
import utilities.utils as utils
from utilities.utils import str2bool
from models.sdr import train


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.train: 
        train(args)
    utils.plot_results(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spoken Digit Recognition')

    # Study Case
    parser.add_argument('--model', default='rnn', type=str, help='network name')
    parser.add_argument('--train', default=True, type=str2bool, help='train or test')
    parser.add_argument('--gpu', default=False, type=str2bool, help='GPU acceleration')

    # Dataset Parameters
    parser.add_argument('--data_dir', default='./data/', type=str, help='dataset directory')

    # Training Parameters
    parser.add_argument('--seed', default=1, type=int, help='random seed')   
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=2, type=int, help='training batch size')
    parser.add_argument('--epoch', default=30, type=int, help='training iterations')
    parser.add_argument('--patience', default=5, type=int, help='early stopping patience')

    # Plot and Save options
    parser.add_argument('--listen_sample', default=False, type=str2bool, help='plot sample wave')
    parser.add_argument('--save_dir', default='./data')
    parser.add_argument('--out_dir', default='./outputs')

    args = parser.parse_args()

    main(args)
