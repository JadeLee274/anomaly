from typing import *
import os
import warnings
import argparse
from torch.backends import cudnn
from utils.utils import *
from solver import Solver
warnings.filterwarnings('ignore')


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--k', type=int, default=3, help='Lambda for Assiciation Discrepancy')
    parser.add_argument('--window_size', type=int, default=100, help='Window size')
    parser.add_argument('--input_c', type=int, default=38, help='Dataset dimension')
    parser.add_argument('--output_c', type=int, default=38, help='Output dimension. Set equal to input_c')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--pretrained_model', type=Optional[str], default=None, help='Pre-trained model path')
    parser.add_argument('--dataset', type=str, default='credit', help='Dataset naml')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Choose train/test')
    parser.add_argument('--data_path', type=str, default='../data/credit.csv', help='Dataset path. Ensure that the train/test data are in the same directory.')
    parser.add_argument('--model_save_path', type=str, default='/data/home/tmdals274/anomaly/save/anotran', help='Model save path')
    parser.add_argument('--anomaly_ratio', type=float, default=4.00, help='Dataset anomaly ratio')
    parser.add_argument('--use_point_adjustment', type=str2bool, default=True, help='Use/Not use point adjustment algorithm')
    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
