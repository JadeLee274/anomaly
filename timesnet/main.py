from typing import *
import os
import argparse
import random
import numpy as np
import torch
import torch.backends
from exp.anomaly_detection import ExpAnomalyDetection
from utils.print_args import print_args


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument(
        '--task-name',
        type=str,
        required=True,
        default='anomaly_detection',
        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]'
    )
    parser.add_argument(
        '--is-training',
        type=int,
        required=True,
        default=1,
        help='status'
    )
    parser.add_argument(
        '--model-id',
        type=str,
        required=True,
        default='test',
        help='model id'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        default='Autoformer',
        help='model name, options: [Autoformer, Transformer, TimesNet]'
    )

    # data loader
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        default='ETTh1',
        help='dataset type'
    )
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/ETT/',
        help='root path of the data file'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='ETTh1.csv',
        help='data file'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='M',
        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='OT',
        help='target feature in S or MS task'
    )
    parser.add_argument(
        '--freq',
        type=str,
        default='h',
        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h'
    )
    parser.add_argument(
        '--checkpoints',
        type=str,
        default='./checkpoints/', 
        help='location of model checkpoints'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=96,
        help='input sequence length'
    )
    parser.add_argument(
        '--label-len',
        type=int,
        default=48,
        help='start token length'
    )
    parser.add_argument(
        '--pred-len',
        type=int,
        default=96,
        help='prediction sequence length'
    )

    # anomaly detection task
    parser.add_argument('--anomaly-ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument(
        '--expand',
        type=int,
        default=2,
        help='expansion factor for Mamba'
    )
    parser.add_argument(
        '--d_conv',
        type=int,
        default=4,
        help='conv kernel size for Mamba'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='for TimesBlock'
    )
    parser.add_argument(
        '--num-kernels',
        type=int,
        default=6,
        help='for Inception')
    parser.add_argument(
        '--enc-in',
        type=int,
        default=7,
        help='encoder input size'
    )
    parser.add_argument(
        '--dec-in',
        type=int,
        default=7,
        help='decoder input size'
    )
    parser.add_argument(
        '--c-out',
        type=int,
        default=7,
        help='output size'
    )
    parser.add_argument(
        '--d-model',
        type=int,
        default=512,
        help='dimension of model'
    )
    parser.add_argument(
        '--n-heads',
        type=int,
        default=8,
        help='num of heads'
    )
    parser.add_argument(
        '--e-layers',
        type=int,
        default=2,
        help='num of encoder layers'
    )
    parser.add_argument(
        '--d-layers',
        type=int,
        default=1,
        help='num of decoder layers'
    )
    parser.add_argument(
        '--d-ff',
        type=int,
        default=2048,
        help='dimension of fcn'
    )
    parser.add_argument(
        '--moving_avg',
        type=int,
        default=25,
        help='window size of moving average'
    )
    parser.add_argument(
        '--factor',
        type=int,
        default=1,
        help='attn factor'
    )
    parser.add_argument(
        '--distil',
        action='store_false',
        help='whether to use distilling in encoder, using this argument means not using distilling',
        default=True
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='dropout'
    )
    parser.add_argument(
        '--embed',
        type=str,
        default='timeF',
        help='time features encoding, options:[timeF, fixed, learned]'
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='gelu',
        help='activation'
    )

    # optimization
    parser.add_argument(
        '--num-workers',
        type=int,
        default=10,
        help='data loader num workers'
    )
    parser.add_argument(
        '--itr',
        type=int,
        default=1,
        help='experiments times'
    )
    parser.add_argument(
        '--train-epochs',
        type=int,
        default=10,
        help='train epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='batch size of train input data'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='early stopping patience'
    )
    parser.add_argument(
        '--learning-rate',
        type=float, 
        default=0.0001, 
        help='optimizer learning rate'
    )
    parser.add_argument(
        '--des',
        type=str,
        default='test',
        help='exp description'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='MSE',
        help='loss function'
    )
    parser.add_argument(
        '--lradj',
        type=str,
        default='type1',
        help='adjust learning rate'
    )
    parser.add_argument(
        '--use-amp',
        action='store_true', 
        help='use automatic mixed precision training', 
        default=False
    )

    # GPU
    parser.add_argument(
        '--use-gpu',
        type=bool,
        default=True,
        help='use gpu'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='gpu'
    )
    parser.add_argument(
        '--gpu-type',
        type=str,
        default='cuda', 
        help='gpu type'
    )
    parser.add_argument(
        '--use-multi-gpu',
        action='store_true',
        help='use multiple gpus',
        default=False
    )
    parser.add_argument(
        '--devices',
        type=str,
        default='0,1,2,3',
        help='device ids of multile gpus'
    )

    # de-stationary projector params
    parser.add_argument(
        '--p-hidden-dims',
        type=int,
        nargs='+',
        default=[128, 128],
        help='hidden layer dimensions of projector (List)'
    )
    parser.add_argument(
        '--p-hidden-layers',
        type=int,
        default=2,
        help='number of hidden layers in projector'
    )

    # metrics (dtw)
    parser.add_argument(
        '--use_dtw',
        type=bool,
        default=False,
        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)'
    )

 
    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'anomaly_detection':
        Exp = ExpAnomalyDetection

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii
            )

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii
        )

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
