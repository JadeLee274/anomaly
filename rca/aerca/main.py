from typing import *
import os
import sys
import logging
import argparse
import warnings
from datasets import linear, lotka_volterra, lorenz96, swat, nonlinear, msds
from args import linear_args, lotka_volterra_args, lorenz96_args, swat_args, msds_args, nonlinear_args
from models import aerca
from utils import utils
warnings.filterwarnings('ignore')


def main(argv: List[str]) -> None:
    """
    Main function to run the AERCA model on a specified dataset

    The script supports multiple datasets. 
    It selects the appropriate dataset class, argument parser, and log file
    based on the provided --dataset_name argument.
    
    If preprocessing_data is set to 1, then the dataset is generated and saved. 
    Otherwise, the existing data is loaded.

    After setting the random seed and loading the data, 
    the AERCA model is instantiated with common parameters 
    and then trained and tested accordingly.

    Args:
        argv (list): List of command-line arguments.
    """
    # Preliminary parsing: retrieve the dataset name.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        '--dataset_name',
        type=str,
        default='linear',
        help='Name of the dataset to run. Options: linear, nonlinear, lotka_volterra, lorenz96, msds, swat'
    )
    pre_args, remaining_args = pre_parser.parse_known_args(argv[1:])
    dataset_name = pre_args.dataset_name.lower()

    # Map dataset named to their configuration: argument parser, dataset class, log file, and slicing flag.
    dataset_mapping = {
        'linear': {
            'args': linear_args.create_arg_parser,
            'dataset_class': linear.Linear,
            'log_file': 'linear.log',
            'use_slice': True,
        },
        'nonlinear': {
            'args': nonlinear_args.create_arg_parser,
            'dataset_class': nonlinear.Nonlinear,
            'log_file': 'nonlinear.log',
            'use_slice': True,
        },
        'lotka_volterra': {
            'args': lotka_volterra_args.create_arg_parser,
            'dataset_class': lotka_volterra.LotkaVolterra,
            'log_file': 'lotka_volterra.log',
            'use_slice': True,
        },
        'lorenz96': {
            'args': lorenz96_args.create_arg_parser,
            'dataset_class': lorenz96.Lorenz96,
            'log_file': 'lorenz96.log',
            'use_slice': True,
        },
        'msds': {
            'args': msds_args.create_arg_parser,
            'dataset_class': msds.MSDS,
            'log_file': 'msds.log',
            'use_slice': False,
        },
        'swat': {
            'args': swat_args.create_arg_parser,
            'dataset_class': swat.SWaT,
            'log_file': 'swat.log',
            'use_slice': False,
        },
    }

    if dataset_name not in dataset_mapping:
        print(f'Dataset {dataset_name} not recognized. Avaliable options are {list(dataset_mapping.keys())}')
        sys.exit(1)
    
    mapping = dataset_mapping[dataset_name]

    # Set up logging: create a logs directory relative to the current directory if it doesn't extist.
    logging_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    log_file_path = os.path.join(logging_dir, mapping['log_file'])
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse the remaining command - line arguments using the dataset-specific argument parser.
    parser = mapping['args']()
    args, unknown = parser.parse_known_args(remaining_args)
    options = vars(args)

    # Set the random seed for reproducibility.
    utils.set_seed(options['seed'])
    print('Set seed: {}.\n'.format(options['seed']))

    # Instantiate the dataset class and generate or load data on the processing flag.
    data_class = mapping['dataset_class'](options)
    if options['preprocessing_data'] == 1:
        print('Preprocessing data: generating and saving new data...\n')
        data_class.generate_example()
        data_class.save_data()
    else:
        print('Loading existing data...\n')
        data_class.load_data()
    
    # Instantiate the AERCA model using the common set of parameters.
    aerca_model = aerca.AERCA(
        num_vars=options['num_vars'],
        hidden_layer_size=options['hidden_layer_size'],
        num_hidden_layers=options['num_hidden_layers'],
        device=options['device'],
        window_size=options['window_size'],
        stride=options['stride'],
        encoder_gamma=options['encoder_gamma'],
        decoder_gamma=options['decoder_gamma'],
        encoder_lambda=options['encoder_lambda'],
        decoder_lambda=options['decoder_lambda'],
        beta=options['beta'],
        lr=options['lr'],
        epochs=options['epochs'],
        recon_threshold=options['recon_threshold'],
        data_name=options['dataset_name'],
        causal_quantile=options['causal_quantile'],
        root_cause_threshold_encoder=options['root_cause_threshold_encoder'],
        root_cause_threshold_decoder=options['root_cause_threshold_decoder'],
        risk=options['risk'],
        initial_level=options['initial_level'],
        num_candidates=options['num_candidates'],
    )

    # Training phase: train the model if enabled.
    if options['training_aerca']:
        if mapping['use_slice']:
            training_data = data_class.data_dict['x_n_list'][:options['training_size']]
        else:
            training_data = data_class.data_dict['x_n_list']
        print('Starting training AERCA model...')
        aerca_model._training(xs=training_data)
        print('Training Done\n')
    
    # Testing phase for causal discovery (applis only if slicing is used).
    if mapping['use_slice']:
        test_causal = data_class.data_dict['x_n_list'][options['training_size']:]
        print('Starting testing AERCA model for causal discovery...')
        aerca_model._testing_causal_discover(
            xs=test_causal,
            causal_struct_value=data_class.data_dict['causal_struct'],
        )
        print('Test for causal discovery done\n')

    # Testing phase for root cause analysis.
    if mapping['use_slice']:
        test_x_ab = data_class.data_dict['x_ab_list'][options['training_size']:]
        test_label = data_class.data_dict['label_list'][options['training_size']:]
    else:
        test_x_ab = data_class.data_dict['x_ab_list']
        test_label = data_class.data_dict['label_list']
    print('Starting testing AERCA model for root cause analysis...')
    aerca_model._testing_root_cause(xs=test_x_ab, labels=test_label)
    print('Test for root cause analysis done\n')

    print('Done')

    return None


if __name__ == '__main__':
    main(sys.argv)
