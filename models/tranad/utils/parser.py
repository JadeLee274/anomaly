import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    metavar='-d',
    type=str,
    required=False,
    default='synthetic',
)
parser.add_argument(
    '--model',
    metavar='-m',
    type=str,
    required=False,
    default='LSTM-Multivariate',
)
parser.add_argument(
    '--test',
    action='store_true',
)
parser.add_argument(
    '--retrain',
    action='store_true',
)
parser.add_argument(
    '--less',
    action='store_true',
)
args = parser.parse_args()
