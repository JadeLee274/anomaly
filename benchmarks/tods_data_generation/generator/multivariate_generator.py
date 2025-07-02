from typing import *
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *


class MultivariateDataGenerator:
    def __init__(
        self,
        dim: int,
        stream_length: int,
        behavior: List[Callable[[int, Vector, float, float], Vector]],
        behavior_config: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        self.dim = dim
        self.stream_length = stream_length
        self.behavior = behavior if behavior is not None else [sine] * dim
        self.behavior_config = behavior_config if behavior_config is not None else [{}] * dim
        self.data = np.empty(shape=[0, stream_length], dtype=float)
        self.label = None
        self.data_origin = None
        self.timestamp = np.arange(stream_length)
        self.generate_timeseries()

    def generate_timeseries(self) -> None:
        for i in range(self.dim):
            self.behavior_config[i]['length'] = self.stream_length
            self.data = np.append(
                arr=self.data,
                values=[self.behavior[i](**self.behavior_config[i])],
                axis=0,
            )
        self.data_origin = self.data.copy()
        self.label = np.zeros(self.stream_length, dtype=int)
        return None
    
    def point_global_anomalies(
        self,
        dim_no: int,
        ratio: float,
        factor: float,
        radius: int,
    ) -> None:
        """
        Add point global anomalies to original data

        Args:
            dim_no: anomaly is added to this feature
            ratio: what ratio anomalies will be added
            factor: the larger, the anomalies are farther from normal data
            radius: the radirs of collective anomalies range
        """
        position = (
            np.random.rand(round(self.stream_length * ratio)) * self.stream_length
        ).astype(int)
        maximum = max(self.data[dim_no])
        minimum = min(self.data[dim_no])

        for i in position:
            local_std = self.data_origin[dim_no][max(0, i - radius): min(i + radius, self.stream_length)].std()
            self.data[dim_no][i] = self.data_origin[dim_no][i] *factor * local_std
            
            if 0 <= self.data[dim_no][i] < maximum: self.data[dim_no][i] = maximum
            if 0 > self.data[dim_no][i] > minimum: self.data[dim_no][i] = maximum
            
            self.label[i] = 1
        
        return None
    
    def point_contextual_anomalies(
        self,
        dim_no: int,
        ratio: float,
        factor: float,
        radius: int,
    ) -> None:
        """
        Add collective global anomalies to original data

        Args:
            dim_no: anomaly is added to this feature
            ratio: what ratio anomalies will be added
            factor: the larger, the anomalies are farther from normal data.
                    Notice: point contextual anomalies will not exceed the range of [min, max] of original data
            radius: the radius of collective anomalies range
        """
        position = (
            np.random.rand(round(self.stream_length * ratio)) * self.stream_length
        ).astype(int)
        maximum = max(self.data[dim_no])
        minimum = min(self.data[dim_no])

        for i in position:
            local_std = self.data_origin[dim_no][max(0, i - radius): min(i + radius, self.stream_length)].std()
            self.data[dim_no][i] = self.data_origin[dim_no][i] * factor * local_std
            
            if self.data[dim_no][i] > maximum: self.data[dim_no][i] = maximum * min(0.95, abs(np.random.normal(0, 1)))
            if self.data[dim_no][i] < minimum: self.data[dim_no][i] = minimum * min(0.95, abs(np.random.normal(0, 1)))

            self.label[i] = 1
        
        return None
    
    def collective_global_anomalies(
        self,
        dim_no: int,
        ratio: float,
        radius: int,
        option: str = 'square',
        coef: int = 3,
        noise_amp: float = 0.0,
        level: int = 5,
        freq: float = 0.04,
        offset: float = 0.0,
        base: List[float] = [0., ]
    ) -> None:
        """
        Add collective global anomalies to original data

        Args:
            dim_no: anomaly is added to this feature
            ratio: what ratio anomalies will be added
            radius: the radius of collective anomalies range
            option: if 'square': 'level' 'freq' and 'offset' are used to generate square sine wave
                    if 'other': 'base' is used to generate outlier shape
            level: how many sine waves will square_wave synthesis
            base: a list of values that we want to substitute normal data when we generate anomalies
        """
        position = (
            np.random.rand(round(self.stream_length * ratio)) * self.stream_length
        ).astype(int)
        valid_option = {'square', 'other'}

        if option not in valid_option:
            raise ValueError("'option' must be one of %r" % valid_option)
        
        if option == 'square':
            sub_data = square_sine(
                level=level,
                length=self.stream_length,
                freq=freq,
                coef=coef,
                offset=offset,
                noise_amp=noise_amp,
            )
        else:
            sub_data = collective_global_synthetic(
                length=self.stream_length,
                base=base,
                coef=coef,
                noise_amp=noise_amp,
            )
        
        for i in position:
            start = max(0, i - radius)
            end = min(self.stream_length, i + radius)
            self.data[dim_no][start: end] = sub_data[start: end]
            self.label[start: end] = 1
        
        return None
    
    def collective_trend_anomalies(
        self,
        dim_no: int,
        ratio: float,
        factor: float,
        radius: int,
    ) -> None:
        """
        Add collective trend anomalies to original data

        Args:
            dim_no: anomaly is added to this feature
            ratio: what ratio anomalies will be added
            factor: how dramatic will the trend be
            radius: the radius of collective anomalies range
        """
        position = (
            np.random.rand(round(self.stream_length * ratio)) * self.stream_length
        ).astype(int)

        for i in position:
            start = max(0, i - radius)
            end = min(self.stream_length, i + radius)
            slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
            self.data[dim_no][start: end] = self.data_origin[dim_no][start: end] + slope
            self.data[dim_no][end: ] = self.data[dim_no][end: ] + slope[-1]
            self.label[start: end] = 1

        return None
    
    def collective_seasonal_anomalies(
        self,
        dim_no: int,
        ratio: float,
        factor: float,
        radius: int,
    ) -> None:
        """
        Add collective seasonal anomalies to original data

        Args:
            dim_no: anomaly is added to this feature
            ratio: what ratio anomalies will be added
            factor: how many times will frequency multiple
            radius: the radius of collective anomalies range
        """
        position = (
            np.random.rand(round(self.stream_length * ratio / (2 * radius))) *self.stream_length
        ).astype(int)
        seasonal_config = self.behavior_config[dim_no]
        seasonal_config['freq'] = factor * self.behavior_config[dim_no]['freq']

        for i in position:
            start = max(0, i + radius)
            end = min(self.stream_length, i + radius)
            self.data[dim_no][start: end] = self.behavior[dim_no](**seasonal_config)[start: end]
            self.label[start: end] = 1

        return None
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Set seed. Default None'
    )
    # Whether to give one anomaly per feature or not. Default False.
    # If true, the generation process follows the original code/paper.
    args.add_argument(
        '--separate_anomaly_types',
        type=bool,
        default=False,
        help='Separate anomalies or not. If True, num_features is restricted to 5; if false, the choice of arguments are free, but choose the anomaly_ratio carefully.'
    )
    args.add_argument(
        '--stream_length',
        type=int,
        default=400,
        help='Length of the series'
    )
    args.add_argument(
        '--num_features',
        type=int,
        default=5,
        help='Dimension of the series'
    )
    args.add_argument(
        '--anomaly_ratio',
        type=float,
        default=0.05,
        help='Total ratio of anomalies'
    )
    args = args.parse_args()

    if args.seed:
        np.random.seed(args.seed)
    
    if args.separate_anomaly_types:
        DATA_SAVE_DIR = f'../datasets/multivariate/anomalies_separated/len_{args.stream_length}/data'
        IMG_SAVE_DIR = f'../datasets/multivariate/anomalies_separated/len_{args.stream_length}/data_img'
    else:
        DATA_SAVE_DIR = f'../datasets/multivariate/anomalies_mixed/dim_{args.num_features}/len_{args.stream_length}/data'
        IMG_SAVE_DIR = f'../datasets/multivariate/anomalies_mixed/dim_{args.num_features}/len_{args.stream_length}/data_img'

    ANO_RATIO = args.anomaly_ratio / 5

    if not os.path.exists(DATA_SAVE_DIR):
        os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    
    if not os.path.exists(IMG_SAVE_DIR):
        os.makedirs(IMG_SAVE_DIR, exist_ok=True)

    # The case when we want to give only one type of anomaly per dimension.
    # In this case, the number of dimension is restricted to 5.
    # This is the original code following the paper. 
    if args.separate_anomaly_types:

        BEHAVIOR = [sine, cosine, sine, cosine, sine]
        BEHAVIOR_CONFIG = [
            {'freq': 0.04, 'coef': 1.5, 'offset': 0.0, 'noise_amp': 0.05},
            {'freq': 0.04, 'coef': 2.5, 'offset': 0.0, 'noise_amp': 0.05},
            {'freq': 0.04, 'coef': 1.5, 'offset': 0.0, 'noise_amp': 0.05},
            {'freq': 0.04, 'coef': 2.5, 'offset': 2.0, 'noise_amp': 0.05},
            {'freq': 0.04, 'coef': 1.5, 'offset': -2.0, 'noise_amp': 0.05},
        ]

        multivariate_data = MultivariateDataGenerator(
            dim=args.num_features,
            stream_length=args.stream_length,
            behavior=BEHAVIOR,
            behavior_config=BEHAVIOR_CONFIG,
            )
        
        multivariate_data.point_global_anomalies(
            dim_no=0,
            ratio=ANO_RATIO,
            factor=3.5,
            radius=5,
            )
        multivariate_data.point_contextual_anomalies(
            dim_no=1,
            ratio=ANO_RATIO,
            factor=2.5,
            radius=5,
        )
        multivariate_data.collective_global_anomalies(
            dim_no=2,
            ratio=ANO_RATIO,
            radius=5,
            option='square',
            coef=1.5,
            noise_amp=0.03,
            level=20,
            freq=0.04,
            offset=0.0,
        )
        multivariate_data.collective_seasonal_anomalies(
            dim_no=3,
            ratio=ANO_RATIO,
            factor=3,
            radius=5,
        )
        multivariate_data.collective_trend_anomalies(
            dim_no=4,
            ratio=ANO_RATIO,
            factor=0.5,
            radius=5,
        )

        df = pd.DataFrame(
            {
                'col_0': multivariate_data.data[0],
                'col_1': multivariate_data.data[1],
                'col_2': multivariate_data.data[2],
                'col_3': multivariate_data.data[3],
                'col_4': multivariate_data.data[4],
                'anomaly': multivariate_data.label,
            }
        )

        df.to_csv(
            path_or_buf=os.path.join(
                DATA_SAVE_DIR,
                f'set_{len(os.listdir(DATA_SAVE_DIR))}.csv'
            ),
            index=False,
        )
        ano_ratio = len(df[df['anomaly'] == 1]) / len(df['anomaly'])
        print(f'Dataset generated: num_features {args.num_features} | length {args.stream_length} | anomaly ratio {ano_ratio}')

        plt.figure(figsize=(10, 15))

        for i in range(1, 6):
            plt.subplot(510 + i)
            plt.plot(multivariate_data.timestamp, multivariate_data.data[i - 1])
            plt.title(f'{i}-th feature')

        plt.suptitle(f'num_features {args.num_features} | length {args.stream_length} | anomaly ratio {ano_ratio}')
        plt.tight_layout()
        plt.savefig(os.path.join(
            IMG_SAVE_DIR,
            f'set_{len(os.listdir(IMG_SAVE_DIR))}.jpg')
        )
        plt.close()
    
    # The case we don't want to separate anomaly types.
    # That is, when we want to give more variations to the dataset.
    # Also in this case, you can customize the dimension of data.
    else:
        BEHAVIOR = []
        BEHAVIOR_CONFIG = []

        for i in range(args.num_features):
            BEHAVIOR.append(np.random.choice([sine, cosine]))
            coef = np.random.choice([1.5, 2.5])
            offset = np.random.choice([-2.0, 0.0, 2.0])
            BEHAVIOR_CONFIG.append(
                {'freq': 0.04, 'coef': coef, 'offset': offset, 'noise_amp': 0.05}
            )

        multivariate_data = MultivariateDataGenerator(
            dim=args.num_features,
            stream_length=args.stream_length,
            behavior=BEHAVIOR,
            behavior_config=BEHAVIOR_CONFIG,
        )

        df = pd.DataFrame({})
        for i in range(args.num_features):
            multivariate_data.point_global_anomalies(
                dim_no=i,
                ratio=args.anomaly_ratio,
                factor=3.5,
                radius=5,
            )
            multivariate_data.point_contextual_anomalies(
                dim_no=i,
                ratio=args.anomaly_ratio,
                factor=2.5,
                radius=5,
            )
            multivariate_data.collective_global_anomalies(
                dim_no=i,
                ratio=args.anomaly_ratio,
                radius=5,
                option='square',
                coef=1.5,
                noise_amp=0.03,
                level=20,
                freq=0.04,
                offset=0.0,
            )
            multivariate_data.collective_seasonal_anomalies(
                dim_no=i,
                ratio=args.anomaly_ratio,
                factor=3,
                radius=5,
            )
            multivariate_data.collective_trend_anomalies(
                dim_no=i,
                ratio=args.anomaly_ratio,
                factor=0.5,
                radius=5,
            )
            df[f'col_{i}'] = multivariate_data.data[i]
        
        df['anomaly'] = multivariate_data.label

        df.to_csv(
            path_or_buf=os.path.join(
                DATA_SAVE_DIR,
                f'set_{len(os.listdir(DATA_SAVE_DIR))}.csv'
            ),
            index=False,
        )

        ano_ratio = len(df[df['anomaly'] == 1]) / len(df['anomaly'])
        print(f'Dataset generated: num_features {args.num_features} | length {args.stream_length} | anomaly ratio {ano_ratio}')

        plt.figure(figsize=(10, 7 * args.num_features))

        for i in range(1, args.num_features + 1):
            plt.subplot(args.num_features, 1, i)
            plt.plot(multivariate_data.timestamp, multivariate_data.data[i - 1])
            plt.title(f'{i}-th feature')

        plt.suptitle(f'num_features {args.num_features} | length {args.stream_length} | anomaly ratio {ano_ratio}')
        plt.savefig(
            os.path.join(IMG_SAVE_DIR, f'set_{len(os.listdir(IMG_SAVE_DIR))}.jpg')
        )
        plt.close()
