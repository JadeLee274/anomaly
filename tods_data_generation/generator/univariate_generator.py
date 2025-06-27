from typing import *
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
Vector = np.ndarray
Matrix = np.ndarray
DATA_SAVE_DIR = '../datasets/univariate/data'
IMG_SAVE_DIR = '../datasets/univariate/data_img'


class UnivariateDataGenerator:
    def __init__(
        self,
        stream_length: int,
        behavior: Callable[[float], float] = sine,
        behavior_config: Optional[Dict[str, Callable[[float], float]]] = None,
    ) -> None:
        self.stream_length = stream_length
        self.behavior = behavior
        self.behavior_config = behavior_config if behavior_config is not None else {}
        self.data = None
        self.label = None
        self.data_origin = None
        self.timestamp = np.arange(stream_length)
        self.generate_timeseries()

    def generate_timeseries(self) -> None:
        self.behavior_config['length'] = self.stream_length
        self.data = self.behavior(**self.behavior_config)
        self.data_origin = self.data.copy()
        self.label = np.zeros(self.stream_length, dtype=int)
    
        return None
    
    def point_global_anomalies(
        self,
        ratio: float,
        factor: float,
        radius: int,
    ) -> None:
        """
        Add point global anomalies to original data

        Args:
            ratio: what ratio anomalies will be added
            factor: the larger, the anomalies are farther from normal data
            radius: the radius of collective anomalies range
        """
        position = (
            np.random.rand(round(self.stream_length * ratio)) * self.stream_length
        ).astype(int)
        maximum = max(self.data)
        minimum = min(self.data)

        for i in position:
            local_std = self.data_origin[max(0, i - radius): min(self.stream_length, i + radius)].std()
            self.data[i] = self.data_origin[i] * factor * local_std
            if 0 <= self.data[i] < maximum: self.data[i] = maximum
            if 0 > self.data[i] > minimum: self.data[i] = minimum
            self.label[i] = 1
        
        return None
    
    def point_contextual_anomalies(
        self,
        ratio: float,
        factor: float,
        radius: int,
    ) -> None:
        """
        Add point contextual anomalies to original data

        Args:
            ratio: what ratio anomalies will be added
            factor: the larger, the anomalies are farther from normal data
                    Notice: point contextual anomalies will not exceed the range of [min, max] of the original data
            radius: the radius of collective anomalies range
        """
        position = (
            np.random.rand(round(self.stream_length * ratio)) * self.stream_length
        ).astype(int)

        maximum = max(self.data)
        minimum = min(self.data)

        for i in position:
            local_std = self.data_origin[max(0, i - radius): min(self.stream_length, i + radius)].std()
            self.data[i] = self.data_origin[i] * factor * local_std
            if self.data[i] > maximum: self.data[i] = maximum * min(0.95, abs(np.random.normal(0, 0.5)))
            if self.data[i] < maximum: self.data[i] = minimum * min(0.95, abs(np.random.normal(0, 0.5)))
            self.label[i] = 1
        
        return None
    
    def collective_global_anomalies(
        self,
        ratio: float,
        radius: int,
        option: str = 'square',
        coef: float = 3.0,
        noise_amp: float = 0.0,
        level: int = 5,
        freq: float = 0.04,
        offset: float = 0.0,
        base: List[float] = [0., ]
    ) -> None:
        """
        Add collective global anomalies to original data

        Args:
            ratio: what ratio anomalies will be added
            radius: the radius of collective anomalies range
            option: if 'square': 'level' 'freq' and 'offset' are used to generate square sine wave
                    if 'other': 'base' is used to generate outlier shape
            level: how many sine waves will square_wave synthesis
            base: a llist of values that we want to substitute normal data when we generate anomalies
        """
        position = (
            np.random.rand(round(self.stream_length * ratio / (2 * radius))) * self.stream_length
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
                noise_amp=noise_amp
            )
        else:
            sub_data = collective_global_synthetic(
                length=self.stream_length,
                base=base,
                coef=coef,
                noise_amp=noise_amp
            )
        
        for i in position:
            start = max(0, i - radius)
            end = min(self.stream_length, i + radius)
            self.data[start: end] = sub_data[start: end]
            self.label[start: end] = 1

        return None
    
    def collective_trend_anomalies(
        self,
        ratio: float,
        factor: float,
        radius: float,
    ) -> None:
        """
        Add collective trend anomalies to original data

        Args:
            ratio: what ratio anomalies will be added
            factor: the larger, the anomalies are farther from normal data
                    Notice: point contextual anomalies will not exceed the range of [min, max] of the original data
            radius: the radius of collective anomalies range
        """
        position = (
            np.random.rand(round(self.stream_length * ratio / (2 * radius))) * self.stream_length
        ).astype(int)
        
        for i in position:        
            start = max(0, i - radius)
            end = min(self.stream_length, i + radius)        
            slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
            self.data[start: end] = self.data_origin[start: end] + slope
            self.data[end: ] = self.data[end: ] + slope[-1]
            self.label[start: end] = 1
        
        return None
        
    def collective_seasonal_anomalies(
        self,
        ratio: float,
        factor: float,
        radius: float,
    ) -> None:
        """
        Add collective seasonal anomalies to original data

        Args:
            ratio: what ratio anomalies will be added
            factor: the larger, the anomalies are farther from normal data
                    Notice: point contextual anomalies will not exceed the range of [min, max] of the original data
            radius: the radius of collective anomalies range
        """
        position = (
            np.random.rand(round(self.stream_length * ratio / (2 * radius))) * self.stream_length
        ).astype(int)
        seasonal_config = self.behavior_config
        seasonal_config['freq'] = factor * self.behavior_config['freq']

        for i in position:        
            start = max(0, i - radius)
            end = min(self.stream_length, i + radius)
            self.data[start: end] = self.behavior(**seasonal_config)[start: end]
            self.label[start: end] = 1
        
        return None
    

if __name__ == '__main__':
    # np.random.seed(100)

    BEHAVIOR_CONFIG = {
        'freq': 0.04,
        'coef': 1.5,
        'offset': 0.0,
        'noise_amp': 0.05,
    }
    BASE = [
        1.4529900e-01,
        1.2820500e-01,
        9.4017000e-02,
        7.6923000e-02,
        1.1111100e-01,
        1.4529900e-01,
        1.7948700e-01,
        2.1367500e-01,
        2.1367500e-01
    ]

    univariate_data = UnivariateDataGenerator(
        stream_length=400,
        behavior=sine,
        behavior_config=BEHAVIOR_CONFIG,
    )
    univariate_data.collective_global_anomalies(
        ratio=0.05,
        radius=5,
        option='square',
        coef=1.5,
        noise_amp=0.03,
        level=20,
        freq=0.04,
        base=BASE,
        offset=0.0,
    )
    univariate_data.collective_seasonal_anomalies(
        ratio=0.05,
        factor=3,
        radius=5,
    )
    univariate_data.collective_trend_anomalies(
        ratio=0.05,
        factor=0.5,
        radius=5,
    )
    univariate_data.point_global_anomalies(
        ratio=0.05,
        factor=3.5,
        radius=5,
    )
    univariate_data.point_contextual_anomalies(
        ratio=0.05,
        factor=2.5,
        radius=5,
    )

    if not os.path.exists(DATA_SAVE_DIR):
        os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    
    if not os.path.exists(IMG_SAVE_DIR):
        os.makedirs(IMG_SAVE_DIR, exist_ok=True)

    df = pd.DataFrame(
        {
            'value': univariate_data.data,
            'anomaly': univariate_data.label,
        }
    )
    df.to_csv(
        path_or_buf=os.path.join(DATA_SAVE_DIR, f'set_{len(os.listdir(DATA_SAVE_DIR))}.csv'),
        index=False,
    )

    plt.plot(univariate_data.timestamp, univariate_data.data)
    anomaly_idxs = np.where(univariate_data.label == 1)[0]
    anomalies = list(univariate_data.data[anomaly_idxs])
    outcome = series_segmentation(anomaly_idxs)
    
    for anomaly in outcome:
        if len(anomaly) == 1:
            plt.plot(anomaly, univariate_data.data[anomaly], 'ro')
        else:
            if len(anomaly) != 0:
                plt.axvspan(anomaly[0], anomaly[-1], color='red', alpha=0.5)
    
    plt.savefig(os.path.join(IMG_SAVE_DIR, f'set_{len(os.listdir(IMG_SAVE_DIR))}.jpg'))
    plt.close()
