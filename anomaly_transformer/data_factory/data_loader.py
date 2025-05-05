from typing import *
import os
import pickle
import random
import collections
import numbers
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
Matrix = np.ndarray


class PSMSegLoader(object):
    def __init__(
        self,
        data_path: str,
        window_size: int,
        step: int,
        mode: str = 'train'
    ) -> None:
        self.mode = mode
        self.step = step
        self.window_size = window_size
        self.scaler = StandardScaler() # for normalization

        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:] # get values except index
        data = np.nan_to_num(data) # relplace nan, inf, -inf to number
        self.scaler.fit(data) # calculate mean and std of data
        self.train = self.scaler.transform(data) # normalize data
        
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data) # normalize without nan_to_num

        self.val = self.test

        self.test_labels = pd.read_csv(
            data_path + '/test_label.csv'
        ).values[:, 1:]

        print('train:', self.train.shape)
        print('test:', self.test.shape)

    def __len__(self) -> int:
        """
        Number of images in the object dataset
        """
        if self.mode == 'train':
            return (self.train.shape[0] - self.window_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.window_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.window_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.window_size) // self.window_size + 1
        
    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Matrix, Matrix]:
        index = index * self.step
        if self.mode == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[:self.window_size])
            )
        elif self.mode == 'val':
            return (
                np.float32(self.val[index: index + self.win_size]),
                np.float32(self.test_labels[:self.window_size])
            )
        elif self.mode == 'test':
            return (
                np.float32(self.test[index: index + self.window_size]),
                np.float32(self.test_labels[index: index + self.window_size])
            )
        else:
            return (
                np.float32(
                    self.test[
                        index // self.step * self.window_size:
                        index // self.step * self.window_size + self.window_size
                    ]
                ),
                np.float32(
                    self.test_labels[
                        index // self.step * self.window_size: 
                        index // self.step * self.window_size + self.window_size
                    ]
                )
            )
        

class MSLSegLoader(object):
    def __init__(
        self,
        data_path: str,
        window_size: int,
        step: int,
        mode: str = 'train',
    ) -> None:
        self.mode = mode
        self.step = step
        self.window_size = window_size
        self.scaler = StandardScaler()
        
        data = np.load(data_path + '/MSL_train.npy')
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)
        
        test_data = np.load(data_path + '/MSL_test.npy')
        self.test = self.scaler.transform(test_data)

        self.val = self.test
        self.test_labels = np.load(data_path + '/MSL_test_label.npy')

        print('train:', self.train.shape)
        print('test:', self.test.shape)

    def __len__(self):
        if self.mode == 'train':
            return (self.train.shape[0] - self.window_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.window_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.window_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.window_size) // self.window_size + 1
        
    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Matrix, Matrix]:
        if self.mode == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[: self.window_size])
            )
        elif self.mode == 'val':
            return (
                np.float32(self.val[index: index + self.window_size]),
                np.float32(self.test_labels[: self.window_size])
            )
        elif self.mode == 'test':
            return (
                np.float32(self.test[index: index + self.window_size]),
                np.float32(self.test_labels[index: index + self.window_size])
            )
        else:
            return (
                np.float32(
                    self.test[
                        index // self.step * self.window_size:
                        index // self.step * self.window_size + self.window_size
                    ]
                ),
                np.float32(
                    self.test_labels[
                        index // self.step * self.window_size:
                        index // self.step * self.window_size + self.window_size
                    ]
                )
            )
        

class SMAPSegLoader(object):
    def __init__(
        self,
        data_path: str,
        window_size: int,
        step: int,
        mode: str = 'train',
    ) -> None:
        self.mode = mode
        self.step = step
        self.window_size = window_size
        self.scaler = StandardScaler()

        data = np.load(data_path + '/SMAP_train.npy')
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)

        test_data = np.load(data_path + '/SMAP_test.npy')
        self.test = self.scaler.transform(test_data)

        self.val = self.test
        self.test_labels = np.load(data_path + '/SMAP_test_label.npy')

        print('train:', self.train.shape)
        print('test:', self.test.shape)

    def __len__(self) -> int:
        if self.mode == 'train':
            return (self.train.shape[0] - self.window_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.window_size) // self.step + 1
        elif self.mode == 'train':
            return (self.test.shape[0] - self.window_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.window_size) // self.window_size + 1
        
    def __getitem__(
        self,
        index: int
    ) -> Tuple[Matrix, Matrix]:
        if self.mode == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[: self.window_size])
            )
        elif self.mode == 'val':
            return (
                np.float32(self.val[index: index + self.window_size]),
                np.float32(self.test_labels[: self.window_size])
            )
        elif self.mode == 'test':
            return (
                np.float32(self.test[index: index + self.window_size]),
                np.float32(self.test_labels[: self.window_size])
            )
        else:
            return (
                np.float32(
                    self.test[
                        index // self.step * self.window_size: 
                        index // self.step * self.window_size + self.window_size
                    ]
                ),
                np.float32(
                    self.test_labels[
                        index // self.step * self.window_size:
                        index // self.step * self.window_size + self.window_size
                    ]
                )
            )
        


class SMDSegLoader(object):
    def __init__(
        self,
        data_path: str,
        window_size: int,
        step: int,
        mode: str = 'train',
    ) -> None:
        self.mode = mode
        self.step = step
        self.window_size = window_size
        self.scaler = StandardScaler()

        data = np.load(data_path + '/SMD_train.npy')
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)

        test_data = np.load(data_path + '/SMD_test.npy')
        self.test = self.scaler.transform(test_data)

        self.val = self.test
        self.test_labels = np.load(data_path + '/SMD_test_label.npy')

        print('train:', self.train.shape)
        print('test:', self.test.shape)

    def __len__(self) -> int:
        if self.mode == 'train':
            return (self.train.shape[0] - self.window_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.window_size) // self.step + 1
        elif self.mode == 'train':
            return (self.test.shape[0] - self.window_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.window_size) // self.window_size + 1
        
    def __getitem__(
        self,
        index: int
    ) -> Tuple[Matrix, Matrix]:
        if self.mode == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[: self.window_size])
            )
        elif self.mode == 'val':
            return (
                np.float32(self.val[index: index + self.window_size]),
                np.float32(self.test_labels[: self.window_size])
            )
        elif self.mode == 'test':
            return (
                np.float32(self.test[index: index + self.window_size]),
                np.float32(self.test_labels[: self.window_size])
            )
        else:
            return (
                np.float32(
                    self.test[
                        index // self.step * self.window_size: 
                        index // self.step * self.window_size + self.window_size
                    ]
                ),
                np.float32(
                    self.test_labels[
                        index // self.step * self.window_size:
                        index // self.step * self.window_size + self.window_size
                    ]
                )
            )
        

def get_loader_segment(
    data_path: str,
    batch_size: int,
    window_size: int = 100,
    step: int = 100,
    mode: str = 'train',
    dataset: str = 'KDD',
) -> DataLoader:
    if dataset == 'SMD':
        dataset = SMDSegLoader(data_path, window_size, step, mode)
    elif dataset == 'MSL':
        dataset = MSLSegLoader(data_path, window_size, 1, mode)
    elif dataset == 'SMAP':
        dataset = SMAPSegLoader(data_path, window_size, 1, mode)
    elif dataset == 'PSM':
        dataset = PSMSegLoader(data_path, window_size, 1, mode)
    
    shuffle = False
    if mode == 'train':
        shuffle = True
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )

    return data_loader
