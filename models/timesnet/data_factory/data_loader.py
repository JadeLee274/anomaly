from typing import *
import os
import warnings
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_from_tsfile_to_dataframe
from utils.augmentation import run_augmentation_single
Vector = np.ndarray
Matrix = np.ndarray

warnings.filterwarnings('ignore')


class PSMSegLoader(Dataset):
    def __init__(
        self,
        args,
        root_path: str,
        window_size: int,
        step: int = 1,
        flag: str = 'train'
    ) -> None:
        self.flag = flag
        self.step = step
        self.window_size = window_size
        self.scaler = StandardScaler() # for normalization

        train_data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        train_data = train_data.values[:, 1:] # get values except index
        train_data = np.nan_to_num(train_data) # relplace nan, inf, -inf to number
        self.scaler.fit(train_data) # calculate mean and std of data
        train_data = self.scaler.transform(train_data) # normalize data
        data_len = len(train_data)

        self.train = train_data[:int(data_len * 0.8)]
        self.val = train_data[int(data_len * 0.8):]
        
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data) # normalize without nan_to_num

        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]

        print('train:', self.train.shape)
        print('test:', self.test.shape)

    def __len__(self) -> int:
        """
        Number of images in the object dataset
        """
        if self.flag == 'train':
            return (self.train.shape[0] - self.window_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.window_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.window_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.window_size) // self.window_size + 1
        
    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Matrix, Matrix]:
        index = index * self.step
        if self.flag == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
            )
        elif self.flag == 'val':
            return (
                np.float32(self.val[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
            )
        elif self.flag == 'test':
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
        

class MSLSegLoader(Dataset):
    def __init__(
        self,
        args,
        root_path: str,
        window_size: int,
        step: int = 1,
        flag: str = 'train',
    ) -> None:
        self.flag = flag
        self.step = step
        self.window_size = window_size
        self.scaler = StandardScaler()
        
        train_data = np.load(root_path + '/MSL_train.npy')
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        data_len = len(train_data)
        
        self.train = train_data[:int(data_len * 0.8)]
        self.val = train_data[int(data_len * 0.8):]
        
        test_data = np.load(os.path.join(root_path, 'MSL_test.npy'))
        self.test = self.scaler.transform(test_data)

        self.test_labels = np.load(os.path.join(root_path, 'MSL_test_label.npy'))

        print('train:', self.train.shape)
        print('test:', self.test.shape)

    def __len__(self):
        if self.flag == 'train':
            return (self.train.shape[0] - self.window_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.window_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.window_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.window_size) // self.window_size + 1
        
    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Matrix, Matrix]:
        index = index * self.step
        if self.flag == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
            )
        elif self.flag == 'val':
            return (
                np.float32(self.val[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
            )
        elif self.flag == 'test':
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
        

class SMAPSegLoader(Dataset):
    def __init__(
        self,
        args,
        root_path: str,
        window_size: int,
        step: int = 1,
        flag: str = 'train',
    ) -> None:
        self.flag = flag
        self.step = step
        self.window_size = window_size
        self.scaler = StandardScaler()

        train_data = np.load(os.path.join(root_path, 'SMAP_train.npy'))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        data_len = len(train_data)

        self.train = train_data[:int(data_len * 0.8)]
        self.val = train_data[int(data_len * 0.8):]

        test_data = np.load(os.path.join(root_path, 'SMAP_test.npy'))
        self.test = self.scaler.transform(test_data)

        self.test_labels = np.load(os.path.join(root_path, 'SMAP_test_label.npy'))

        print('train:', self.train.shape)
        print('test:', self.test.shape)

    def __len__(self) -> int:
        if self.flag == 'train':
            return (self.train.shape[0] - self.window_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.window_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.window_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.window_size) // self.window_size + 1
        
    def __getitem__(
        self,
        index: int
    ) -> Tuple[Matrix, Matrix]:
        index = index * self.step
        if self.flag == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
            )
        elif self.flag == 'val':
            return (
                np.float32(self.val[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
            )
        elif self.flag == 'test':
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
        

class SMDSegLoader(Dataset):
    def __init__(
        self,
        args,
        root_path: str,
        window_size: int,
        step: int = 100,
        flag: str = 'train',
    ) -> None:
        self.flag = flag
        self.step = step
        self.window_size = window_size
        self.scaler = StandardScaler()

        train_data = np.load(os.path.join(root_path, 'SMD_train.npy'))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        data_len = len(train_data)

        self.train = train_data[:int(data_len * 0.8)]
        self.val = train_data[int(data_len * 0.8):]

        test_data = np.load(os.path.join(root_path, 'SMD_test.npy'))
        self.test = self.scaler.transform(test_data)

        self.test_labels = np.load(os.path.join(root_path, 'SMD_test_label.npy'))

        print('train:', self.train.shape)
        print('test:', self.test.shape)

    def __len__(self) -> int:
        if self.flag == 'train':
            return (self.train.shape[0] - self.window_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.window_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.window_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.window_size) // self.window_size + 1
        
    def __getitem__(
        self,
        index: int
    ) -> Tuple[Matrix, Matrix]:
        index = index * self.step
        if self.flag == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[: self.window_size])
            )
        elif self.flag == 'val':
            return (
                np.float32(self.val[index: index + self.window_size]),
                np.float32(self.test_labels[: self.window_size])
            )
        elif self.flag == 'test':
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
        

class SWATSegLoader(Dataset):
    def __init__(
        self,
        args,
        root_path: str,
        window_size: int,
        step: int = 1,
        flag: str = 'train',
    ) -> None:
        self.flag = flag
        self.step = step
        self.window_size = window_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        train_data = train_data.values[:, :-1]
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        data_len = len(train_data)

        self.train = train_data[:int(data_len * 0.8)]
        self.val = train_data[int(data_len * 0.8):]

        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        test_data = test_data.values[:, :-1]
        test_data = self.scaler.transform(test_data)

        self.test = test_data
        self.test_labels = labels

        print('train: ', self.train.shape)
        print('test: ', self.test.shape)
    
    def __len__(self) -> int:
        if self.flag == 'train':
            return (self.train.shape[0] - self.window_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.window_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.window_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.window_size) // self.window_size + 1
    
    def __getitem__(
        self,
        index: int
    ) -> Tuple[Matrix, Matrix]:
        index = index * self.step
        if self.flag == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
            )
        elif self.flag == 'val':
            return (
                np.float32(self.val[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
            )
        elif self.flag == 'test':
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
