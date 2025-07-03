from typing import *
import os
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
Vector = np.ndarray
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

        data = pd.read_csv(os.path.join(data_path, 'train.csv'))
        data = data.values[:, 1:] # get values except index
        data = np.nan_to_num(data) # relplace nan, inf, -inf to number
        self.scaler.fit(data) # calculate mean and std of data
        self.train = self.scaler.transform(data) # normalize data
        
        test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data) # normalize without nan_to_num

        self.val = self.test

        self.test_labels = pd.read_csv(os.path.join(data_path, 'test_label.csv')).values[:, 1:]

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
    ) -> Tuple[Matrix, Vector]:
        index = index * self.step
        if self.mode == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
            )
        elif self.mode == 'val':
            return (
                np.float32(self.val[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
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
        
        data = np.load(os.path.join(data_path, 'MSL_train.npy'))
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)
        
        test_data = np.load(os.path.join(data_path, 'MSL_test.npy'))
        self.test = self.scaler.transform(test_data)

        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, 'MSL_test_label.npy'))

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
    ) -> Tuple[Matrix, Vector]:
        index = index * self.step
        if self.mode == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
            )
        # Tha label attatched to train set is a dummy label. See solver.py for more details.
        elif self.mode == 'val':
            return (
                np.float32(self.val[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
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

        data = np.load(os.path.join(data_path, 'SMAP_train.npy'))
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)

        test_data = np.load(os.path.join(data_path, 'SMAP_test.npy'))
        self.test = self.scaler.transform(test_data)

        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, 'SMAP_test_label.npy'))

        print('train:', self.train.shape)
        print('test:', self.test.shape)

    def __len__(self) -> int:
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
        index: int
    ) -> Tuple[Matrix, Vector]:
        index = index * self.step
        if self.mode == 'train':
            return (
                np.float32(self.train[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
            )
        elif self.mode == 'val':
            return (
                np.float32(self.val[index: index + self.window_size]),
                np.float32(self.test_labels[0: self.window_size])
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
        

class SWaTSegLoader(object):
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
        self.train_scaler = StandardScaler()
        self.test_scaler = StandardScaler()

        train_csv_path = os.path.join(data_path, 'SWaT_Normal.csv')
        test_csv_path = os.path.join(data_path, 'SWaT_Abnormal.csv')

        # The following annoated codes are for converting .xlsx into .csv.
        # As SWaT data is given as .xlsx file, I added the code for convenience.
        # Using these parts only once is sufficient. 

        # train_data = pd.read_excel(
        #     os.path.join(data_path, 'SWaT_Dataset_Normal_v1.xlsx'),
        #     header=1,
        # )
        # train_data.to_csv(train_csv_path)
        train_data = pd.read_csv(train_csv_path)
        train_data.drop(columns=[' Timestamp', 'Normal/Attack'], inplace=True)
        train_data = train_data.values[:, :-1]
        self.train_scaler.fit(train_data)
        self.train = self.train_scaler.transform(train_data)

        # test_data = pd.read_excel(
        #     os.path.join(data_path, 'SWaT_Dataset_Attack_v0.xlsx'),
        #     engine='openpyxl',
        #     header=1,
        # )
        # test_data.to_csv(test_csv_path)
        test_data = pd.read_csv(test_csv_path)
        test_data.drop(columns=[' Timestamp'], inplace=True)
        test_data = test_data.values[:, 1:]
        test_labels = test_data[:, -1]
        test_labels = np.where(test_labels == 'Normal', 0, 1)
        self.test_labels = test_labels
        test_data = test_data[:, :-1]
        self.test_scaler.fit(test_data)
        self.test = self.test_scaler.transform(test_data)
        self.val = self.test

        ano_ratio = (test_labels[test_labels == 1]).sum() / len(test_labels)

        print(f'Train Size: {len(self.train)}')
        print(f'Test Size: {len(self.test)}') 
        print(f'Anomaly Ratio: {(100 * ano_ratio):.2f}%\n')
    
    def __len__(self) -> int:
        if self.mode == 'train':
            return (self.train.shape[0] - self.window_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.window_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.window_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.window_size) // self.window_size + 1
        
    def __getitem__(self, index: int) -> Tuple[Matrix, Vector]:
        index = index * self.step
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

        data = np.load(os.path.join(data_path + '/SMD_train.npy'))
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
        elif self.mode == 'test':
            return (self.test.shape[0] - self.window_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.window_size) // self.window_size + 1
        
    def __getitem__(self, index: int) -> Tuple[Matrix, Vector]:
        index = index * self.step
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
        

class CreditSegLoader(object):
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

        data = pd.read_csv(os.path.join(data_path, 'creditcard.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        train_idx = round(0.8 * len(data))

        train_data = data[:train_idx]
        test_data = data[train_idx:]
        train_data = train_data[train_data[:, -1] != 1] # elimanate abnormalities in train data

        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data[:, :-1] # separate labels of test data
        test_labels = test_data[:, -1]

        # put back each label value
        self.test_labels = np.where(test_labels > 0, 1, 0)
        self.test = test_data[:, :-1]
        self.val = self.test

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
        index: int,
    ) -> Tuple[Matrix, Matrix]:
        index = index * self.step
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
    elif dataset == 'SWaT':
        dataset = SWaTSegLoader(data_path, window_size, 1, mode)
    elif dataset == 'Credit':
        dataset = CreditSegLoader(data_path, window_size, 1, mode)
    
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
