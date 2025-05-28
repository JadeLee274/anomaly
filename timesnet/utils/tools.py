from typing import *
import os
import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


plt.switch_backend('agg')


def adjust_learning_rate(
    optimizer: torch.optim,
    epoch: int,
    args,
) -> None:
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
    elif args.lradj == 'type3':
        lr_adjust = {
            epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == 'cosine':
        lr_adjust = {
            epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: int = 0,
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        path: str,
    ) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early Stopping Counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(
        self,
        val_loss: float,
        model: nn.Module,
        path: str
    ) -> None:
        if self.verbose:
            print(f'Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def adjustment(
    gt: Tensor,
    pred: Tensor,
) -> Tuple[Tensor, Tensor]:
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        
        if anomaly_state:
            pred[i] = 1
    
    return gt, pred
