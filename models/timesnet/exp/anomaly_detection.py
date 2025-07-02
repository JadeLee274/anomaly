from typing import *
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.multiprocessing
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score
from data_factory.processing import data_provider
from exp.exp_basic import ExpBasic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings('ignore')


class ExpAnomalyDetection(ExpBasic):
    def __init__(
        self,
        args,
    ) -> None:
        super(ExpAnomalyDetection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].TimesNet(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model
    
    def _get_data(self, flag: str):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def train(
        self,
        setting,
    ) -> nn.Module:
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)

        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
        )
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                out = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                out = out[:, :, f_dim:]
                loss = criterion(out, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f'\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
            
            print(f'Epoch: {epoch + 1} cost time: {time.time() - epoch_time}')
            train_loss = np.average(train_loss)
            val_loss = self.vali(val_data, val_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f'Epoch {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Val Loss: {val_loss:.7f} Test Loss: {test_loss:.7f}')
            early_stopping(val_loss, self.model, path)

            if early_stopping.early_stop:
                print('Early Stopping')
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(
        self,
        vali_data,
        vali_loader,
        criterion,
    ) -> float:
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                out = self.model(batch_x, None, None, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                out = out[:, :, f_dim:]
                pred = out.detach().cpu()
                true = batch_x.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def test(
        self,
        setting: str,
        test: int = 0,
    ) -> None:
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        if test:
            print('Loading Model...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) statistic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                out = self.model(batch_x, None, None, None)
                score = torch.mean(self.anomaly_criterion(batch_x, out), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []

        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            out = self.model(batch_x, None, None, None)
            score = torch.mean(self.anomaly_criterion(batch_x, out), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print('Threshold: ', threshold)

        # (3) evalutaion on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prfs(gt, pred, average='binary')
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}')

        f = open('result_anomaly_detection.txt', 'a')
        f.write(setting + "  \n")
        f.write(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:4f}')
        f.write('\n')
        f.write('\n')
        f.close()

        return
