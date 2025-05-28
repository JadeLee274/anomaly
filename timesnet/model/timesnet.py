from typing import *
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from .layers import DataEmbedding
from .blocks import TimesBlock


class TimesNet(nn.Module):
    def __init__(
        self,
        configs,
    ) -> None:
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList(
            [TimesBlock(configs) for _ in range(configs.e_layers)]
        )
        self.enc_embedding = DataEmbedding(
            c_in=configs.enc_in,
            d_model=configs.d_model,
            embed_type=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout,
        )
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name == 'ltsf' or self.task_name == 'stsf':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len
            )
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True,
            )
        
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True,
            )
        
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )
    
    def forecast(
            self,
            x_enc: Tensor,
            x_mark_enc: Tensor,
            x_dec: Tensor,
            x_mark_dec: Tensor,
        ) -> Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul((stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add((means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))

        return dec_out

    def imputation(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor,
        x_dec: Tensor,
        x_mark_dec: Tensor,
        mask: Tensor,
    ) -> Tensor:
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul((stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add((means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        
        return dec_out
    
    def anomaly_detection(
        self,
        x_enc: Tensor,
    ) -> Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, None)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        dec_out = self.projection(enc_out)
        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return dec_out
    
    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        return output
    
    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor,
        x_dec: Tensor,
        x_mark_dec: Tensor,
        mask: Optional[Tensor] = None,
    ) -> None:
        if self.task_name == 'ltsf' or self.task_name == 'stsf':
            dec_out = self.forecast(
                x_enc=x_enc,
                x_mark_enc=x_mark_enc,
                x_dec=x_dec,
                x_mark_dec=x_mark_dec,
            )
        
        elif self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc=x_enc,
                x_mark_enc=x_mark_enc,
                x_dec=x_dec,
                x_mark_dec=x_mark_dec,
                mask=mask,
            )
            return dec_out
        
        elif self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        
        elif self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        
        return None
