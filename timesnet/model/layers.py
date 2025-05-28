from typing import *
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int,
    ) -> None:
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    tensor=m.weight,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    

class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, :x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int,
    ) -> None:
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(num_embeddings=c_in, embedding_dim=d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        embed_type: str = 'fixed',
        freq: str = 'h',
    ) -> None:
        super().__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        if freq == 't':
            self.minute_embed = Embed(c_in=minute_size, d_model=d_model)
        
        self.hour_embed = Embed(c_in=hour_size, d_model=d_model)
        self.weekday_embed = Embed(c_in=weekday_size, d_model=d_model)
        self.day_embed = Embed(c_in=day_size, d_model=d_model)
        self.month_embed = Embed(c_in=month_size, d_model=d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return minute_x + hour_x + weekday_x + day_x + month_x
    

class TimeFeatureEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        embed_type: str = 'timeF',
        freq: str = 'h',
    ) -> None:
        super().__init__()
        freq_map = {
            'h': 4,
            't': 5,
            's': 6,
            'm': 1,
            'a': 1,
            'w': 2,
            'd': 3,
            'b': 3,
        }
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = 'fixed',
        freq: str = 'h',
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model,
                embed_type=embed_type,
                freq=freq,
            )
        else:
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=d_model,
                embed_type=embed_type,
                freq=freq,
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor,
    ) -> Tensor:
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        
        return self.dropout(x)
