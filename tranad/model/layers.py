from typing import *
import torch
from torch import Tensor
import torch.nn as nn


class Attention(nn.Module):
    def __init__(
        self,
        n_features: int,
    ) -> None:
        super().__init__()
        self.name = 'Attention'
        self.lr = 0.0001
        self.n_features = n_features
        self.n_window = 5
        self.n = self.n_features * self.n_window
        self.atts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.n, n_features ** 2),
                    nn.ReLU()
                ) for _ in range(1)
            ]
        )

    def forward(
        self,
        g: Tensor
    ) -> Tuple[Tensor, Tensor]:
        for at in self.atts:
            ats = at(g.view(-1)).reshape(self.n_features, self.n_features)
            g = torch.matmul(g, ats)
        return g, ats


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        target2 = self.self_attn(target, target, target)[0]
        target = target + self.dropout1(target2)
        target2 = self.multihead_attention(target, memory, memory)[0]
        target = target + self.dropout2(target2)
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target))))
        target = target + self.dropout3(target2)
        return target
    