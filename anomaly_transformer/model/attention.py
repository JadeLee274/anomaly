from typing import *
import os
import numpy as np
import torch
import torch.nn as nn
import math
from math import sqrt
Tensor = torch.Tensor


class TriangularCausalMask():
    def __init__(
        self,
        B: int,
        L: int,
        device: str = 'cpu'
    ) -> None:
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                input=torch.ones(mask_shape, dtype=torch.bool),
                diagonal=1
            ).to(device)
        
    @property
    def mask(self):
        return self._mask
        

class AnomalyAttention(nn.Module):
    def __init__(
        self,
        window_size: int,
        mask_flag: bool = True,
        scale = None,
        attention_dropout: float = 0.0,
        output_attention: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)
        
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        sigma: Tensor,
        attention_mask: Tensor,
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]
    ]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attention_mask is None:
                attention_mask = TriangularCausalMask(
                    B, L, device=queries.device
                )
            scores.masked_fill_(
                mask=attention_mask.mask,
                value=-np.inf,
            )
        attention = scale * scores

        sigma = sigma.transpose(1, 2) # B L H -> B H L
        window_size = attention.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size) # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(
            sigma.shape[0], sigma.shape[1], 1, 1
        ).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attention, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)
        

class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model: int,
        num_heads: int,
        d_keys = None,
        d_values = None,
    ) -> None:
        super().__init__()
        d_keys = d_keys or (d_model // num_heads)
        d_values = d_values or (d_model // num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(
            in_features=d_model,
            out_features=d_keys * num_heads,
        )
        self.key_projection = nn.Linear(
            in_features=d_model,
            out_features=d_keys * num_heads,
        )
        self.value_projection = nn.Linear(
            in_features=d_model,
            out_features=d_values * num_heads,
        )
        self.sigma_projection = nn.Linear(
            in_features=d_model,
            out_features=num_heads,
        )
        self.out_projection = nn.Linear(
            in_features=d_values * num_heads,
            out_features=d_model,
        )
        self.n_heads = num_heads
    
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries=queries,
            keys=keys,
            values=values,
            sigma=sigma,
            attention_mask=attention_mask,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma
