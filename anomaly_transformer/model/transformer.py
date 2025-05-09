from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding
Tensor = torch.Tensor


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = 'relu',
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_size=1,
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        new_x, attention, mask, sigma = self.attention(
            x, x, x, attention_mask=attention_mask
        )
        x = x + self.dropout(new_x) # x += self.dropout(new_x) <- causes inplace error
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attention, mask, sigma
    

class Encoder(nn.Module):
    def __init__(
        self,
        attention_layers,
        norm_layer=None,
    ) -> None:
        super().__init__()
        self.attention_layers = nn.ModuleList(attention_layers)
        self.norm = norm_layer

    def forward(
        self,
        x: Tensor,
        attention_mask = None,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor]]:
        series_list = []
        prior_list = []
        sigma_list = []
        for attention_layer in self.attention_layers:
            x, series, prior, sigma = attention_layer(
                x, attention_mask=attention_mask
            )
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, series_list, prior_list, sigma_list
    

class AnomalyTransformer(nn.Module):
    def __init__(
        self,
        window_size: int,
        enc_in: int,
        c_out: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_enc_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.0,
        activation: str = 'gelu',
        output_attention: bool = True,
    ) -> None:
        super().__init__()
        self.output_attention = output_attention

        self.embedding = DataEmbedding(
            c_in=enc_in,
            d_model=d_model,
            dropout=dropout)
        
        self.encoder = Encoder(
            attention_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=AnomalyAttention(
                            window_size=window_size,
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model=d_model,
                        num_heads=num_heads
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(num_enc_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(
            in_features=d_model,
            out_features=c_out,
            bias=True,
        )

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out
    