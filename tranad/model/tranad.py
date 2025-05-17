from typing import *
from math import sqrt
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder
import torch.optim as optim
import pickle
from utils.constants import *
from .encoding import PositionalEncoding
from .layers import Attention, TransformerEncoderLayer, TransformerDecoderLayer
    

class TranADBasic(nn.Module):
    def __init__(
        self,
        n_features: int,
    ) -> None:
        super().__init__()
        self.name = 'TranADBasic'
        self.lr = lr
        self.batch = 128
        self.n_features = n_features
        self.n_window = 10
        self.n = self.n_features * self.n_window
        self.pos_encoder = PositionalEncoding(
            d_model=n_features,
            dropout=0.1,
            max_len=self.n_window
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=n_features,
                num_heads=n_features,
                d_ff=16,
                dropout=0.1,
            ),
            num_layers=1,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=n_features,
                num_heads=n_features,
                d_ff=16,
                dropout=0.1,
            ),
            num_layers=1,
        )
        self.activation = nn.Sigmoid()

    def forward(
        self,
        src: Tensor,
        target: Tensor,
    ) -> Tensor:
        src = src * sqrt(self.n_features)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        x = self.transformer_decoder(target, memory)
        x = self.activation(x)
        return x

class TranADTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
    ) -> None:
        super().__init__()
        self.name = 'TranADTransformer'
        self.lr = lr
        self.batch_size = 128
        self.n_features = n_features
        self.n_hidden = 8
        self.n_windows = 10
        self.n = 2 * self.n_features * self.n_windows
        self.transformer_encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n),
            nn.ReLU(True),
        )
        self.transformer_decoder1 = nn.Sequential(
            nn.Linear(self.n, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, 2 * n_features),
            nn.ReLU(),
        )
        self.transformer_decoder2 = nn.Sequential(
            nn.Linear(self.n, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, 2 * n_features),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            nn.Sigmoid(),
        )

    def encode(
        self,
        src: Tensor,
        c: Tensor,
        target: Tensor,
    ) -> Tensor:
        src = torch.cat((src, c), dim=2)
        src = src.permute(1, 0, 2).flatten(start_dim=1)
        target = self.transformer_encoder(src)
        return target

    def forward(
        self,
        src: Tensor,
        target: Tensor,
    ) -> Tensor:
        # Phase 1: W.O. anomaly scores
        c = torch.zeros_like(src)
        x1 = self.transformer_decoder1(self.encode(src, c, target))
        x1 = x1.reshape(-1, 1, 2 * self.n_features).permute(1, 0, 2)
        x1 = self.fc(x1)
        # Phase 2: With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.transformer_decoder2(self.encode(src, c, target))
        x2 = x2.reshape(-1, 1, 2 * self.n_features).permute(1, 0, 2)
        x2 = self.fc(x2)
        return x1, x2
    

class TranADAdversarial(nn.Module):
    def __init__(
        self,
        n_features: int,
    ) -> None:
        super().__init__()
        self.name = 'TranADAdversarial'
        self.lr = lr
        self.batch = 128
        self.n_features = n_features
        self.n_windows = 10
        self.n = self.n_features * self.n_windows
        self.pos_encoder = PositionalEncoding(
            d_model=2 * n_features,
            dropout=0.1,
            max_len=self.n_windows,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=2 * n_features,
                num_heads=n_features,
                d_ff=16,
                dropout=0.1,
            ),
            num_layers=1,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=TransformerDecoder(
                decoder_layer=TransformerDecoderLayer(
                    d_model=2 * n_features,
                    num_heads=n_features,
                    d_ff=16,
                    dropout=0.1,
                )
            ),
            num_layers=1,
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            nn.Sigmoid(),
        )

    def encode_decode(
        self,
        src: Tensor,
        c: Tensor,
        target: Tensor,
    ) -> Tensor:
        src = torch.cat((src, c), dim=2)
        src = src * sqrt(self.n_features)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        target = target.repeat(1, 1, 2)
        x = self.transformer_decoder(target, memory)
        x = self.fc(x)
        return x
    
    def forward(
        self,
        src: Tensor,
        target: Tensor,
    ) -> Tensor:
        # Phase 1: W.O. anomaly scores
        c = torch.zeros_like(src)
        x = self.encode_decode(src, c, target)

        # Phase 2: With anomlay scores
        c = (x - src) ** 2
        x = self.encode_decode(src, c, target)

        return x
    

class TranADSelfConditioning(nn.Module):
    def __init__(
        self,
        n_features: int,
    ) -> None:
        super().__init__()
        self.name = 'TranADSelfConditioning'
        self.lr = lr
        self.batch_size = 128
        self.n_features = n_features
        self.n_windows = 10
        self.n = self.n_features * self.n_windows
        self.pos_encoder = PositionalEncoding(
            d_model=2 * n_features,
            dropout=0.1,
            max_len=self.n_windows,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=2 * n_features,
                num_heads=n_features,
                d_ff=16,
                dropout=0.1,
            ),
            num_layers=1,
        )
        self.transformer_decoder1 = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=2 * n_features,
                num_heads=n_features,
                d_ff=16,
                dropout=0.1,
            ),
            num_layers=1,
        )
        self.transformer_decoder2 = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=2 * n_features,
                num_heads=n_features,
                d_ff=16,
                dropout=0.1,
            ),
            num_layers=1,
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            nn.Sigmoid(),
        )

    def encode(
        self,
        src: Tensor,
        c: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        src = torch.cat((src, c), dim=2)
        src = src * sqrt(self.n_features)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        target = target.repeat(1, 1, 2)
        return target, memory
    
    def forward(
        self,
        src: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Phase 1: W.O. anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fc(self.transformer_decoder1(*self.encode(src, c, target)))

        # Phase 2: With anomaly scores
        x2 = self.fc(self.transformer_decoder2(*self.encode(src, c, target)))

        return x1, x2


class TranAD(nn.Module):
    def __init__(
        self,
        n_features: int,
    ) -> None:
        super().__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch_size = 128
        self.n_features = n_features
        self.n_windows = 10
        self.n = self.n_features * self.n_windows
        self.pos_encoder = PositionalEncoding(
            d_model=2 * n_features,
            dropout=0.1,
            max_len=self.n_windows,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=2 * n_features,
                num_heads=n_features,
                d_ff=16,
                dropout=0.1,
            ),
            num_layers=1,
        )
        self.transformer_decoder1 = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=2 * n_features,
                num_heads=n_features,
                d_ff = 16,
                dropout=0.1,
            ),
            num_layers=1,
        )
        self.transformer_decoder2 = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=2 * n_features,
                num_heads=n_features,
                d_ff = 16,
                dropout=0.1,
            ),
            num_layers=1,
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            nn.Sigmoid(),
        )

    def encode(
        self,
        src: Tensor,
        c: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        src = torch.cat((src, c), dim=2)
        src = src * sqrt(self.n_features)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        target = target.repeat(1, 1, 2)
        return target, memory
    
    def forward(
        self,
        src: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Phase 1: W.O. anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fc(self.transformer_decoder1(*self.encode(src, c, target)))

        # Phase 2: With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fc(self.transformer_decoder2(*self.encode(src, c, target)))

        return x1, x2
