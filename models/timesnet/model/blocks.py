from typing import *
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


def FFTforPeriod(
    x: Tensor,
    k: int = 2,
) -> Tuple[int, ]:
    x_fft = fft.rfft(x, dim=1)
    freq_list = abs(x_fft).mean(0).mean(-1)
    freq_list[0] = 0
    _, top_list = torch.topk(freq_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    
    return period, abs(x_fft).mean(-1)[:, top_list]


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 16,
        init_weight: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels = []

        for i in range(num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * i + 1,
                    padding=i,
                )
            )
        
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._init_weights()
        
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    tensor=m.weight,
                    mode='fan_out',
                    nonlinearity='relu',
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:

        res_list = []
        
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        
        res = torch.stack(res_list, dim=-1).mean(-1)

        return res


class TimesBlock(nn.Module):
    def __init__(
        self,
        configs
    ) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            InceptionBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_ff,
                num_kernels=configs.num_kernels,
            ),
            nn.GELU(),
            InceptionBlock(
                in_channels=configs.d_ff,
                out_channels=configs.d_model,
                num_kernels=configs.num_kernels,
            ),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        B, T, N = x.size()
        period_list, period_weight = FFTforPeriod(x, self.k)

        res = []

        for i in range(self.k):
            period = period_list[i]
            
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros(
                    [
                        x.shape[0],
                        (length - (self.seq_len + self.pred_len)),
                        x.shape[2]
                    ]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            
            else:
                length = self.seq_len + self.pred_len
                out = x
            
            out = out.reshape(
                B, length // period, period, N
            ).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        
        res = torch.stack(res, dim=-1)

        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x

        return res
