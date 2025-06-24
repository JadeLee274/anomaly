from typing import *
import torch
from torch import Tensor
import torch.nn as nn


class SENNGC(nn.Module):
    def __init__(
        self,
        num_vars: int,
        order: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
        device: torch.device,
    ) -> None:
        """
        Generalized VAR (GVAR) model based on self-explaining neural netrowrs
        @param num_vars: number of variables (p).
        @param order: model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        """
        super().__init__()
        self.coeff_nets = nn.ModuleList()

        for k in range(order):
            modules = [
                nn.Sequential(
                    nn.Linear(num_vars, hidden_layer_size),
                    nn.ReLU(),
                )
            ]
            if num_hidden_layers > 1:
                for j in range(num_hidden_layers - 1):
                    modules.extend(
                        nn.Sequential(
                            nn.Linear(hidden_layer_size, hidden_layer_size),
                            nn.ReLU(),
                        )
                    )
                modules.extend(
                    nn.Sequential(
                        nn.Linear(hidden_layer_size, num_vars**2),
                        nn.Tanh(),
                    )
                )
                self.coeff_nets.append(nn.Sequential(*modules))
        
        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.device = device

    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight_data)
            m.bias.data.fill_(0.1)
    
    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if x[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print('WARNING: inputs should be of shape BS x K x P')
        
        coeffs = None
        preds = torch.zeros((x.shape[0], self.num_vars)).to(self.device)

        for k in range(self.order):
            coeff_net_k = self.coeff_nets[k]
            coeffs_k = coeff_net_k(x[:, k, :])
            coeffs_k = torch.reshape(coeffs_k, (x.shape[0], self.num_vars, self.num_vars))
            
            if coeffs is None:
                coeffs = torch.unsqueeze(coeffs_k, 1)
            else:
                coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), dim=1)
            
            preds = preds + torch.matmul(coeffs_k, x[:, k, :].unsqueeze(dim=2)).squeeze()
        
        return preds, coeffs
