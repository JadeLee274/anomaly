import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
Tensor = torch.Tensor


def to_var(
    x: Tensor,
    volatile: bool = False,
) -> Tensor:
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
