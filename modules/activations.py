
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def adaptive_elu(x, xshift, yshift, inplace=False):
    return F.elu(x - xshift, inplace=inplace) + yshift


class Square(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x**2

class AdaptiveELU(nn.Module):
    """
    Exponential Linear Unit shifted by user specified values.
    This helps to ensure the output to stay positive.
    
    """

    def __init__(self, xshift=0.0, yshift=1.0, inplace=True, **kwargs):
        super(AdaptiveELU, self).__init__(**kwargs)

        self.xshift = xshift
        self.yshift = yshift
        self.inplace = inplace

    def forward(self, x):
        return adaptive_elu(x, self.xshift, self.yshift, inplace=self.inplace)

NLtypes = {
    'lin': None,
    'relu': nn.ReLU(),
    'elu': AdaptiveELU(),
    'square': Square(),
    'softplus': nn.Softplus(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'gelu': nn.GELU()
    }