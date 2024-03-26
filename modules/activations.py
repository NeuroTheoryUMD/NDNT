import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def adaptive_elu(x, xshift, yshift, inplace=False):
    """
    Exponential Linear Unit shifted by user specified values.

    Args:
        x (torch.Tensor): input tensor
        xshift (float): shift in x direction
        yshift (float): shift in y direction
        inplace (bool): whether to modify the input tensor in place

    Returns:
        torch.Tensor: output tensor
    """
    return F.elu(x - xshift, inplace=inplace) + yshift


class Square(nn.Module):
    """
    Square activation function.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Forward pass of the square activation function.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return x**2

class AdaptiveELU(nn.Module):
    """
    Exponential Linear Unit shifted by user specified values.
    This helps to ensure the output to stay positive.
    """

    def __init__(self, xshift=0.0, yshift=1.0, inplace=True, **kwargs):
        """
        Initialize the AdaptiveELU activation function.

        Args:
            xshift (float): shift in x direction
            yshift (float): shift in y direction
            inplace (bool): whether to modify the input tensor in place
            **kwargs: additional keyword arguments

        Returns:
            None
        """
        super(AdaptiveELU, self).__init__(**kwargs)

        self.xshift = xshift
        self.yshift = yshift
        self.inplace = inplace

    def forward(self, x):
        """
        Forward pass of the AdaptiveELU activation function.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
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
