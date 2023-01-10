import torch
import torch.nn as nn
from .convlayers import ConvLayer
import numpy as np


class BiConvLayer1D(ConvLayer):
    """
    Filters that act solely on filter-dimension (dim-0)

    """ 

    def __init__(self, **kwargs ):
        """Same arguments as ConvLayer, but will reshape output to divide space in half"""

        super().__init__(**kwargs)

        self.output_dims[0] = self.output_dims[0]*2
        self.output_dims[1] = self.output_dims[1]//2

    # END BinConvLayer1D.__init__

    @classmethod
    def layer_dict(cls, **kwargs):
        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'biconv'
        return Ldict
    # END [classmethod] BinConvLayer1D.layer_dict

