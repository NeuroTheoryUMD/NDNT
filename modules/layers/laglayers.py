from numpy.lib.arraysetops import isin
import torch
from torch.nn import functional as F
import torch.nn as nn

#from .convlayer import ConvLayer
from .ndnlayer import NDNLayer

import numpy as np

class LagLayer(NDNLayer):
    """
    Temporal-convolutional NDN Layer

    Takes time-embedded stimulus with input dims
    Args (required):
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_dims: width of convolutional kernel (int or list of ints)
    Args (optional):
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'

    """
    def __init__(self, input_dims=None, num_filters=None, num_lags=None, **kwargs):
        
        assert input_dims is not None, "LagLayer: input_dims must be specified"
        assert num_filters is not None, "LagLayer: num_filters must be specified"
        assert num_lags is not None, "LagLayer: num_lags must be specified"
        assert num_lags < input_dims[3], "LagLayer: num_lags must be less than input_dims"

        filter_dims = input_dims.copy()
        filter_dims[3] = num_lags
        super().__init__(input_dims=input_dims, num_filters=num_filters, filter_dims=filter_dims, **kwargs)

        # These dims are processed by filter, and must match between stimulus and filter dims
        self.num_lags = num_lags
        output_dims = self.output_dims  # this automatically sets num_outputs as well
        output_dims[3] = self.input_dims[3]-num_lags+1
        self.output_dims = output_dims
        self.num_folded_dims = np.prod(filter_dims[:3])
    #END LagLayer.__init__

    def forward(self, x):

        w = self.preprocess_weights()

        # Use 1-d convolve 
        s = x.view( [-1, self.num_folded_dims, self.input_dims[3]] ) # [B,C,T]
        w = w.view( [self.num_folded_dims, self.num_lags, -1] ).permute(2,0,1) # [C,T,N] -> [N,C,T]

        y = F.conv1d( 
            #x.view( [-1, self.num_folded_dims, self.input_dims[3]] ), # [B,C,T],
            #w.view( [self.num_folded_dims, self.num_lags, -1] ), # [C,T,N]
            s, w,
            bias=self.bias)
        
        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)
        
        if self._ei_mask is not None:
            y = y * self._ei_mask[None,:,None]
        
        y = y.reshape((-1, self.num_outputs))

        return y
    #END LagLayer.forward

    def plot_filters( self, cmaps='gray', num_cols=8, row_height=2, time_reverse=False):
        # Overload plot_filters to automatically time_reverse
        super().plot_filters( 
            cmaps=cmaps, num_cols=num_cols, row_height=row_height, 
            time_reverse=time_reverse)

    @classmethod
    def layer_dict(cls, num_lags=None, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'lag'
        Ldict['num_lags'] = num_lags
        return Ldict
    # END [classmethod] TconvLayer.layer_dict
