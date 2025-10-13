#from numpy.lib.arraysetops import isin
import torch
from torch.nn import functional as F
import torch.nn as nn

#from .convlayer import ConvLayer
from .ndnlayer import NDNLayer

import numpy as np

class LagLayer(NDNLayer):
    """
    Operates spatiotemporal filter om time-embeeded stimulus
    Spatiotemporal filter should have less lags than stimulus -- then ends up with some lags left
    Filter is full spatial width and the number of lags is explicity specified in initializer
    (so, inherits spatial and chanel dimensions of stimulus input)

    Args (required):
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        num_lags: number of lags in spatiotemporal filter
    
    Args (optional):
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
    """
    def __init__(self, input_dims=None, num_filters=None, num_lags=None, **kwargs):
        """
        Initialize the layer.

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: int, number of output filters
            num_lags: int, number of lags in spatiotemporal filter
            **kwargs: keyword arguments to pass to the parent class
        """
        assert input_dims is not None, "LagLayer: input_dims must be specified"
        assert num_filters is not None, "LagLayer: num_filters must be specified"
        assert num_lags is not None, "LagLayer: num_lags must be specified"
        assert num_lags < input_dims[3], "LagLayer: num_lags must be less than input_dims"

        filter_dims = input_dims.copy()
        filter_dims[3] = num_lags
        super().__init__(input_dims=input_dims, num_filters=num_filters, filter_dims=filter_dims, **kwargs)

        # These dims are processed by filter, and must match between stimulus and filter dims
        self.num_lags = num_lags
        self.output_dims[3] = self.input_dims[3]-num_lags+1
        self.num_folded_dims = np.prod(filter_dims[:3])
    #END LagLayer.__init__

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
            x: torch.Tensor, input tensor of shape (batch_size, *input_dims)

        Returns:
            y: torch.Tensor, output tensor of shape (batch_size, *output_dims)
        """
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
        
        y = y.reshape((-1, np.prod(self.output_dims)))

        return y
    #END LagLayer.forward

    def plot_filters( self, cmaps='gray', num_cols=8, row_height=2, time_reverse=False):
        """
        Overload plot_filters to automatically time_reverse.

        Args:
            cmaps: str or list of str, colormap(s) to use for plotting
            num_cols: int, number of columns in the plot
            row_height: int, height of each row in the plot
            time_reverse: bool, whether to reverse the time axis

        Returns:
            fig: matplotlib.figure.Figure, the figure object
            axs: list of matplotlib.axes.Axes, the axes objects
        """
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

        Args:
            num_lags: int, number of lags in spatiotemporal filter
            **kwargs: keyword arguments to pass to the parent class

        Returns:
            Ldict: dict, dictionary of layer parameters
        """

        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'lag'
        Ldict['num_lags'] = num_lags
        return Ldict
    # END [classmethod] TconvLayer.layer_dict
