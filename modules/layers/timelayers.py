import torch
import torch.nn as nn
from .ndnlayer import NDNLayer
import numpy as np
class TimeLayer(NDNLayer):
    """
    Layer to track experiment time and a weighted output.
    """ 

    def __init__(self,
            start_time=0,
            end_time=1000,
            input_dims=None,
            num_bases=None,
            num_filters=None,
            filter_dims=None,
            pos_constraint=True,
            **kwargs,
        ):
        """
        TimeLayer: Layer to track experiment time and a weighted output.

        Args:
            start_time: float, start time of experiment
            end_time: float, end time of experiment
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_bases: number of tent basis functions
            num_filters: number of output filters
            filter_dims: tuple or list of ints, (num_channels, height, width, lags)
            pos_constraint: bool, whether to enforce non-negative weights
            **kwargs: additional arguments to pass to NDNLayer
        """
        assert num_filters is not None, "TimeLayer: Must specify num_filters"
        assert num_bases is not None, "TimeLayer: Must specify num_bases"
        
        if not pos_constraint:
            pos_constraint = True

        if num_filters is None:            
            num_filters = input_dims[0]

        # inputs are a single column of frame times
        # that get converted into a tent basis
        input_dims = [num_bases, 1, 1, 1] # assume non-convolutional

        # number of filters (and size of filters) is set by channel dims on the input
        filter_dims = [num_bases, 1, 1, 1]

        super().__init__(input_dims,
            num_filters,
            filter_dims=filter_dims,
            pos_constraint=pos_constraint, **kwargs)

        self.output_dims = self.input_dims

        self.register_buffer( 'tent_centers', torch.linspace(start_time, end_time, num_bases))
        self.register_buffer('step', self.tent_centers[1] - self.tent_centers[0])

        self.num_outputs = int(np.prod(self.output_dims))
        if self.NL is None:
            self.NL = nn.ReLU()
        
        self.weight.data.fill_(1.0/self.num_outputs)
        self.bias.data.fill_(0)
        self.bias.requires_grad = False
    

    def forward(self, x):
        """
        Forward pass through the TimeLayer.

        Args:
            x: torch.Tensor, input tensor

        Returns:
            y: torch.Tensor, output tensor
        """
        # Pull weights and process given pos_constrain and normalization conditions
        w = self.preprocess_weights()

        # convert to tent basis
        x = x - self.tent_centers
        
        x = self.NL( 1-x.abs()/self.step )
    
        x = x@w
        
        return x