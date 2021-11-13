
import numpy as np
import torch
from torch import nn
# from typing import Tuple, Union

from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

from NDNT.modules.regularization import Regularization
from .. activations import NLtypes

from copy import deepcopy

'''
1. NDNLayer (base class, linear layer)
2. ConvLayer (convolutional NDNLayer, can be 1D or 2D)
3. STConvLayer (Spatio-Temporal Convolutional NDNLayer, uses the batch dimension as contiguous time)
4. DivNormLayer (Divisive Normalization Layer, can operate on linear or conv layers, has no spatial normalization pool)
5. FixationLayer 

'''

class NDNLayer(nn.Module):
    """
    Base class for NDN layers.
    Handles the weight initialization and regularization

    Args:
        input_dims: list of 4 ints, dimensions of input
            [Channels, Height, Width, Lags]
            use 1 if the dimension is not used
        num_filters: int, number of filters in layer
        NLtype: str, type of nonlinearity to use (see activations.py)
        norm: int, type of filter normalization to use (0=None, 1=filters are unit vectors, 2=maxnorm?)
        pos_constraint: bool, whether to constrain weights to be positive
        num_inh: int, number of inhibitory units (creates ei_mask and makes "inhibitory" units have negative output)
        bias: bool, whether to include bias term
        weight_init: str, type of weight initialization to use (see reset_parameters, default 'xavier_uniform')
        bias_init: str, type of bias initialization to use (see reset_parameters, default 'zeros')
        reg_vals: dict, regularization values to use (see regularizers.py)

    NDNLayers have parameters 'weight' and 'bias' (if bias=True)

    The forward has steps:
        1. preprocess_weights
        2. main computation (this is just a linear layer for the base class)
        3. nonlinearity
        4. ei_mask
    
    weight is always flattened to a vector, and then reshaped to the appropriate size
    use get_weights() to get the weights in the correct shape
    
    preprocess_weights() applies positive constraint and normalization if requested
    compute_reg_loss() computes the regularization loss for the regularization specified in reg_vals

    """
    def __init__(self, input_dims=None,
            num_filters=None,
            filter_dims=None,
            NLtype:str='lin',
            norm_type:int=0,
            pos_constraint=False,
            num_inh:int=0,
            bias:bool=False,
            weights_initializer:str='xavier_uniform',
            bias_initializer:str='zeros',
            reg_vals:dict=None,
            **kwargs,
            ):

        assert input_dims is not None, "NDNLayer: Must specify input_dims"
        assert num_filters is not None, "NDNLayer: Must specify num_filters"

        # if len(kwargs) > 0:
        #     print("NDNLayer: unknown kwargs:", kwargs)

        super().__init__()

        self.input_dims = input_dims
        self.num_filters = num_filters
        if filter_dims is None:
            self.filter_dims = deepcopy(input_dims)
        else:
            self.filter_dims = filter_dims
        
        output_dims = [num_filters, 1, 1, 1]
        self.output_dims = output_dims
        self.num_outputs = np.prod(self.output_dims)
        
        self.norm_type = norm_type
        self.pos_constraint = pos_constraint
        self.conv = False
        
        # Was this implemented correctly? Where should NLtypes (the dictionary) live?
        if NLtype in NLtypes:
            self.NL = NLtypes[NLtype]
        else:
            print("Nonlinearity undefined.")
            # return error

        self.shape = tuple([np.prod(self.filter_dims), self.num_filters])

        # Make layer variables
        self.weight = Parameter(torch.Tensor(size=self.shape))
        if bias:
            self.bias = Parameter(torch.Tensor(self.num_filters))
        else:
            self.register_buffer('bias', torch.zeros(self.num_filters))

        if self.pos_constraint:
            self.register_buffer("minval", torch.tensor(0.0))
            # Does this apply to both weights and biases? Should be separate?
            # How does this compare without explicit constraint? Maybe better, maybe worse...

        self.reg = Regularization( filter_dims=self.filter_dims, vals=reg_vals)

        if num_inh == 0:
            self.ei_mask = None
        else:
            self.register_buffer('ei_mask', torch.ones(self.num_filters))  
            self.ei_mask[-(num_inh+1):0] = -1

        self.reset_parameters( weights_initializer, bias_initializer )

    def reset_parameters(self, weights_initializer=None, bias_initializer=None, param=None) -> None:
        '''
        Initialize the weights and bias parameters of the layer.
        Args:
            weights_initializer: str, type of weight initialization to use
                options: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal', 'zeros', 'ones'
            bias_initializer: str, type of bias initialization to use
        '''
        # Default initializer, although others possible
        if weights_initializer is None:
            weights_initializer = 'uniform'
        if bias_initializer is None:
            bias_initializer = 'zeros'

        if weights_initializer == 'uniform':
            if param is None:
                param = 5**0.5
            init.kaiming_uniform_(self.weight, a=param)
        elif weights_initializer == 'zeros':
            init.zeros_(self.weight)
        elif weights_initializer == 'normal':
            if param is None:
                mean = 0.0
                std = 1.0
            elif isinstance(param, list):
                mean, std = param
            else:
                RuntimeError("if using 'normal', param must be None or a list of length 2")
            init.normal_(self.weight, mean=mean, std=std)

        elif weights_initializer == 'xavier_uniform':  # also known as Glorot initialization
            if param is None:
                param = 1.0
            init.xavier_uniform_(self.weight, gain=param)  # set gain based on weights?
        elif weights_initializer == 'xavier_normal':  # also known as Glorot initialization
            if param is None:
                param = 1.0
            init.xavier_normal_(self.weight, gain=param)  # set gain based on weights?
        else:
            print('weights initializer not defined')

        if bias_initializer == 'zeros':
            init.zeros_(self.bias)
        elif bias_initializer == 'uniform':
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)          
        else:
            print('bias initializer not defined')
    
    def preprocess_weights(self):
        # Apply positive constraints
        if self.pos_constraint:
            w = torch.maximum(self.weight, self.minval)
        else:
            w = self.weight
        # Add normalization
        if self.norm_type > 0: # so far just standard filter-specific normalization
            w = F.normalize( w, dim=0 )
        return w

    def forward(self, x):
        # Pull weights and process given pos_constrain and normalization conditions
        w = self.preprocess_weights()
        # Simple linear processing and bias
        x = torch.matmul(x, w) + self.bias

        # Nonlinearity
        if self.NL is not None:
            x = self.NL(x)

        # Constrain output to be signed
        if self.ei_mask is not None:
            x = x * self.ei_mask

        return x 
        
    def compute_reg_loss(self):
        return self.reg.compute_reg_loss(self.preprocess_weights())

    def get_weights(self, to_reshape=True, time_reverse=False):
        ws = self.preprocess_weights().detach().cpu().numpy()
        num_filts = ws.shape[-1]
        if time_reverse:
            ws_tmp = np.reshape(ws, self.filter_dims + [num_filts])
            num_lags = self.filter_dims[3]
            if num_lags > 1:
                ws_tmp = ws_tmp[:, :, :, range(num_lags-1, -1, -1), :] # time reversal here
            ws = ws_tmp.reshape((-1, num_filts))
        if to_reshape:
            return ws.reshape(self.filter_dims + [num_filts]).squeeze()
        else:
            return ws.squeeze()

    def list_parameters(self):
        for nm, pp in self.named_parameters(recurse=False):
            if pp.requires_grad:
                print("      %s:"%nm, pp.size())
            else:
                print("      NOT FIT: %s:"%nm, pp.size())

    def set_parameters(self, name=None, val=None ):
        """
        Turn fitting for named params on or off.
        If name is none, do for whole layer.
        """
        assert isinstance(val, bool), 'val must be a boolean (True or False).'
        for nm, pp in self.named_parameters(recurse=False):
            if name is None:
                pp.requires_grad = val
            elif nm == name:
                pp.requires_grad = val

    def plot_filters( self, cmaps='gray', num_cols=8, row_height=2, time_reverse=True):
        """
        Plot the filters in the layer.
        Args:
            cmaps: str or colormap, colormap to use for plotting
            num_cols: int, number of columns to use in plot
            row_height: int, number of rows to use in plot
            time_reverse: bool, whether to reverse the time dimension
        """
        ws = self.get_weights(time_reverse=time_reverse)
        # check if 1d: otherwise return error for now
        num_filters = ws.shape[-1]
        if num_filters < 8:
            num_cols = num_filters
        num_rows = np.ceil(num_filters/num_cols).astype(int)
        if self.input_dims[2] == 1:
            import matplotlib.pyplot as plt
            # then 1-d spatiotemporal plots
            fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
            fig.set_size_inches(16, row_height*num_rows)
            plt.tight_layout()
            for cc in range(num_filters):
                ax = plt.subplot(num_rows, num_cols, cc+1)
                plt.imshow(ws[:,:,cc].T, cmap=cmaps)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            plt.show()
        else:
            print("Not implemented plotting 3-d filters yet.")