
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
            pos_constraint=0,
            num_inh:int=0,
            bias:bool=False,
            weights_initializer:str='xavier_uniform',
            initialize_center=False,
            bias_initializer:str='zeros',
            reg_vals:dict=None,
            **kwargs,
            ):

        assert input_dims is not None, "NDNLayer: Must specify input_dims"
        assert num_filters is not None, "NDNLayer: Must specify num_filters"
        #assert len(num_filters) > 0, "NDNLayer: num_filters is incorrectly specified."

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
        self.output_dims = output_dims  # this automatically sets num_outputs as well
        #self.num_outputs = np.prod(self.output_dims) # Make this assigned through property
        
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
        self.weight_scale = np.sqrt(self.shape[0]) / 100
        # self.weight_scale = np.sqrt(self.shape[0]) * 100

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

        self.reg = Regularization( filter_dims=self.filter_dims, vals=reg_vals, num_outputs=num_filters )

        # Now taken care of in model properties
        self.num_inh = num_inh
        #if num_inh == 0:
        #    self.ei_mask = None
        #else:
        #    self.register_buffer('ei_mask', torch.cat( (torch.ones(self.num_filters-num_inh), -torch.ones(num_inh))) )

        # Set inital weight and bias values
        self.reset_parameters( weights_initializer, bias_initializer )
        if initialize_center:
            self.initialize_gaussian_envelope()
    # END NDNLayer.__init__

    @property
    def output_dims(self):
        return self._output_dims

    @output_dims.setter
    def output_dims(self, value):
        self._output_dims = value
        self.num_outputs = int(np.prod(self._output_dims))

    @property
    def num_inh(self):
        return self._num_inh

    @num_inh.setter
    def num_inh(self, value):
        assert value >= 0, "num_inh cannot be negative"
        self._num_inh = value
        if value == 0:
            self._ei_mask = None
        else:
            self.register_buffer('_ei_mask', 
                torch.cat( (torch.ones(self.num_filters-self._num_inh), -torch.ones(self._num_inh))) )

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
                #param = 5**0.5
                param = 1.0/self.num_filters
            init.kaiming_uniform_(self.weight, a=param)
        elif weights_initializer == 'zeros':
            init.zeros_(self.weight)
        elif weights_initializer == 'ones':
            init.ones_(self.weight)
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

        if self.pos_constraint>0:
            self.weight.data = abs(self.weight)
        elif self.pos_constraint<0:
            self.weight.data = -abs(self.weight)

        if self.norm_type == 1:
            self.weight.data = F.normalize( self.weight.data, dim=0 ) / self.weight_scale   

        if bias_initializer == 'zeros':
            init.zeros_(self.bias)
        elif bias_initializer == 'uniform':
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)          
        else:
            print('bias initializer not defined')
    
    def initialize_gaussian_envelope(self):
        from NDNT.utils import initialize_gaussian_envelope
        w_centered = initialize_gaussian_envelope( self.get_weights(to_reshape=False), self.filter_dims)
        self.weight.data = torch.tensor(w_centered, dtype=torch.float32)

    def preprocess_weights(self):
        # Apply positive constraints
        if self.pos_constraint>0:
            #w = torch.maximum(self.weight, self.minval)
            #w = torch.square(self.weight)
            w = self.weight.clamp(min=0)
        elif self.pos_constraint<0:
            w = self.weight.clamp(max=0)
            # note this is instead of w = self.weight.clamp(min=0)
        else:
            w = self.weight
        # Add normalization
        if self.norm_type == 1: # so far just standard filter-specific normalization
            w = F.normalize( w, dim=0 ) / self.weight_scale
        return w

    def forward(self, x):
        # Pull weights and process given pos_constrain and normalization conditions
        w = self.preprocess_weights()

        # Simple linear processing and bias
        x = torch.matmul(x, w)
        if self.norm_type == 2:
            x = x / self.weight_scale

        x = x + self.bias

        # Nonlinearity
        if self.NL is not None:
            x = self.NL(x)

        # Constrain output to be signed
        if self._ei_mask is not None:
            x = x * self._ei_mask

        return x 
    # END NDNLayer.forward
        
    def compute_reg_loss(self):
        return self.reg.compute_reg_loss(self.preprocess_weights())

    def get_weights(self, to_reshape=True, time_reverse=False, num_inh=0):
        """num-inh can take into account previous layer inhibition weights"""
        
        ws = self.preprocess_weights().detach().cpu().numpy()
        num_filts = ws.shape[-1]
        if time_reverse or (num_inh>0):
            ws_tmp = np.reshape(ws, self.filter_dims + [num_filts])
            num_lags = self.filter_dims[3]
            if time_reverse and (num_lags > 1):
                ws_tmp = ws_tmp[:, :, :, range(num_lags-1, -1, -1), :] # time reversal here
            if num_inh > 0:
                ws_tmp[range(-num_inh, 0), ...] *= -1
            ws = ws_tmp.reshape((-1, num_filts))
        if to_reshape:
            ws = ws.reshape(self.filter_dims + [num_filts]).squeeze()
        else:
            ws = ws.squeeze()

        if num_filts == 1:
            # Add singleton dimension corresponding to number of filters
            ws = ws[..., None]
        return ws

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

    def set_reg_val( self, reg_type, reg_val=None):
        self.reg.set_reg_val( reg_type=reg_type, reg_val=reg_val )

    def plot_filters( self, time_reverse=None, **kwargs):
        """
        Plot the filters in the layer.
        Args:
            cmaps: str or colormap, colormap to use for plotting (default 'gray')
            num_cols: int, number of columns to use in plot (default 8)
            row_height: int, number of rows to use in plot (default 2)
            time_reverse: bool, whether to reverse the time dimension (default depends on dimension)
        """

        # Make different defaults for time_reverse depending on 1 vs 2D
        if time_reverse is None:
            if self.input_dims[2] == 1:
                time_reverse = True
            else:
                time_reverse = False
            
        ws = self.get_weights(time_reverse=time_reverse)
        
        if self.input_dims[2] == 1:
            from NDNT.utils import plot_filters_ST1D
            plot_filters_ST1D(ws, **kwargs)
        else:
            if self.input_dims[0] == 1:
                from NDNT.utils import plot_filters_ST2D
                plot_filters_ST2D(ws, **kwargs)
            else:
                from NDNT.utils import plot_filters_ST3D
                plot_filters_ST3D(ws, **kwargs)
    # END NDNLayer.plot_filters()

    @staticmethod
    def dim_info( input_dims=None, num_filters=None, **kwargs):
        """
        This uses the methods in the init to determine the input_dims, output_dims, filter_dims, and actual size of
        the weight tensor (weight_shape), given inputs, and package in a dictionary. This should be overloaded with each
        child of NDNLayer if want to use -- but this is external to function of actual layer.
        """
        assert input_dims is not None, "NDNLayer: Must include input_dims."
        assert num_filters is not None, "NDNLayer: Must include num_filters."

        filter_dims = deepcopy(input_dims)
        output_dims = tuple([num_filters, 1, 1, 1])
        num_outputs = np.prod(output_dims) # this will always be the case so might be extraneous, but is convenient....
        weight_shape = tuple([np.prod(filter_dims), num_filters])  # likewise

        dinfo = {
            'input_dims': tuple(input_dims), 'filter_dims': tuple(filter_dims), 
            'output_dims': output_dims, 'num_outputs': num_outputs,
            'weight_shape': weight_shape}

        return dinfo
    # END [static] NDNLayer.dim_info

    @classmethod
    def layer_dict(cls, 
            input_dims=None, num_filters=None, # necessary parameters
            bias=False, NLtype='lin', norm_type=0,
            initialize_center=False, num_inh=0, pos_constraint=False, # optional parameters
            **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        # Check that input_dims are four-dimensional: otherwise assume filter_dimension missing, and lags
        if input_dims is not None:
            if len(input_dims) == 2:
                input_dims = [1] + input_dims + [1]
            elif len(input_dims) == 3:
                input_dims = [1] + input_dims

        return {
            'layer_type': 'normal',
            'input_dims': input_dims,
            'num_filters': num_filters,
            'NLtype': NLtype,
            'norm_type': norm_type,
            'pos_constraint': pos_constraint,
            'num_inh': num_inh,
            'bias': bias,
            'weights_initializer': 'xavier_uniform',
            'bias_initializer': 'zeros',
            'initialize_center': initialize_center,
            'reg_vals': {}
        }

