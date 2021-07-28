
import numpy as np
import torch
from torch import nn

from torch.nn import functional as F

from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
#from torch.nn.common_types import _size_2_t, _size_3_t # for conv2,conv3 default
#from torch.nn.modules.utils import _triple # for posconv3

from regularization import Regularization
from copy import deepcopy

NLtypes = {
    'lin': None,
    'relu': nn.ReLU(),
    'square': torch.square, # this doesn't exist: just apply exponent?
    'softplus': nn.Softplus(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid()
    }

class NDNlayer(nn.Module):
    """This is meant to be a base layer-class and overloaded to create other types of layers.
    As a result, it is meant to establish the basic components of a layer, which includes parameters
    corresponding to weights (num_filters, filter_dims), biases, nonlinearity ("NLtype"), and regularization. 
    Other operations that are performed in the base-layer would be pos_constraint (including setting the
    eimask), and normalization. All aspects can be overloaded, although if so, there may be no need
    to inherit NDNLayer as a base class."""

    # def __repr__(self):
    #     s = super().__repr__()
    #     s += 'NDNlayer'

    def __init__(self, layer_params, reg_vals=None):
        super(NDNlayer, self).__init__()

        #assert not layer_params['conv'], "layer_params error: This is not a conv layer."
        self.input_dims = layer_params['input_dims']
        self.num_filters = layer_params['num_filters']
        self.filter_dims = layer_params['filter_dims']
        self.output_dims = layer_params['output_dims']
        self.NLtype = layer_params['NLtype']
        
        # Was this implemented correctly? Where should NLtypes (the dictionary) live?
        if self.NLtype in NLtypes:
            self.NL = NLtypes[self.NLtype]
        else:
            print("Nonlinearity undefined.")
            # return error

        #self.in_features = in_features
        # self.flatten = nn.Flatten()
        
        self.shape = tuple([np.prod(self.filter_dims), self.num_filters])
        self.norm_type = layer_params['norm_type']
        self.pos_constraint = layer_params['pos_constraint']

        # Make layer variables
        self.weights = Parameter(torch.Tensor(size=self.shape))
        if layer_params['bias']:
            self.bias = Parameter(torch.Tensor(self.num_filters))
        else:
            self.register_buffer('bias', torch.zeros(self.num_filters))

        if self.pos_constraint:
            self.register_buffer("minval", torch.tensor(0.0))
            # Does this apply to both weights and biases? Should be separate?
            # How does this compare without explicit constraint? Maybe better, maybe worse...

        self.reg = Regularization( filter_dims=self.filter_dims, vals=reg_vals)

        if layer_params['num_inh'] == 0:
            self.ei_mask = None
        else:
            self.register_buffer('ei_mask', torch.ones(self.num_filters))  
            self.ei_mask[-(layer_params['num_inh']+1):0] = -1

        self.reset_parameters( layer_params['initializer'] )
    # END NDNlayer.__init__

    def reset_parameters(self, initializer=None) -> None:
        
        # Default initializer, although others possible
        if initializer is None:
            initializer = 'uniform'

        init.kaiming_uniform_(self.weights, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)          

    def prepare_weights(self):
        # Apply positive constraints
        if self.pos_constraint:
            w = torch.maximum(self.weights, self.minval)
        else:
            w = self.weights
        # Add normalization
        if self.norm_type > 0: # so far just standard filter-specific normalization
            w = F.normalize( w, dim=0 )
        return w

    def forward(self, x):
        # Pull weights and process given pos_constrain and normalization conditions
        w = self.prepare_weights()

        # Simple linear processing and bias
        x = torch.matmul(x, w) + self.bias

        # Nonlinearity
        if self.NL is not None:
            x = self.NL(x)
        return x 
        
    def compute_reg_loss(self):
        return self.reg.compute_reg_loss(self.weights)

    def get_weights(self, to_reshape=True):
        ws = self.weights.detach().cpu().numpy()
        if to_reshape:
            num_filts = ws.shape[-1]
            return ws.reshape(self.filter_dims + [num_filts]).squeeze()
        else:
            return ws.squeeze()
    ## END NDNLayer class

class ConvLayer(NDNlayer):
    """Making this handle 1 or 2-d although could overload to make 1-d from the 2-d""" 
    def __repr__(self):
        s = super().__repr__()
        s += 'Convlayer'

    def __init__(self, layer_params, reg_vals=None):
        # All parameters of filter (weights) should be correctly fit in layer_params
        super(ConvLayer, self).__init__(layer_params, reg_vals=reg_vals)
        if layer_params['stride'] is None:
            self.stride = 1   
        else: 
            self.stride = layer_params['stride']
        if layer_params['dilation'] is None:
            self.dilation = 1
        else:
            self.dilation = layer_params['dilation']
        # Check if 1 or 2-d convolution required
        self.is1D = (self.input_dims[2] == 1)
        if self.stride > 1:
            print('Warning: Manual padding not yet implemented when stride > 1')
            self.padding = 0
        else:
            #self.padding = 'same' # even though in documentation, doesn't seem to work
            # Do padding by hand
            w = self.filter_dims[1]
            if w%2 == 0:
                print('warning: only works with odd kernels, so far')
            self.padding = (w-1)//2 # this will result in same/padding

        # Combine filter and temporal dimensions for conv -- collapses over both
        self.folded_dims = self.input_dims[0]*self.input_dims[3]
        self.num_outputs = np.prod(self.output_dims)
        # QUESTION: note that this and other constants are used as function-arguments in the 
        # forward, but the numbers are not combined directly. Is that mean they are ok to be
        # numpy, versus other things? (check...) 

    def forward(self, x):

        # Reshape weight matrix and inputs (note uses 4-d rep of tensor before combinine dims 0,3)
        s = x.reshape([-1]+self.input_dims).permute(0,1,4,2,3)
        w = self.prepare_weights().reshape(self.filter_dims+[-1]).permute(4,0,3,1,2)
        # puts output_dims first, as required for conv
        # Collapse over irrelevant dims for dim-specific convs
        if self.is1D:
            y = F.conv1d(
                torch.reshape( s, (-1, self.folded_dims, self.input_dims[1]) ),
                torch.reshape( w, (-1, self.folded_dims, self.filter_dims[1]) ), 
                bias=self.bias,
                stride=self.stride, padding=self.padding, dilation=self.dilation)
        else:
            y = F.conv2d(
                torch.reshape( s, (-1, self.folded_dims, self.input_dims[1], self.input_dims[2]) ),
                torch.reshape( w, (-1, self.folded_dims, self.input_dims[1], self.input_dims[2]) ), 
                bias=self.bias,
                stride=self.stride, padding=self.padding, dilation=self.dilation)

        y = torch.reshape(y, (-1, self.num_outputs))
        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)
        return y

class Readout(nn.Module):
    def initialize(self, *args, **kwargs):
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(
                lambda x: not x.startswith("_") and ("gamma" in x or "pool" in x or "positive" in x), dir(self)
        ):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"
