
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



class NDNlayer(nn.Module):

    # def __repr__(self):
    #     s = super().__repr__()
    #     s += 'NDNlayer'

    def __init__(self, layer_params, reg_vals=None):
        super(NDNlayer, self).__init__()

        assert not layer_params['conv'], "layer_params error: This is not a conv layer."
        self.input_dims = layer_params['input_dims']
        self.num_filters = layer_params['num_filters']
        self.filter_dims = layer_params['filter_dims']
        self.output_dims = layer_params['output_dims']
        self.NLtype = layer_params['NLtype']
        
        # Will want to replace this with Jake's fancy function
        if self.NLtype == 'lin':
            self.NL = None
        elif self.NLtype == 'relu':
            self.NL = F.relu
        elif self.NLtype == 'quad':
            self.NL = F.square  # this doesn't exist: just apply exponent?
        elif self.NLtype == 'softplus':
            self.NL = F.softplus
        elif self.NLtype == 'tanh':
            self.NL = F.tanh
        elif self.NLtype == 'sigmoid':
            self.NL = F.sigmoid
        else:
            print("Nonlinearity undefined.")
            # return error

        #self.in_features = in_features
        # self.flatten = nn.Flatten()
        
        self.shape = tuple([np.prod(self.input_dims), self.num_filters])
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

    def forward(self, x):
        w = self.weights
        if self.pos_constraint:
            w = torch.maximum(w, self.minval)

        # Add normalization
        x = torch.matmul(x, w) + self.bias
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
    