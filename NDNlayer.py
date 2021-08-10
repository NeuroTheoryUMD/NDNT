
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

        self.input_dims = layer_params['input_dims']
        self.num_filters = layer_params['num_filters']
        # In input_dims is determined by network (and not filled in previously), this fills in with input_dims
        if layer_params['filter_dims'] is None:
            layer_params['filter_dims'] = deepcopy(layer_params['input_dims'])
        self.filter_dims = layer_params['filter_dims']
        self.output_dims = layer_params['output_dims']
        self.NLtype = layer_params['NLtype']
        self.norm_type = layer_params['norm_type']
        self.pos_constraint = layer_params['pos_constraint']
        
        # Was this implemented correctly? Where should NLtypes (the dictionary) live?
        if self.NLtype in NLtypes:
            self.NL = NLtypes[self.NLtype]
        else:
            print("Nonlinearity undefined.")
            # return error

        self.shape = tuple([np.prod(self.filter_dims), self.num_filters])

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

    def preprocess_weights(self):
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
        w = self.preprocess_weights()

        # Simple linear processing and bias
        x = torch.matmul(x, w) + self.bias

        # Nonlinearity
        if self.NL is not None:
            x = self.NL(x)
        return x 
        
    def compute_reg_loss(self):
        #return self.reg.compute_reg_loss(self.weights)
        return self.reg.compute_reg_loss(self.preprocess_weights())

    def get_weights(self, to_reshape=True):
        ws = self.preprocess_weights().detach().cpu().numpy()
        if to_reshape:
            num_filts = ws.shape[-1]
            return ws.reshape(self.filter_dims + [num_filts]).squeeze()
        else:
            return ws.squeeze()

    def list_params(self):
        for nm, pp in self.named_parameters(recurse=False):
            if pp.requires_grad:
                print("      %s:"%nm, pp.size())
            else:
                print("      NOT FIT: %s:"%nm, pp.size())

    def set_params(self, name=None, val=None ):
        """Turn fitting for named params on or off. If name is none, do for whole layer."""
        assert isinstance(val, bool), 'val must be set.'
        for nm, pp in self.named_parameters(recurse=False):
            if name is None:
                pp.requires_grad = val
            elif nm == name:
                pp.requires_grad = val
    ## END NDNLayer class

class ConvLayer(NDNlayer):
    """Making this handle 1 or 2-d although could overload to make 1-d from the 2-d""" 
    # def __repr__(self):
    #     s = super().__repr__()
    #     s += 'Convlayer'

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
            # Do padding by hand -- will want to generalize this for two-ds
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
        w = self.preprocess_weights().reshape(self.filter_dims+[-1]).permute(4,0,3,1,2)
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
                torch.reshape( w, (-1, self.folded_dims, self.filter_dims[1], self.filter_dims[2]) ), 
                bias=self.bias,
                stride=self.stride, padding=self.padding, dilation=self.dilation)

        y = torch.reshape(y, (-1, self.num_outputs))
        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)
        return y


class DivNormLayer(NDNlayer):
    """Jake's div normalization implementation: not explicitly convolutional""" 

    # def __repr__(self):
    #     s = super().__repr__()
    #     s += 'DivNormLayer'

    def __init__(self, layer_params, reg_vals=None):
        # number of filters (and size of filters) is set by channel dims on the input
        num_filters = layer_params['input_dims'][0]
        layer_params['num_filters']  = num_filters
        layer_params['filter_dims'] = [num_filters, 1, 1, 1]
        layer_params['pos_constraint'] = True # normalization weights must be positive
        super(DivNormLayer, self).__init__(layer_params, reg_vals=reg_vals)

        self.output_dims = self.input_dims
        self.num_outputs = np.prod(self.output_dims)

    def forward( self, x):
        # Pull weights and process given pos_constrain and normalization conditions
        w = self.preprocess_weights()

        # Nonlinearity (apply first)
        if self.NL is not None:
            x = self.NL(x)

        x = x.reshape([-1] + self.input_dims)

        # Linear processing to create divisive drive
        xdiv = torch.einsum('nc...,ck->nk...', x, w)

        if len(x.shape)==2:
            xdiv = xdiv + self.bias[None,:]
        elif len(x.shape)==3:
            xdiv = xdiv + self.bias[None,:,None] # is 1D convolutional
        elif len(x.shape)==4:
            xdiv = xdiv + self.bias[None,:,None,None] # is 2D convolutional
        elif len(x.shape)==5:
            xdiv = xdiv + self.bias[None,:,None,None,None] # is 2D convolutional
        else:
            raise NotImplementedError('DivNormLayer only supports 2D, 3D, and 4D tensors')
            
        # apply divisive drive
        x = x / xdiv.clamp_(0.001) # divide
        
        x = x.reshape((-1, self.num_outputs))
        return x


class ReadoutLayer(NDNlayer):
    def initialize(self, *args, **kwargs):
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def __repr__(self):
        s = super().__repr__()
        #s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(
                lambda x: not x.startswith("_") and ("gamma" in x or "pool" in x or "positive" in x), dir(self)
        ):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"

    def __init__(self, layer_params, shifter_present=False, reg_vals=None):

        # Make sure filter_dims is filled in, and to single the spatial filter dimensions
        if layer_params['filter_dims'] is None:
            layer_params['filter_dims'] = deepcopy(layer_params['input_dims'])
        layer_params['filter_dims'][1:3] = [1,1]

        super(ReadoutLayer,self).__init__(layer_params)
        assert layer_params['filter_dims'][3] == 1, 'Cant handle temporal filter dims here, yet.'

        # Determine whether one- or two-dimensional readout
        if layer_params['input_dims'][2] == 1:
            self.num_space_dims = 1
        else:
            self.num_space_dims = 2
        # pytorch lightning helper to save all hyperparamters
        #self.save_hyperparameters()

        # Additional readout parameters (below)
        if self.pos_constraint:
            self.register_buffer("minval", torch.tensor(0.0))
            # Does this apply to both weights and biases? Should be separate?

        # self.flatten = nn.Flatten()
        self.batch_sample = layer_params['batch_sample']
        self.init_mu_range = layer_params['init_mu_range']
        self.init_sigma = layer_params['init_sigma']
        if self.init_mu_range > 1.0 or self.init_mu_range <= 0.0 or self.init_sigma <= 0.0:
            raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")
        
        if self.num_space_dims == 1:
            self.gauss_type = 'isotropic'  # only 1-d to sample sigma in
        else:
            self.gauss_type = layer_params['gauss_type']
        self.align_corners = layer_params['align_corners']

        # position grid shape
        self.grid_shape = (1, self.num_filters, 1, self.num_space_dims)
        if self.gauss_type == 'full':
            self.sigma_shape = (1, self.num_filters, 2, 2)
        elif self.gauss_type == 'uncorrelated':
            self.sigma_shape = (1, self.num_filters, 1, 2)
        elif self.gauss_type == 'isotropic':
            self.sigma_shape = (1, self.num_filters, 1, 1)
        else:
            raise ValueError(f'gauss_type "{self.gauss_type}" not known')

        # initialize means and spreads
        self._mu = Parameter(torch.Tensor(*self.grid_shape))  # mean location of gaussian for each neuron
        self.sigma = Parameter(torch.Tensor(*self.sigma_shape))  # standard deviation for gaussian for each neuron

        self.initialize_spatial_mapping()
        # Not needed anymore: automatically handled by NDNlayer
        #self.initialize_features()
        
        # Not clear yet how best to handle regularization -- leaving current and ability to pass in
        #self.register_buffer("regvalplaceholder", torch.zeros((1,2)))

    @property
    def features(self):
        ## WHAT DOES THIS FUNCTION DO? ###############################
        ## looks like it modifies param weights (_features is paramters, but applies constraints if set)
        if self.pos_constraint:
            feat = F.relu(self.weights)
        else:
            feat = self.weights
        
        return feat

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    @property
    def mu(self):
        return self._mu

    def initialize_spatial_mapping(self):
        """
        Initializes the mean, and sigma of the Gaussian readout 
        """
        self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)  # random initializations uniformly spread....
        if self.gauss_type != 'full':
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)

    def sample_grid(self, batch_size, sample=None):
        """
        Returns the grid locations from the core by sampling from a Gaussian distribution
        DAN: more specifically, it returns sampled positions for each batch over all elements, given mus and sigmas
        DAN: if not 'sample', it just gives mu back (so testing has no randomness)
        DAN: this is some funny bit of code, but don't think I need to touch it past what I did
        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
        """
        with torch.no_grad():
            self.mu.clamp(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
            if self.gauss_type != 'full':
                self.sigma.clamp(min=0)  # sigma/variance is always a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]
                
        sample = self.training if sample is None else sample
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

        if self.num_space_dims == 1:
            # this is dan's 1-d kluge code -- necessary because grid_sampler has to have last dimension of 2. maybe problem....
            grid2d = norm.new_zeros(*(grid_shape[:3]+(2,)))  # for consistency and CUDA capability
            grid2d[:,:,:,0] = (norm * self.sigma + self.mu).clamp(-1,1).squeeze(-1)
            return grid2d
            #return (norm * self.sigma + self.mu).clamp(-1,1) # this needs second dimension
        else:
            if self.gauss_type != 'full':
                return (norm * self.sigma + self.mu).clamp(-1,1) # grid locations in feature space sampled randomly around the mean self.mu
            else:
                return (torch.einsum('ancd,bnid->bnic', self.sigma, norm) + self.mu).clamp_(-1,1) # grid locations in feature space sampled randomly around the mean self.mu

    def forward(self, x, sample=None, shift=None):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted

        Returns:
            y: neuronal activity
        """
        x = x.reshape([-1]+self.input_dims[:3])

        N, c, w, h = x.size()   # N is number of time points -- note that its full dimensional....
        c_in, w_in, h_in = self.input_dims[:3]
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")
        
        feat = self.features  # this is the filter weights for each unit
        feat = feat.reshape(1, c, self.num_filters)
        bias = self.bias
        outdims = self.num_filters

        if self.batch_sample:
            # sample the grid_locations separately per sample per batch
            grid = self.sample_grid(batch_size=N, sample=sample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all sample in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(N, outdims, 1, self.num_space.dims)
        
        if shift is not None:
            # shifter is run outside the readout forward
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners)

        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        
        if self.NL is not None:
            y = self.NL(y)
        return y

    def set_readout_locations(self, locs):
        """This hasn't been tested yet, but should be self-explanatory"""
        if len(locs.shape) == 1:
            locs = np.expand_dims(locs,1)
        NC, num_dim = locs.shape
        assert num_dim == self.num_space_dims, 'Incorrect number of spatial dimensions'
        assert NC == self.num_filters, 'Incorrect number of neuron positions.'
        self._mu = deepcopy(locs)

    def get_readout_locations(self):
        """Currently returns center location and sigmas, as list"""
        return self.mu.detach().cpu().numpy().squeeze(), self.sigma.detach().cpu().numpy().squeeze()

    # def __repr__(self):
    #     """
    #     returns a string with setup of this model
    #     """
    #     c, w, h = self.input_dims[:3] #self.in_shape
    #     r = self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.num_filters) + ")"
    #     if self.bias is not None:
    #         r += " with bias"
    #     if self.shifter is not None:
    #         r += " with shifter"

    #     for ch in self.children():
    #         r += "  -> " + ch.__repr__() + "\n"
    #     return r


