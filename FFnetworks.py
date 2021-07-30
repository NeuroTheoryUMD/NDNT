import numpy as np
import torch
from torch import nn

from copy import deepcopy
#from .regularization import reg_setup_ffnet
from NDNlayer import *

LayerTypes = {
    'normal': NDNlayer,
    'conv': ConvLayer
}

NLtypes = {
    'lin': None,
    'relu': nn.ReLU(),
    'square': torch.square, # this doesn't exist: just apply exponent?
    'softplus': nn.Softplus(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid()
    }

class FFnetwork(nn.Module):

    #def __repr__(self):
    #    s = super().__repr__()
    #    # Add information about module to print out

    def __init__(self, ffnet_params):
        """ffnet_params is a dictionary constructed by other utility functions
        reg_params is a dictionary of reg type and list of values for each layer,
        i.e., {'d2xt':[None, 0.2], 'l1':[1e-4,None]}"""
        super(FFnetwork, self).__init__()

        # Format and record inputs into ffnet
        self.layer_list = deepcopy(ffnet_params['layer_list'])
        self.layer_types = deepcopy(ffnet_params['layer_types'])
        self.xstim_n = ffnet_params['xstim_n']
        self.ffnets_in = deepcopy(ffnet_params['ffnet_n'])
        self.input_dims = deepcopy(ffnet_params['input_dims'])
        #self.conv = ffnet_params['conv']    # I don't think this does anything

        num_layers = len(self.layer_list)

        # Check that first layer has matching input dims (to FFnetwork)
        if self.layer_list[0]['input_dims'] is None:
            self.layer_list[0]['input_dims'] = ffnet_params['input_dims']

        # Process regularization into layer-specific list. Will save at this level too
        
        reg_params = self.__reg_setup_ffnet( ffnet_params['reg_list'] )
        # can be saved, but difficult to update. just save reg vals within layers

        # Make each layer as part of an array
        self.layers = nn.ModuleList()
        for ll in range(num_layers):
            layer_type = self.layer_types
            self.layers.append(
                LayerTypes[self.layer_types[ll]](self.layer_list[ll], reg_vals=reg_params[ll]) )
    # END FFnetwork.__init__
 
    def forward(self, x):        
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __reg_setup_ffnet(self, reg_params=None):
        # Set all default values to none
        num_layers = len(self.layer_list)
        layer_reg_list = []
        for nn in range(num_layers):
            #layer_reg_list.append(deepcopy(_allowed_reg_types))
            layer_reg_list.append(deepcopy({}))  # only put regs in that are there

        # Set specific regularization
        if reg_params is not None:
            for kk, vv in reg_params.items():
                if not isinstance(vv, list):
                    vv = [vv]
                if len(vv) > num_layers:
                    print("Warning: reg params too long for", kk)
                for nn in range(np.minimum(num_layers, len(vv))):
                    layer_reg_list[nn][kk] = vv[nn]
        return layer_reg_list

    def prepare_regularization(self):
        """Makes regularization modules for training"""
        for layer in self.layers:
            layer.reg.build_reg_modules()

    def compute_reg_loss(self):
        rloss = 0
        for layer in self.layers:
            rloss += layer.compute_reg_loss()
        return rloss


class Readout(nn.Module):
    """
    A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
    and covariance of that Gaussian are learned.

    Args:
        in_shape (list, tuple): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]. Default: 0.1
        init_sigma (float): The standard deviation of the Gaussian with `init_sigma` when `gauss_type` is
            'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the
            covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        gauss_type (str): Which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
        shifter (dict): Parameters for a predictor of shfiting grid locations. Has to have a form like
                        {
                        'hidden_layers':1,
                        'hidden_features':20,
                        'final_tanh': False,
                        }
"""
    #def __repr__(self):
    #    s = super().__repr__()
    #    # Add information about module to print out

    def __init__(self, ffnet_params):
        """This essentially used the constructor for Point1DGaussian, with dicationary input:"""
        super(Readout, self).__init__()

        # pytorch lightning helper to save all hyperparamters
        #self.save_hyperparameters()
        self.input_dims = deepcopy(ffnet_params['input_dims'])
        self.ffnets_in = deepcopy(ffnet_params['ffnet_n'])
        self.outdims = ffnet_params['num_cells']
        self.shifter = ffnet_params['shifter_network']
        # Assume 1-d for this one
        num_space_dims = 1

        # sample a different location per example
        self.batch_sample = ffnet_params['batch_sample']
        # constrain feature vector to be positive
        self.constrain_positive = ffnet_params['constrain_positive']

        self.init_mu_range = ffnet_params['init_mu_range']
        self.init_sigma = ffnet_params['init_sigma']
        if self.init_mu_range > 1.0 or self.init_mu_range <= 0.0 or self.init_sigma <= 0.0:
            raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")

        # position grid shape
        self.grid_shape = (1, self.outdims, 1, num_space_dims)
        self.sigma_shape = (1, self.outdims, 1, num_space_dims)
        # initialize means and spreads
        self._mu = Parameter(torch.Tensor(*self.grid_shape))  # mean location of gaussian for each neuron
        self.sigma = Parameter(torch.Tensor(*self.sigma_shape))  # standard deviation for gaussian for each neuron

        self.initialize_features()
        
        if ffnet_params['bias']:
            self.bias = Parameter(torch.Tensor(self.outdims))
            #self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        # Not clear yet how best to handle regularization -- leaving current and ability to pass in
        self.register_buffer("regvalplaceholder", torch.zeros((1,2)))
        self.reg_list = deepcopy(ffnet_params['reg_list'])

        self.gauss_type = ffnet_params['gauss_type=']
        self.align_corners = ffnet_params['align_corners']
        self.act_func = ffnet_params['act_func']
    
        self.initialize()

    @property
    def features(self):
        ## WHAT DOES THIS FUNCTION DO? ###############################
        ## looks like it modifies param weights (_features is paramters, but applies constraints if set)
        if self.constrain_positive:
            feat = F.relu(self._features)
        else:
            feat = self._features

        if self._shared_features:
            feat = self.scales * feat[..., self.feature_sharing_index]
        
        return feat

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    def feature_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if self._original_features:
            if average:
                return self._features.abs().mean()
            else:
                return self._features.abs().sum()
        else:
            return 0

    @property
    def mu(self):
        return self._mu

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
            self.sigma.clamp(min=0)  # sigma/variance i    s always a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]
                
        sample = self.training if sample is None else sample   #### NOT CLEAR WHAT THIS DOES (or how it does....)
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

        grid2d = norm.new_zeros(*(grid_shape[:3]+(2,)))  # for consistency and CUDA capability
        grid2d[:,:,:,0] = (norm * self.sigma + self.mu).clamp(-1,1).squeeze(-1)
        return grid2d
        #return (norm * self.sigma + self.mu).clamp(-1,1) # grid locations in feature space sampled randomly around the mean self.mu


    def initialize(self):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """

        self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)  # random initializations uniformly spread....
        self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)

        #self._features.data.fill_(1 / self.in_shape[0])
        self._features.data.fill_(1 / self.input_dims[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def initialize_features(self, match_ids=None):
        import numpy as np
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        #c, w, h = self.in_shape
        c, w, h = self.input_dims[:3]
        self._original_features = True
        if match_ids is not None:
            raise ValueError(f'match_ids to combine across session "{match_ids}" is not implemented yet')
        else:
            self._features = Parameter(torch.Tensor(1, c, 1, self.outdims))  # feature weights for each channel of the core
            self._shared_features = False
    

    def forward(self, x, sample=None, shift=None, out_idx=None):
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

        #c_in, w_in, h_in = self.in_shape
        c_in, w_in, h_in = self.input_dims[:3]
        #print(c, w, h)
        #print(c_in, w_in, h_in)
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")
        feat = self.features  # this is the filter weights for each unit
        feat = feat.reshape(1, c, self.outdims)
        bias = self.bias
        outdims = self.outdims


        if self.batch_sample:
            # sample the grid_locations separately per sample per batch
            grid = self.sample_grid(batch_size=N, sample=sample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all sample in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(N, outdims, 1, 1)

        if shift is not None:
            # shifter is run outside the readout forward
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners)
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        
        y = NLtypes[self.act_func](y)
        return y

    def prepare_regularization(self):
        return

    def compute_reg_loss(self):
        return 0.0

    def regularizer(self):
        if self.shifter is None:
            out = 0
        else:
            out = self.shifter(self.regvalplaceholder).abs().sum()*10
        # enforce the shifter to have 0 shift at 0,0 in
        return out

    def __repr__(self):
        """
        returns a string with setup of this model
        """
        c, w, h = self.input_dims[:3] #self.in_shape
        r = self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        if self.shifter is not None:
            r += " with shifter"

        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r

