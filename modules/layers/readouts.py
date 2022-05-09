import torch
from torch.nn import Parameter
from torch.nn import functional as F

import numpy as np

from copy import deepcopy
from .ndnlayer import NDNLayer

class ReadoutLayer(NDNLayer):

    def __init__(self, 
            input_dims=None,
            num_filters=None,
            filter_dims=None, 
            batch_sample=True,
            init_mu_range=0.1,
            init_sigma=0.2,
            gauss_type: str='uncorrelated', # 'isotropic', 'uncorrelated', or 'full'
            align_corners=False,
            mode='bilinear',  # 'nearest' is also possible
            **kwargs):

        assert input_dims is not None, "ReadoutLayer: Must specify input_dims"
        assert num_filters is not None, "ReadoutLayer: Must specify num_filters"

        # Make sure filter_dims is filled in, and to single the spatial filter dimensions
        if filter_dims is None:
            filter_dims = deepcopy(input_dims)
        
        filter_dims[1:3] = [1,1]
        
        assert filter_dims[3] == 1, 'Cant handle temporal filter dims here, yet.'
        
        super().__init__(input_dims,
            num_filters,
            filter_dims=filter_dims,
            **kwargs)
        
        # Determine whether one- or two-dimensional readout
        if input_dims[2] == 1:
            self.num_space_dims = 1
        else:
            self.num_space_dims = 2

        # Additional readout parameters (below)
        if self.pos_constraint:
            self.register_buffer("minval", torch.tensor(0.0))
            # Does this apply to both weights and biases? Should be separate?

        # self.flatten = nn.Flatten()
        self.batch_sample = batch_sample
        self.sample_mode = mode

        self.init_mu_range = init_mu_range
        self.init_sigma = init_sigma

        if self.init_mu_range > 1.0 or self.init_mu_range <= 0.0 or self.init_sigma <= 0.0:
            raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")
        
        if self.num_space_dims == 1:
            self.gauss_type = 'isotropic'  # only 1-d to sample sigma in
        else:
            self.gauss_type = gauss_type
        self.align_corners = align_corners

        # position grid shape
        self.grid_shape = (1, self.num_filters, 1, self.num_space_dims)
        if self.gauss_type == 'full':
            #self.sigma_shape = (1, self.num_filters, 2, 2)
            self.sigma_shape = (self.num_filters, 2)
        elif self.gauss_type == 'uncorrelated':
            #self.sigma_shape = (1, self.num_filters, 1, 2)
            self.sigma_shape = (self.num_filters, 2)
        elif self.gauss_type == 'isotropic':
            #self.sigma_shape = (1, self.num_filters, 1, 1)
            self.sigma_shape = (self.num_filters, 1)
        else:
            raise ValueError(f'gauss_type "{self.gauss_type}" not known')

        # initialize means and spreads
        map_size = (self.num_filters, self.num_space_dims)
        self._mu = Parameter(torch.Tensor(*map_size))
        self.sigma = Parameter(torch.Tensor(*self.sigma_shape))

        self.initialize_spatial_mapping()
    # END ReadoutLayer.__init__

    @property
    def features(self):

        if self.pos_constraint:
            #feat = F.relu(self.weight)
            feat = F.square(self.weight)
        else:
            feat = self.weight
        
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
        # if self.gauss_type != 'full':
        #     self.sigma.data.fill_(self.init_sigma)
        # else:
        #self.sigma.data.uniform_(0, self.init_sigma)
        self.sigma.data.fill_(self.init_sigma)
        
        #self.weight.data.fill_(1 / self.input_dims[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

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

        grid_shape = (batch_size,) + self.grid_shape[1:3]
                
        sample = self.training if sample is None else sample
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

        if self.num_space_dims == 1:
            # this is dan's 1-d kluge code -- necessary because grid_sampler has to have last dimension of 2. maybe problem....
            grid2d = norm.new_zeros(*(grid_shape[:3]+(2,)))  # for consistency and CUDA capability
            #grid2d[:,:,:,0] = (norm * self.sigma + self.mu).clamp(-1,1).squeeze(-1)
            ## SEEMS dim-4 HAS TO BE IN THE SECOND DIMENSION (RATHER THAN FIRST)
            grid2d[:,:,:,1] = (norm * self.sigma[None, :] + self.mu[None, :]).clamp(-1,1)
            return grid2d
            #return (norm * self.sigma + self.mu).clamp(-1,1) # this needs second dimension
        else:
            if self.gauss_type != 'full':
                # grid locations in feature space sampled randomly around the mean self.mu
                #return (norm * self.sigma + self.mu).clamp(-1,1)
                return (norm[:,:,:,None] * self.sigma[None, :, None, :] + self.mu[None, :, None, :]).clamp(-1,1) 
            else:
                return (torch.einsum('ancd,bnid->bnic', self.sigma[None, :, None, :], norm) + self.mu[None, :, None, :]).clamp_(-1,1) # grid locations in feature space sampled randomly around the mean self.mu
    # END ReadoutLayer.sample_grid() 

    def passive_readout(self):
        """This will not fit mu and std, but set them to zero. It will pass identities in,
        so number of input filters must equal number of readout units"""

        assert self.filter_dims[0] == self.output_dims[0], "Must have #filters = #output units."
        # Set parameters for default readout
        self.sigma.data.fill_(0)
        self._mu.data.fill_(0)
        self.weight.data.fill_(0)
        for nn in range(self.num_filters):
            self.weight.data[nn,nn] = 1

        self.set_parameters(val=False)
        

    def forward(self, x, shift=None):
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
            grid = self.sample_grid(batch_size=N, sample=self.training)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all sample in the batch
            grid = self.sample_grid(batch_size=1, sample=self.training).expand(N, outdims, 1, self.num_space.dims)
        
        if shift is not None:
            # shifter is run outside the readout forward
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, mode=self.sample_mode, align_corners=self.align_corners, padding_mode='border')
        # note I switched this from the default 'zeros' so that it wont try to go past the border

        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        
        if self.NL is not None:
            y = self.NL(y)
        
        return y
    # END ReadoutLayer.forward

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

    def passive_readout(self):
        """This will not fit mu and std, but set them to zero. It will pass identities in,
        so number of input filters must equal number of readout units"""

        assert self.filter_dims[0] == self.output_dims[0], "Must have #filters = #output units."
        # Set parameters for default readout
        self.sigma.data.fill_(0)
        self._mu.data.fill_(0)
        self.weight.data.fill_(0)
        for nn in range(self.num_filters):
            self.weight.data[nn,nn] = 1

        self.set_parameters(val=False)

    @classmethod
    def layer_dict(cls, NLtype='softplus', **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'readout'
        # Added arguments
        Ldict['batch_sample'] = True
        Ldict['init_mu_range'] = 0.1
        Ldict['init_sigma'] = 0.2
        Ldict['gauss_type'] = 'uncorrelated'
        Ldict['align_corners'] = False
        Ldict['mode'] = 'bilinear'  # also 'nearest'
        # Change default -- readouts will usually be softplus
        Ldict['NLtype'] = NLtype

        return Ldict
    # END [classmethod] ReadoutLayer.layer_dict


class FixationLayer(NDNLayer):
    """
    FixationLayer for fixation 
    """
    def __init__(self,
            input_dims=None,
            num_filters=None,
            filter_dims=None, 
            batch_sample=False,
            #init_mu_range=0.1,
            init_sigma=0.3,
            single_sigma=False,
            #gauss_type: str='uncorrelated', # 'isotropic', 'uncorrelated', or 'full'
            #align_corners=False,
            bias=False,
            NLtype='lin',
            **kwargs):
        """layer weights become the shift for each fixation, sigma is constant over each dimension"""
        #self.num_fixations = layer_params['input_dims'][0]
        assert np.prod(input_dims[1:]) == 1, 'something wrong with fix-layer input_dims'
        filter_dims = deepcopy(input_dims)
        assert num_filters in [1,2], 'incorrect num_filters in fix-layer'
        bias = False

        super().__init__(input_dims=input_dims,
            num_filters=num_filters,
            filter_dims=filter_dims,
            NLtype=NLtype,
            bias=bias,
            **kwargs)

        assert filter_dims[3] == 1, 'Cant handle temporal filter dims here, yet.'

        # Determine whether one- or two-dimensional readout
        self.num_space_dims = num_filters
        
        self.batch_sample = batch_sample
        self.init_sigma = init_sigma
        self.single_sigma = single_sigma
        
        # shared sigmas across all fixations
        if self.single_sigma:
            self.sigmas = Parameter(torch.Tensor(self.num_space_dims)) 
        else:
            # individual sigmas for each fixation 
            self.sigmas = Parameter(torch.Tensor(self.filter_dims[0],1))  
        
        self.sigmas.data.fill_(self.init_sigma)
        self.weight.data.fill_(0.0)
        
        self.sample = False
    # END FixationLayer.__init__

    def forward(self, x, shift=None):
        """
        The input is the sampled fixation-stim
            y: neuronal activity
        """
        # with torch.no_grad():
        # self.weight.clamp(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
        # self.sigmas.clamp(min=0)  # sigma/variance is always a positive quantity

        N = x.shape[0]  # batch size
        # If using X-matrix to extract relevant weights
        # y = (x@self.weight) 
        # use indexing: x is just a list of weight indices

        #y = F.tanh(self.weight[x,:]) #* self.spatial_mults  # deprecated?
        y = torch.tanh(self.weight[x-1,:])   # fix_n

        if self.single_sigma:
            s = self.sigmas**2 
        else:
            s = self.sigmas[x-1]**2

        # this will be batch-size x num_spatial dims (corresponding to mean loc)        
        if self.sample:  # can turn off sampling, even with training
            if self.training:
                # add sigma-like noise around mean locations
                if self.batch_sample:
                    sample_shape = (1,) + (self.num_space_dims,)
                    gaus_sample = y.new(*sample_shape).normal_().repeat(N,1)
                else:
                    sample_shape = (N,) + (self.num_space_dims,)
                    gaus_sample = y.new(*sample_shape).normal_()
                
                y = (gaus_sample * s + y).clamp(-1,1)

        return y
    # END FixationLayer.forward
    
    def initialize(self, *args, **kwargs):
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    @classmethod
    def layer_dict(cls, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'fixation'
        # Added arguments
        Ldict['batch_sample'] = True
        Ldict['init_mu_range'] = 0.1
        Ldict['init_sigma'] = 0.5
        Ldict['single_sigma'] = False
        Ldict['gauss_type'] = 'uncorrelated'
        Ldict['align_corners'] = False

        return Ldict
    # END [classmethod] FixatonLayer.layer_dict