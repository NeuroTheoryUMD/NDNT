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
        self.mu = Parameter(torch.Tensor(*map_size))
        self.sigma = Parameter(torch.Tensor(*self.sigma_shape))

        self.initialize_spatial_mapping()
        self.sample = True
    # END ReadoutLayer.__init__

    @property
    def features(self):

        if self.pos_constraint:
            #feat = F.relu(self.weight)
            feat = self.weight**2
        else:
            feat = self.weight
        
        return feat

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    #@property
    #def mu(self):
    #    return self.mu

    def initialize_spatial_mapping(self):
        """
        Initializes the mean, and sigma of the Gaussian readout 
        """
        self.mu.data.uniform_(-self.init_mu_range, self.init_mu_range)  # random initializations uniformly spread....
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
            grid2d[:,:,:,1] = (norm * self.sigma[None, None, None, :] + self.mu[None, None, None, :]).clamp(-1,1)
            return grid2d
            #return (norm * self.sigma + self.mu).clamp(-1,1) # this needs second dimension
        else:
            if self.gauss_type != 'full':
                # grid locations in feature space sampled randomly around the mean self.mu
                #return (norm * self.sigma + self.mu).clamp(-1,1)
                return (norm[:,:,:,None] * self.sigma[None, :, None, :] + self.mu[None, :, None, :]).clamp(-1,1) 
            else:
                return (torch.einsum('ancd,bnid->bnic', self.sigma[None, :, None, :]**2, norm) + self.mu[None, :, None, :]).clamp_(-1,1) # grid locations in feature space sampled randomly around the mean self.mu
    # END ReadoutLayer.sample_grid() 

    def get_weights(self, to_reshape=True, time_reverse=None, num_inh=None):
        """overloaded to read not use preprocess weights but instead use layer property features"""

        assert time_reverse is None, "  READOUT: time_reverse will not work here."
        assert num_inh is None, "  READOUT: num_inh will not work here."
        ws = self.features
        if to_reshape:
            ws = ws.reshape(self.filter_dims + [self.num_filters]).squeeze()
        else:
            ws = ws.squeeze()

        if self.num_filters == 1:
            # Add singleton dimension corresponding to number of filters
            ws = ws[..., None]
    
        return ws.detach().numpy()

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
            grid = self.sample_grid(batch_size=N, sample=self.sample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all sample in the batch
            grid = self.sample_grid(batch_size=1, sample=self.sample).expand(N, outdims, 1, self.num_space_dims)
        
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
        self.mu = deepcopy(locs)

    def get_readout_locations(self):
        """Currently returns center location and sigmas, as list"""
        return self.mu.detach().cpu().numpy().squeeze(), self.sigma.detach().cpu().numpy().squeeze()    

    def passive_readout(self):
        """This will not fit mu and std, but set them to zero. It will pass identities in,
        so number of input filters must equal number of readout units"""

        assert self.filter_dims[0] == self.output_dims[0], "Must have #filters = #output units."
        # Set parameters for default readout
        self.sigma.data.fill_(0)
        self.mu.data.fill_(0)
        self.weight.data.fill_(0)
        for nn in range(self.num_filters):
            self.weight.data[nn,nn] = 1

        self.set_parameters(val=False)
        self.sample = False

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


class ReadoutLayer3d(ReadoutLayer):

    def __init__(self, 
            input_dims=None,
            **kwargs):

        assert input_dims[3] > 1, "ReadoutLayer3d: should not be using this layer if no 3rd dimension of input"

        # Need to tuck lag dimension into channel-dims so parent constructor makes the right filter shape
        input_dims_mod = deepcopy(input_dims)
        input_dims_mod[0] *= input_dims[3]
        input_dims_mod[3] = 1
                
        super().__init__(input_dims=input_dims_mod, **kwargs)
        self.input_dims = input_dims
        #print('OLD FILTER DIMS', self.filter_dims)
        self.filter_dims = [input_dims[0], input_dims[3], 1, 1]  # making spatial so max_space can be used
        #print('NEW FILTER DIMS', self.filter_dims)
        # Redo regularization with new filter_dims
      
        from ...modules.regularization import Regularization
        reg_vals = self.reg.vals
        self.reg = Regularization( filter_dims=self.filter_dims, vals=reg_vals, num_outputs=self.num_filters )

    # END ReadoutLayer3d.__init__

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
        x = x.reshape([-1]+self.input_dims)  # 3d change -- has extra dimension at end
        N, c, w, h, T = x.size()   # N is number of time points -- note that its full dimensional....
        c *= T
        # get those last filter dims up before spatial
        x = x.permute(0,1,4,2,3).reshape([N, c, w, h]) 

        #c_in, w_in, h_in = self.input_dims[:3]
        #if (c_in, w_in, h_in) != (c, w, h):
        #    raise ValueError("the specified feature map dimension is not the readout's expected input dimension")
        
        feat = self.features  # this is the filter weights for each unit
        feat = feat.reshape(1, -1, self.num_filters)  # 3d change -- this is num_chan x num_angles now

        bias = self.bias
        outdims = self.num_filters

        if self.batch_sample:
            # sample the grid_locations separately per sample per batch
            grid = self.sample_grid(batch_size=N, sample=self.sample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all sample in the batch
            grid = self.sample_grid(batch_size=1, sample=self.sample).expand(N, outdims, 1, self.num_space_dims)
        
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
    # END ReadoutLayer3d.forward

    def passive_readout(self):
        """This might have to be redone for readout3d to take into account extra filter dim"""
        print("WARNING: SCAF3d: this function is not vetted and likely will clunk")
        assert self.filter_dims[0] == self.output_dims[0], "Must have #filters = #output units."
        # Set parameters for default readout
        self.sigma.data.fill_(0)
        self.mu.data.fill_(0)
        self.weight.data.fill_(0)
        for nn in range(self.num_filters):
            self.weight.data[nn,nn] = 1

        self.set_parameters(val=False)

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
        Ldict['layer_type'] = 'readout3d'
        return Ldict
    # END [classmethod] ReadoutLayer3d.layer_dict


class FixationLayer(NDNLayer):
    """
    FixationLayer for fixation 
    """
    def __init__(self,
            num_fixations=None, 
            num_spatial_dims=2,
            #input_dims=None,  # this has to be set to [1,1,1,1] by default
            batch_sample=False,
            fix_n_index=False,
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
        #assert np.prod(input_dims[1:]) == 1, 'something wrong with fix-layer input_dims'
        assert num_fixations is not None, "FIX LAYER: Must set number of fixations"
        assert num_spatial_dims in [1,2], "FIX LAYER: num_space must be set to spatial dimensionality (1 or 2)"
        assert not bias, "FIX LAYER: cannot have bias term"
        bias = False

        super().__init__(
            num_filters = num_spatial_dims,
            filter_dims = [1, num_fixations, 1, 1],
            NLtype = NLtype,
            bias=bias,
            **kwargs)

        # Determine whether one- or two-dimensional readout
        self.num_spatial_dims = num_spatial_dims
        self.num_fixations = num_fixations

        self.batch_sample = batch_sample
        self.init_sigma = init_sigma
        self.single_sigma = single_sigma
        self.fix_n_index = fix_n_index

        # shared sigmas across all fixations
        if self.single_sigma:
            self.sigmas = Parameter(torch.Tensor(self.num_spatial_dims)) 
        else:
            # individual sigmas for each fixation 
            self.sigmas = Parameter(torch.Tensor(self.num_fixations,1))  
        
        self.sigmas.data.fill_(self.init_sigma)
        self.weight.data.fill_(0.0)
        
        self.sample = True
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
        
        # In case x gets passed in with trivial second dim (squeeze)
        if len(x.shape)> 1:
            print('  WARNING: fixations should not have second dimension -- squeeze it')
            x = x.squeeze()-1

        # Actual fixations labeled 1-NFIX
        # Will make by default fixation label '0' lead to no shift
        shift_array = F.pad(self.weight, (0,0,1,0))
        y = torch.tanh(shift_array[x,:])   # fix_n

        if self.single_sigma:
            #s = self.sigmas**2
            s = self.sigmas 
        else:
            #s = self.sigmas[x]**2
            sigma_array = F.pad(self.sigmas, (0,0,1,0))
            s = sigma_array[x]

        # this will be batch-size x num_spatial dims (corresponding to mean loc)        
        if self.sample:  # can turn off sampling, even with training
            if self.training:
                # add sigma-like noise around mean locations
                if self.batch_sample:
                    sample_shape = (1,) + (self.num_spatial_dims,)
                    gaus_sample = y.new(*sample_shape).normal_().repeat(N,1)
                else:
                    sample_shape = (N,) + (self.num_spatial_dims,)
                    gaus_sample = y.new(*sample_shape).normal_()
                
                y = (gaus_sample * s + y).clamp(-1,1)

        return y
    # END FixationLayer.forward
    
    def initialize(self, *args, **kwargs):
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    @classmethod
    def layer_dict(cls, num_fixations=None, num_spatial_dims=2, init_sigma=0.25, input_dims=None, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """
        if input_dims is not None:
            assert np.prod(input_dims) == 1, "FIX DICT: Set filter_dims to #fixations and leave input_dims alone"
        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'fixation'
        # delete standard layer info for purposes of constructor
        del Ldict['input_dims']
        del Ldict['num_filters']
        del Ldict['bias']
        # Added arguments
        Ldict['batch_sample'] = True
        Ldict['fix_n_index'] = False
        Ldict['init_mu_range'] = 0.1
        Ldict['init_sigma'] = init_sigma
        Ldict['single_sigma'] = False
        Ldict['gauss_type'] = 'uncorrelated'
        Ldict['align_corners'] = False
        Ldict['num_fixations'] = num_fixations
        Ldict['num_spatial_dims'] = 2
        Ldict['input_dims'] = [1,1,1,1]
        return Ldict
    # END [classmethod] FixatonLayer.layer_dict