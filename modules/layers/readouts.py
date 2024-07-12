import torch
from torch.nn import Parameter
from torch.nn import functional as F

import numpy as np

from copy import deepcopy
from .ndnlayer import NDNLayer

class ReadoutLayer(NDNLayer):
    """
    ReadoutLayer for spatial readout.
    """
    def __init__(self, 
            input_dims=None,
            num_filters=None,
            filter_dims=None, 
            batch_sample=True,
            init_mu_range=0.1,
            init_sigma=0.2,
            gauss_type: str='isotropic', # 'isotropic', 'uncorrelated', or 'full'
            align_corners=False,
            mode='bilinear',  # 'nearest' is also possible
            **kwargs):
        """
        ReadoutLayer: Spatial readout layer for NDNs.

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            filter_dims: tuple or list of ints, (num_channels, height, width, lags)
            batch_sample: bool, whether to sample grid locations separately per sample per batch
            init_mu_range: float, range for uniform initialization of means
            init_sigma: float, standard deviation for uniform initialization of sigmas
            gauss_type: str, 'isotropic', 'uncorrelated', or 'full'
            align_corners: bool, whether to align corners in grid_sample
            mode: str, 'bilinear' or 'nearest'
        """
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

        # For use with scaffold_level_regularization
        self.level_reg = False
        self._level_reg_val = 0.0  # just so info is available
        self.register_buffer("_scaffold_level_weights", torch.zeros((self.weight.shape[0], 1), dtype=torch.float32))

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
        Initializes the mean, and sigma of the Gaussian readout.

        Args:
            None
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
        More specifically, it returns sampled positions for each batch over all elements, given mus and sigmas
        If 'sample' is false, it just gives mu back (so testing has no randomness)
        This code is inherited and then modified, so different style than most other

        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
        """
        with torch.no_grad():
            #self.mu.clamp(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
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
        """
        Overloaded to read not use preprocess weights but instead use layer property features.
        
        Args:
            to_reshape (bool): whether to reshape the weights to the filter_dims
            time_reverse (bool): whether to reverse the time dimension
            num_inh (int): number of inhibitory units

        Returns:
            ws: weights of the readout layer
        """

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
    # END ReadoutLayer.get_weights()

    def _set_scaffold_reg(self, reg_val=None, scaffold_level_parse=None, level_exponent=1.0):
        """
        Set regularization that applies to scaffold level to force weights towards low levels.
        This function should not be called directly, since will need to pass in non-local scaffold 
        information available only at the NDN level.
        
        Args: 
            reg_val (float or None): regularization weight to set, default None
            scaffold_level_parse (np.array): level weighting with dimension of number of filters in each
                scaffold level [4,25,8,4], but should include total number of weights (in case 4th dim)    
            level_exponent (positive float): how much to weight each level 
        Returns:
            None
         """
        if reg_val is None:
            self.level_reg = False
            self._level_reg_val = 0.0
        else:
            self.level_reg = True
            # Otherwise, construct weighting across filter_input dimensions
            assert level_exponent > 0, "SCAFFOLD REG: level_exponent must be >0"
            assert np.sum(scaffold_level_parse) == self.filter_dims[0], "SCAFFOLD REG: Internal error with scaffold_level_parse"

            self._level_reg_val = reg_val  # just to keep track (should be register_buffer?)

            scaffold_weights = reg_val * np.ones((self.filter_dims[0], self.filter_dims[3]), dtype=np.float32)
            place_holder=0
            for ii in range(len(scaffold_level_parse)):
                scaffold_weights[place_holder+np.arange(scaffold_level_parse[ii]), :] *= ii**level_exponent
                place_holder += scaffold_level_parse[ii]
            self._scaffold_level_weights = torch.tensor( 
                scaffold_weights.reshape([-1]), dtype=torch.float32, device=self.weight.device)
    # END ReadoutLayer._set_scaffold_reg()

    def compute_reg_loss(self):
        """
        Compute the regularization loss for the layer: superceding super, by calling reg_module to do
        this and then adding scaffold weight regularization if needed.

        Args:
            None

        Returns:
            reg_loss: torch.Tensor, regularization loss
        """
        w = self.preprocess_weights()
        regloss = self.reg.compute_reg_loss(w)

        if self.level_reg:
            regloss += torch.sum( w * self._scaffold_level_weights[:, None] ) 
        return regloss
        # ReadoutLayer.compute_reg_loss()

    def forward(self, x, shift=None):
        """
        Propagates the input forwards through the readout.
        
        Args:
            x: input data
            shift (bool): shifts the location of the grid (from eye-tracking data)

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
        """
        This hasn't been tested yet, but should be self-explanatory.
        
        Args:
            locs: locations of the readout units
        """
        if len(locs.shape) == 1:
            locs = np.expand_dims(locs,1)
        NC, num_dim = locs.shape
        assert num_dim == self.num_space_dims, 'Incorrect number of spatial dimensions'
        assert NC == self.num_filters, 'Incorrect number of neuron positions.'
        self.mu = deepcopy(locs)

    def get_readout_locations(self):
        """
        Currently returns center location and sigmas, as list.
        
        Returns:
            mu: center locations of the readout units
            sigma: sigmas of the readout units
        """
        return self.mu.detach().cpu().numpy().squeeze(), self.sigma.detach().cpu().numpy().squeeze()    

    def passive_readout(self):
        """
        This will not fit mu and std, but set them to zero. It will pass identities in,
        so number of input filters must equal number of readout units.
        
        Args:
            None

        Returns:
            None
        """

        assert self.filter_dims[0] == self.output_dims[0], "Must have #filters = #output units."
        # Set parameters for default readout
        self.sigma.data.fill_(0)
        self.mu.data.fill_(0)
        self.weight.data.fill_(0)
        for nn in range(self.num_filters):
            self.weight.data[nn,nn] = 1

        self.set_parameters(val=False)
        self.sample = False

    def fit_mus( self, val=None, sigma=None, verbose=True ):
        """
        Quick function that turns on or off fitting mus/sigmas and toggles sample
        
        Args:
            val (bool): True or False -- must specify
            sigma (float): choose starting sigma value (default no choice)
        """
        assert val is not None, "fit_mus(): must set val to be True or False"
        self.sample = val
        self.set_parameters(val=val, name='mu')
        self.set_parameters(val=val, name='sigma')
        if sigma is not None:
            self.sigma.data.fill_(sigma)
        if verbose:
            if val:
                print("  ReadoutLayer: fitting mus")
            else:
                print("  ReadoutLayer: not fitting mus")
    # END ReadoutLayer.fit_mus()

    def enforce_grid( self ):
        """
        Function that adjusts mus to correspond to precise points on pixel grid
        
        Args:
            None
        """
        L = self.input_dims[1]

        pixels = (L*(self.mu.clone().detach()+1) - 1)/2
        pixels[pixels > (L-1)] = L-1
        pixels[pixels < 0] = 0
        pixels = torch.round(pixels)

        # Convert back to mu values
        self.mu.data = (2*pixels+1)/L - 1
    # END ReadoutLayer.enforce_grid()


    @classmethod
    def layer_dict(cls, NLtype='softplus', mode='bilinear', **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

        Args:
            NLtype: str, type of nonlinearity

        Returns:
            Ldict: dictionary of layer parameters
        """

        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'readout'
        # Added arguments
        Ldict['batch_sample'] = True
        Ldict['init_mu_range'] = 0.1
        Ldict['init_sigma'] = 0.2
        Ldict['gauss_type'] = 'isotropic' #'uncorrelated'
        Ldict['align_corners'] = False
        Ldict['mode'] = mode  # also 'nearest'
        # Change default -- readouts will usually be softplus
        Ldict['NLtype'] = NLtype

        return Ldict
    # END [classmethod] ReadoutLayer.layer_dict


class ReadoutLayer3d(ReadoutLayer):
    """
    ReadoutLayer3d for 3d readout.
    This is a subclass of ReadoutLayer, but with the added dimension of time.
    """
    def __init__(self, 
            input_dims=None,
            **kwargs):
        """
        ReadoutLayer3d: 3d readout layer for NDNs.

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, depth, lags)
        """
        assert input_dims[3] > 1, "ReadoutLayer3d: should not be using this layer if no 3rd dimension of input"

        # Need to tuck lag dimension into channel-dims so parent constructor makes the right filter shape
        input_dims_mod = deepcopy(input_dims)
        input_dims_mod[0] *= input_dims[3]
        input_dims_mod[3] = 1
                
        super().__init__(input_dims=input_dims_mod, **kwargs)
        self.input_dims = input_dims
        #print('OLD FILTER DIMS', self.filter_dims)
        self.filter_dims = [input_dims[0], 1, 1, input_dims[3]]
        #self.filter_dims = [input_dims[0], input_dims[3], 1, 1]  # making spatial so max_space can be used
        #print('NEW FILTER DIMS', self.filter_dims)
        # Redo regularization with new filter_dims
      
        from ...modules.regularization import Regularization
        reg_vals = self.reg.vals
        self.reg = Regularization( filter_dims=self.filter_dims, vals=reg_vals, num_outputs=self.num_filters )
        self.register_buffer('mask', torch.ones( [np.prod(self.filter_dims), self.num_filters], dtype=torch.float32))

    # END ReadoutLayer3d.__init__

    def set_mask( self, mask=None ):
        """
        Sets mask -- instead of plugging in by hand. Registers mask as being set, and checks dimensions.
        Can also use to rest to have trivial mask (all 1s)
        Mask will be numpy, leave mask blank if want to set to default mask (all ones)
        
        Args:
            mask: numpy array of size of filters (filter_dims x num_filters)
        """
        if mask is None:
            self.mask[:,:] = 1.0
        else:
            assert np.prod(mask.shape) == np.prod(self.filter_dims)*self.num_filters
            self.mask = torch.tensor( mask.reshape([-1, self.num_filters]), dtype=torch.float32, device=self.weight.device)
    # END MaskLayer.set_mask()
    
    def forward(self, x, shift=None):
        """
        Propagates the input forwards through the readout

        Args:
            x: input data
            shift (bool): shifts the location of the grid (from eye-tracking data)

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
        
        feat = self.features*self.mask # this is the filter weights for each unit
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
        """
        This might have to be redone for readout3d to take into account extra filter dim.
        """
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

        Args:
            None

        Returns:
            Ldict: dictionary of layer parameters
        """
        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'readout3d'
        return Ldict
    # END [classmethod] ReadoutLayer3d.layer_dict


class FixationLayer(NDNLayer):
    """
    FixationLayer for fixation.
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
        """
        Layer weights become the shift for each fixation, sigma is constant over each dimension.
        
        Args:
            num_fixations: int, number of fixations
            num_spatial_dims: int, number of spatial dimensions
            batch_sample: bool, whether to sample grid locations separately per sample per batch
            fix_n_index: bool, whether to index fixations starting at 1
            init_sigma: float, standard deviation for uniform initialization of sigmas
            single_sigma: bool, whether to use a single sigma for all fixations
            bias: bool, whether to include bias term
            NLtype: str, type of nonlinearity
        """
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
        
        Args:
            x: input data
            shift (bool): shifts the location of the grid (from eye-tracking data)

        Returns:
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
        """
        This will initialize the layer parameters.
        """
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    @classmethod
    def layer_dict(cls, num_fixations=None, num_spatial_dims=2, init_sigma=0.25, input_dims=None, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

        Args:
            num_fixations: int, number of fixations
            num_spatial_dims: int, number of spatial dimensions
            init_sigma: float, standard deviation for uniform initialization of sigmas
            input_dims: tuple or list of ints, (num_channels, height, width, lags)

        Returns:
            Ldict: dictionary of layer parameters
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

class ReadoutLayerQsample(ReadoutLayer3d):
    """
    ReadoutLayerQsample for 3d readout with sampling over angle dimension Q.
    This is a subclass of ReadoutLayer3d.
    """
    def __init__(self, input_dims=None, filter_dims=None, batch_Qsample=True, init_Qmu_range=0.1, init_Qsigma=0.2, Qsample_mode='bilinear', **kwargs):
        """
        ReadoutLayerQsample: 3d readout layer for NDNs.

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            filter_dims: tuple or list of ints, (num_channels, height, width, depth, lags)
            batch_Qample: bool, whether to sample Qgrid locations separately per sample per batch
            init_mu_range: float, range for uniform initialization of means
            init_sigma: float, standard deviation for uniform initialization of sigmas
            Qsample_mode: str, 'bilinear' or 'nearest'

        """

        assert input_dims is not None, "ReadoutLayer: Must specify input_dims"

        # Make sure filter_dims is filled in, and to single the spatial filter dimensions
        if filter_dims is None:
            filter_dims = deepcopy(input_dims)
        
        filter_dims[1:4] = [1,1,1]
        
        assert filter_dims[3] == 1, 'Cant handle temporal filter dims here, yet.'
        
        super().__init__(input_dims=input_dims,
            filter_dims=filter_dims,
            **kwargs)
    
        self.filter_dims[3] = 1

        self.batch_Qsample = batch_Qsample
        self.Qsample_mode = Qsample_mode

        self.init_Qmu_range = init_Qmu_range
        self.init_Qsigma = init_Qsigma

        if self.init_Qmu_range > 1.0 or self.init_Qmu_range <= 0.0 or self.init_Qsigma <= 0.0:
            raise ValueError("either init_Qmu_range doesn't belong to [0.0, 1.0] or init_Qsigma is non-positive")

        self.Qgrid_shape = (1, self.num_filters,1,1)

        self.Qmu = Parameter(torch.Tensor(self.num_filters,1))
        self.Qsigma = Parameter(torch.Tensor(self.num_filters,1))

        self.initialize_spatial_mapping()
        self.Qsample = False

    # END ReadoutLayerQsample.__init__

    def initialize_Q_mapping(self):
        """
        Initializes the mean, and sigma of the Gaussian readout for angle dim.

        Args:
            None
        """
        self.Qmu.data.uniform_(-self.init_Qmu_range)  # random initializations uniformly spread....
        # if self.gauss_type != 'full':
        #     self.sigma.data.fill_(self.init_sigma)
        # else:
        #self.sigma.data.uniform_(0, self.init_sigma)
        self.Qsigma.data.fill_(self.init_Qsigma)
        
        #self.weight.data.fill_(1 / self.input_dims[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def sample_Qgrid(self, batch_size, Qsample=None):
        """
        Returns the grid locations from the core by sampling from a Gaussian distribution
        More specifically, it returns sampled positions for each batch over all elements, given mus and sigmas
        If 'sample' is false, it just gives mu back (so testing has no randomness)
        This code is inherited and then modified, so different style than most other

        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
        """
        
        Qgrid_shape = (batch_size,) + self.Qgrid_shape[1:3]
                
        Qsample = self.training if Qsample is None else Qsample
        if Qsample:
            norm = self.Qmu.new(*Qgrid_shape).normal_()
        else:
            norm = self.Qmu.new(*Qgrid_shape).zero_()  # for consistency and CUDA capability

        out_norm = norm.new_zeros(*(Qgrid_shape[:3]+(2,)))  # for consistency and CUDA capability
        ## SEEMS dim-4 HAS TO BE IN THE SECOND DIMENSION (RATHER THAN FIRST)
        out_norm[:,:,:,0] = (norm * self.Qsigma[None, None, None, :] + self.Qmu[None, None, None, :]).clamp(-1,1)
        out_norm[:,:,:,1] = (norm * self.Qsigma[None, None, None, :] + self.Qmu[None, None, None, :]).clamp(-1,1)
          
        return out_norm

    def fit_Qmus( self, val=None, Qsigma=None, verbose=True ):
        """
        Quick function that turns on or off fitting mus/sigmas and toggles sample
        
        Args:
            val (bool): True or False -- must specify
            Qsigma (float): choose starting sigma value (default no choice)
        """
        assert val is not None, "fit_mus(): must set val to be True or False"
        self.Qsample = val
        self.set_parameters(val=val, name='Qmu')
        self.set_parameters(val=val, name='Qsigma')
        if sigma is not None:
            self.Qsigma.data.fill_(Qsigma)
        if verbose:
            if val:
                print("  ReadoutLayer: fitting Qmus")
            else:
                print("  ReadoutLayer: not fitting Qmus")

    def forward(self, x, shift=None, Qshift=None):
        """
        Propagates the input forwards through the readout

        Args:
            x: input data
            shift (bool): shifts the location of the grid (from eye-tracking data)
            Qshift (bool): shifts the location of the grid

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

        feat = self.features # this is the filter weights for each unit
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

        # Might have to do sample per cell (i.e. for loop)
        if self.batch_Qsample:
            # sample the grid_locations separately per sample per batch
            Qgrid = self.sample_Qgrid(batch_size=N, Qsample=self.Qsample)  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all sample in the batch
            Qgrid = self.sample_Qgrid(batch_size=1, Qsample=self.Qsample).expand(N, outdims,1, 1)
        
        if Qshift is not None:
            # shifter is run outside the readout forward
            Qgrid = Qgrid + Qshift[:, None, None, :]
        
        y = F.grid_sample(x, grid, mode=self.sample_mode, align_corners=self.align_corners, padding_mode='border')
        # note I switched this from the default 'zeros' so that it wont try to go past the border
        
        # reduce grid sample for angle
        z = torch.zeros((N,self.input_dims[0],self.num_filters,1), dtype=torch.float32, device=self.weight.device)
        for i in range(self.num_filters):
            y_i = y[:,:,i,:].reshape(N,self.input_dims[0],self.input_dims[3],1)
            Qgrid_i = Qgrid[:,i,:,:].reshape(N,1,1,2)
            z_i = F.grid_sample(y_i, Qgrid_i, mode=self.Qsample_mode, align_corners=self.align_corners, padding_mode='border')
            z[:,:,i,:] = z_i.reshape(N,self.input_dims[0],1)
        
        z = (z.squeeze(-1) * feat).sum(1).view(N, outdims)
        #z = (z.squeeze(-1)).sum(1).view(N, outdims)

        if self.bias is not None:
            z = z + bias
        
        if self.NL is not None:
            z = self.NL(z)
        
        return z
    # END ReadoutLayer3d.forward

    def passive_readout(self):
        """
        This might have to be redone for readout3d to take into account extra filter dim.
        """
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
    def layer_dict(cls, Qsigma=1, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

        Args:
            None

        Returns:
            Ldict: dictionary of layer parameters
        """
        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'readoutQ'
        Ldict['batch_Qample'] = True
        Ldict['init_Qmu_range'] = 0.1
        Ldict['init_Qsigma'] = 0.2
        Ldict['Qsample_mode'] = 'bilinear'
        return Ldict
    # END [classmethod] ReadoutLayer3d.layer_dict