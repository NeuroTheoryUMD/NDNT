Module NDNT.modules.layers.readouts
===================================

Classes
-------

`FixationLayer(num_fixations=None, num_spatial_dims=2, batch_sample=False, fix_n_index=False, init_sigma=0.3, single_sigma=False, bias=False, NLtype='lin', **kwargs)`
:   FixationLayer for fixation.
    
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

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(num_fixations=None, num_spatial_dims=2, init_sigma=0.25, input_dims=None, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
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

    ### Methods

    `forward(self, x, shift=None) ‑> Callable[..., Any]`
    :   The input is the sampled fixation-stim
        
        Args:
            x: input data
            shift (bool): shifts the location of the grid (from eye-tracking data)
        
        Returns:
            y: neuronal activity

    `initialize(self, *args, **kwargs)`
    :   This will initialize the layer parameters.

`ReadoutLayer(input_dims=None, num_filters=None, filter_dims=None, batch_sample=True, init_mu_range=0.1, init_sigma=0.2, gauss_type: str = 'isotropic', align_corners=False, mode='bilinear', **kwargs)`
:   ReadoutLayer for spatial readout.
    
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

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.modules.layers.readouts.ReadoutLayer3d

    ### Static methods

    `layer_dict(NLtype='softplus', mode='bilinear', **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            NLtype: str, type of nonlinearity
        
        Returns:
            Ldict: dictionary of layer parameters

    ### Instance variables

    `features`
    :

    `grid`
    :

    ### Methods

    `compute_reg_loss(self)`
    :   Compute the regularization loss for the layer: superceding super, by calling reg_module to do
        this and then adding scaffold weight regularization if needed.
        
        Args:
            None
        
        Returns:
            reg_loss: torch.Tensor, regularization loss

    `enforce_grid(self)`
    :   Function that adjusts mus to correspond to precise points on pixel grid
        
        Args:
            None

    `fit_mus(self, val=None, sigma=None, verbose=True)`
    :   Quick function that turns on or off fitting mus/sigmas and toggles sample
        
        Args:
            val (bool): True or False -- must specify
            sigma (float): choose starting sigma value (default no choice)

    `forward(self, x, shift=None) ‑> Callable[..., Any]`
    :   Propagates the input forwards through the readout.
        
        Args:
            x: input data
            shift (bool): shifts the location of the grid (from eye-tracking data)
        
        Returns:
            y: neuronal activity

    `get_readout_locations(self)`
    :   Currently returns center location and sigmas, as list.
        
        Returns:
            mu: center locations of the readout units
            sigma: sigmas of the readout units

    `get_weights(self, to_reshape=True, time_reverse=None, num_inh=None)`
    :   Overloaded to read not use preprocess weights but instead use layer property features.
        
        Args:
            to_reshape (bool): whether to reshape the weights to the filter_dims
            time_reverse (bool): whether to reverse the time dimension
            num_inh (int): number of inhibitory units
        
        Returns:
            ws: weights of the readout layer

    `initialize_spatial_mapping(self)`
    :   Initializes the mean, and sigma of the Gaussian readout.
        
        Args:
            None

    `passive_readout(self)`
    :   This will not fit mu and std, but set them to zero. It will pass identities in,
        so number of input filters must equal number of readout units.
        
        Args:
            None
        
        Returns:
            None

    `sample_grid(self, batch_size, sample=None)`
    :   Returns the grid locations from the core by sampling from a Gaussian distribution
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

    `set_readout_locations(self, locs)`
    :   This hasn't been tested yet, but should be self-explanatory.
        
        Args:
            locs: locations of the readout units

`ReadoutLayer3d(input_dims=None, **kwargs)`
:   ReadoutLayer3d for 3d readout.
    This is a subclass of ReadoutLayer, but with the added dimension of time.
    
    ReadoutLayer3d: 3d readout layer for NDNs.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, depth, lags)

    ### Ancestors (in MRO)

    * NDNT.modules.layers.readouts.ReadoutLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.modules.layers.readouts.ReadoutLayerQsample

    ### Static methods

    `layer_dict(**kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            None
        
        Returns:
            Ldict: dictionary of layer parameters

    ### Methods

    `forward(self, x, shift=None) ‑> Callable[..., Any]`
    :   Propagates the input forwards through the readout
        
        Args:
            x: input data
            shift (bool): shifts the location of the grid (from eye-tracking data)
        
        Returns:
            y: neuronal activity

    `passive_readout(self)`
    :   This might have to be redone for readout3d to take into account extra filter dim.

    `set_mask(self, mask=None)`
    :   Sets mask -- instead of plugging in by hand. Registers mask as being set, and checks dimensions.
        Can also use to rest to have trivial mask (all 1s)
        Mask will be numpy, leave mask blank if want to set to default mask (all ones)
        
        Args:
            mask: numpy array of size of filters (filter_dims x num_filters)

`ReadoutLayerQsample(input_dims=None, filter_dims=None, batch_Qsample=True, init_Qsigma=0.5, Qsample_mode='bilinear', **kwargs)`
:   ReadoutLayerQsample for 3d readout with sampling over angle dimension Q.
    This is a subclass of ReadoutLayer3d.
    
    ReadoutLayerQsample: 3d readout layer for NDNs.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        filter_dims: tuple or list of ints, (num_channels, height, width, depth, lags)
        batch_Qample: bool, whether to sample Qgrid locations separately per sample per batch
        init_sigma: float, standard deviation for uniform initialization of sigmas
        Qsample_mode: str, 'bilinear' or 'nearest'

    ### Ancestors (in MRO)

    * NDNT.modules.layers.readouts.ReadoutLayer3d
    * NDNT.modules.layers.readouts.ReadoutLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Methods

    `degrees2mu(self, theta_deg, to_output=False, continuous=True, max_angle=180)`
    :   Converts degrees into mu-values. If to_output=True, outputs to an array, and otherwise
        stores in the Qmu variable. It detects whether half-circle of full circle using stored angle values
        
        Args:
            theta_deg (np array): array of angles in degrees into mu values, based on 180 or 360 deg wrap-around
            to_output (Boolean): whether to output to variable or store internally as Qmu (default: False, is the latter)
            continuous (Boolean): whether to convert to continuous angle or closest "integer" mu value (def True, continuous)
            max_angle: maximum angle represented in OriConv layers (default 180, but could be 360)
        Returns:
            Qmus: as numpy-array, if to_output is set to True, otherwise, nothing

    `fit_Qmus(self, val=None, Qsigma=None, verbose=True)`
    :   Quick function that turns on or off fitting Qmus/Qsigmas and toggles sample
        
        Args:
            val (bool): True or False -- must specify
            Qsigma (float): choose starting Qsigma value (default no choice)

    `forward(self, x, shift=None) ‑> Callable[..., Any]`
    :   Propagates the input forwards through the readout
        
        Args:
            x: input data
            shift (bool): shifts the location of the grid (from eye-tracking data)
        
        Returns:
            z: neuronal activity

    `mu2degrees(self, mu=None, max_angle=180)`
    :   Converts Qmu-values into degrees. Automatically reads from Qmu variable, and outputs a numpy array
        It detects whether half-circle of full circle using stored angle values
        
        Args:     
            Qmus: if dont want to read from layer, pass in values directly (default None, using self.Qmu)       
            max_angle: maximum angle represented in OriConv layers (default 180, but could be 360)
        
        Returns:
            thetas: angles in degrees (between 0 to max_angle)

    `sample_Qgrid(self, batch_size, Qsample=None)`
    :   Returns the chosen angle (Q) from the core by sampling from a Gaussian distribution
        More specifically, it returns sampled angle for each batch over all elements, given Qmus and Qsigmas
        Also implements wrap around for Qmus so edge mus get mapped back to first angle
        If 'Qsample' is false, it just gives mu back (so testing has no randomness)
        This code is inherited and then modified, so different style than most other
        
        Args:
            batch_size (int): size of the batch
            Qsample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(Qmu,Qsigma), defined per neuron
                            or use the mean, Qmu, of the Gaussian distribution without sampling.
                           if Qsample is None (default), samples from the N(Qmu,Qsigma) during training phase and
                             fixes to the mean, Qmu, during evaluation phase.
                           if Qsample is True/False, overrides the model_state (i.e training or eval) and does as instructed