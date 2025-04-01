Module NDNT.modules.layers.bilayers
===================================

Classes
-------

`BiConvLayer1D(**kwargs)`
:   Filters that act solely on filter-dimension (dim-0)
    
    Same arguments as ConvLayer, but will reshape output to divide space in half.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_dims: width of convolutional kernel (int or list of ints)
        padding: 'same' or 'valid' (default 'same')
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

`BiSTconv1D(input_dims=None, num_filters=None, filter_dims=None, norm_type=None, NLtype='relu', **kwargs)`
:   To be BinocMixLayer (bimix):
    Inputs of size B x C x 2xNX x 1: mixes 2 eyes based on ratio
        filter number is NC -- one for each channel: so infered from input_dims
        filter_dims is [1,1,1,1] -> one number per filter
    ORIGINAL BiSTconv1D: Filters that act solely on filter-dimension (dim-0)
    
    Same arguments as ConvLayer, but will reshape output to divide space in half.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_dims: width of convolutional kernel (int or list of ints)
        padding: 'same' or 'valid' (default 'same')
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Call conv forward, but then option to reshape.
        
        Args:
            x: tensor of shape [batch, num_channels, height, width, lags]
        
        Returns:
            y: tensor of shape [batch, num_outputs]

`BinocLayer1D(input_dims=None, num_filters=None, padding=None, **kwargs)`
:   Takes a monocular convolutional output over both eyes -- assumes first spatial dimension is doubled
    -- reinterprets input as separate filters from each eye, but keeps them grouped
    -- assumes each 2 input filters (for each eye -- 4 inputs total) are inputs to each output filter
    
    Same arguments as ConvLayer, but will reshape output to divide space in half
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        padding: 'same' or 'valid' (default 'same')

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Call conv forward, but then option to reshape
        
        Args:
            x: tensor of shape [batch, num_channels, height, width, lags]
        
        Returns:
            y: tensor of shape [batch, num_outputs]

`BinocShiftLayer(input_dims=None, init_sigma=3, padding=0, weights_initializer=None, norm_type=0, **kwargs)`
:   This processes monocular output that spans 2*NX and fits weight for each filter (decoding binocularity)
    and shift using mu and sigma
    weights are across shifts for each input filter (number output filters = num inputs)
    also chooses shift for each filter
    
    Same arguments as ConvLayer, but will make binocular filter with range of shifts. This assumes
    input from ConvLayer (not BiConv) with num_filters x 72 input dims (can adjust) 
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        filter_width: width of convolutional filter
        num_filters: number of output filters

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x, shift=None) ‑> Callable[..., Any]`
    :   Propagates the input forwards through the readout
        
        Args:
            x: input data
            shift (bool): shifts the location of the grid (from eye-tracking data)
        
        Returns:
            y: neuronal activity

    `norm_sample(self, batch_size, to_sample=None)`
    :   Returns a gaussian (or zeroed) sample, given the batch_size and to_sample parameter
        
        Args:
            batch_size (int): size of the batch
            to_sample (bool/None): determines whether we draw a sample from Gaussian distribution, 
                N(shifts,sigmas), defined per filter or use the mean of the Gaussian distribution without 
                sampling. If to_sample is None (default), samples from the Gaussian during training phase and
                fixes to the mean during evaluation phase. Note that if to_sample is True/False, it overrides 
                the model_state (i.e training or eval) and does as instructed
        Returns:
            out_norm: batch_size x num_filters x shifts applied to all positions

    `preprocess_weights(self, gelu_mult=2.0)`
    :   self.weight is a list of the binocular weighting of each filter

`BinocShiftLayerOld(input_dims=None, filter_width=None, num_shifts=11, num_filters=None, padding=None, **kwargs)`
:   Alternative: this processes monocular output that spans 2*NX and fits weight for each filter (decoding binocularity)
    and shift using mu and sigma
    
    Makes binocular filters from filter bank of monocular filters and applies to binocular stimulus. The monocular
    filters are convolutional (filter_width x num_lags) where filter_width is specified and num_lags matches input
    dims. Input dims should have 2 channels (and only one dim space): 2 x NX x 1 x num_lags
    
    Same arguments as ConvLayer, but will make binocular filter with range of shifts
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        filter_width: width of convolutional filter
        num_filters: number of output filters
        padding: 'same' or 'valid' (default 'same')

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

`ChannelConvLayer(input_dims=None, num_filters=None, filter_width=None, temporal_tent_spacing=None, output_norm=None, window=None, **kwargs)`
:   Channel-Convolutional NDN Layer -- convolutional layer that has each output filter use different
    group of M input filters. So, if there are N output filters, the channel dimension of the input has to be M*N
    
    Args (required):
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_dims: width of convolutional kernel (int or list of ints)
    
    Args (optional):
        padding: 'same' or 'valid' (default 'same')
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
    
    Same arguments as ConvLayer, but will reshape output to divide space in half.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_width: width of convolutional kernel (int or list of ints)
        temporal_tent_spacing: spacing of tent-basis functions in time
        output_norm: normalization to apply to output
        window: window function to apply to output

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Call conv forward, but then option to reshape.
        
        Args:
            x: tensor of shape [batch, num_channels, height, width, lags]
        
        Returns:
            y: tensor of shape [batch, num_outputs]