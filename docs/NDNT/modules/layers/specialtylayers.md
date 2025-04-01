Module NDNT.modules.layers.specialtylayers
==========================================

Classes
-------

`L1convLayer(**kwargs)`
:   First start with non-convolutional version.
    
    L1convLayer: Convolutional layer with L1 regularization.
    
    Set up ConvLayer with L1 regularization.
    
    Args:
        **kwargs: additional arguments to pass to ConvLayer
    
    Returns:
        None

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(**kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

    ### Methods

    `preprocess_weights(self)`
    :   Preprocess weights for L1convLayer.
        
        Returns:
            w: torch.Tensor, preprocessed weights

    `reset_parameters2(self, weights_initializer=None, bias_initializer=None, param=None) ‑> None`
    :   Reset parameters for L1convLayer.
        
        Args:
            weights_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            bias_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            param: dict, additional parameters to pass to the initializer
        
        Returns:
            None

`MaskLayer(input_dims=None, num_filters=None, mask=None, NLtype: str = 'lin', norm_type: int = 0, pos_constraint=0, num_inh: int = 0, bias: bool = False, weights_initializer: str = 'xavier_uniform', output_norm=None, initialize_center=False, bias_initializer: str = 'zeros', reg_vals: dict = None, **kwargs)`
:   MaskLayer: Layer with a mask applied to the weights.
    
    MaskLayer: Layer with a mask applied to the weights.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        mask: np.ndarray, mask to apply to the weights
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
        norm_type: int, normalization type
        pos_constraint: int, whether to enforce non-negative weights
        num_inh: int, number of inhibitory filters
        bias: bool, whether to include bias term
        weights_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        output_norm: str, 'batch', 'batchX', or None
        initialize_center: bool, whether to initialize the center
        bias_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        reg_vals: dict, regularization values
        **kwargs: additional arguments to pass to NDNLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(mask=None, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            mask: np.ndarray, mask to apply to the weights
            **kwargs: additional arguments to pass to NDNLayer
        
        Returns:
            Ldict: dict, dictionary of layer parameters

    ### Methods

    `preprocess_weights(self)`
    :   Preprocess weights for MaskLayer.
        
        Returns:
            w: torch.Tensor, preprocessed weights

`OnOffLayer(input_dims=None, num_filters=None, num_lags=None, temporal_tent_spacing=None, output_norm=None, res_layer=False, **kwargs)`
:   OnOffLayer: Layer with separate on and off filters.
    
    Args (required):
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        num_lags: number of lags in spatiotemporal filter
        
    Args (optional):
        temporal_tent_spacing: int, spacing of tent basis functions
        output_norm: str, 'batch', 'batchX', or None
        res_layer: bool, whether to make a residual layer
        **kwargs: additional arguments to pass to NDNLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.specialtylayers.Tlayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(num_lags=None, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            num_lags: int, number of lags in spatiotemporal filter
            **kwargs: additional arguments to pass to NDNLayer
        
        Returns:
            Ldict: dict, dictionary of layer parameters

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the OnOffLayer.
        
        Args:
            x: torch.Tensor, input tensor
        
        Returns:
            y: torch.Tensor, output tensor

    `plot_filters(self, time_reverse=None, **kwargs)`
    :   Plot the filters for the OnOffLayer.
        
        Args:
            time_reverse: bool, whether to reverse the time dimension
            **kwargs: additional arguments to pass to the plotting function
        
        Returns:
            None

`Tlayer(input_dims=None, num_filters=None, num_lags=None, temporal_tent_spacing=None, output_norm=None, res_layer=False, **kwargs)`
:   NDN Layer where num_lags is handled convolutionally (but all else is normal)
    
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
    
    Tlayer: NDN Layer where num_lags is handled convolutionally (but all else is normal).
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        num_lags: number of lags in spatiotemporal filter
        temporal_tent_spacing: int, spacing of tent basis functions
        output_norm: str, 'batch', 'batchX', or None
        res_layer: bool, whether to make a residual layer
        **kwargs: additional arguments to pass to NDNLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.modules.layers.specialtylayers.OnOffLayer

    ### Static methods

    `layer_dict(num_lags=None, res_layer=False, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the Tlayer.
        
        Args:
            x: torch.Tensor, input tensor
        
        Returns:
            y: torch.Tensor, output tensor

    `plot_filters(self, cmaps='gray', num_cols=8, row_height=2, time_reverse=False)`
    :   Overload plot_filters to automatically time_reverse.
        
        Args:
            cmaps: str or list of str, colormap(s) to use
            num_cols: int, number of columns to use in plot
            row_height: int, height of each row in plot
            time_reverse: bool, whether to reverse the time dimension
        
        Returns:
            None