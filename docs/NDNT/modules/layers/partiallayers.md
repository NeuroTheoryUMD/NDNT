Module NDNT.modules.layers.partiallayers
========================================

Classes
-------

`ConvLayerPartial(input_dims=None, num_filters=None, filter_width=None, num_fixed_filters=0, fixed_dims=None, bias: bool = False, fixed_num_inh=0, num_inh: int = 0, output_norm=None, **kwargs)`
:   ConvLayerPartial: NDNLayer with only some of the weights in the layer fit, determined by
        fixed_dims OR num_fixed_filters
    
    ConvLayerPartial: NDNLayer with only some of the weights in the layer fit, determined by
        fixed_dims OR num_fixed_filters
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_dims: width of convolutional kernel (int or list of ints)
        num_fixed_filters: number of filters to hold constant (default=0)
        fixed_dims: 4-d size of fixed dims, use -1 for not-touching dims. Should have 3 -1s
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
        norm_type: int, normalization type
        pos_constraint: int, whether to enforce non-negative weights
        num_inh: int, number of inhibitory filters
        bias: bool, whether to include bias term
        weights_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        output_norm: str, 'batch', 'batchX', or None
        padding: 'same','valid', or 'circular' (default 'same')
        initialize_center: bool, whether to initialize the center
        bias_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        reg_vals: dict, regularization values
        **kwargs: additional arguments to pass to NDNLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(fixed_dims=None, num_fixed_filters=0, fixed_num_inh=0, filter_width=None, **kwargs)`
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
    :   Preprocess weights: assemble from real weights + fixed and then NDNLayer Process weights clipped in.
        
        Returns:
            w: torch.Tensor, preprocessed weights

`NDNLayerPartial(input_dims=None, num_filters=None, num_fixed_filters=0, fixed_dims=None, fixed_num_inh=0, NLtype: str = 'lin', norm_type: int = 0, pos_constraint=0, num_inh: int = 0, bias: bool = False, weights_initializer: str = 'xavier_uniform', output_norm=None, initialize_center=False, bias_initializer: str = 'zeros', reg_vals: dict = None, **kwargs)`
:   NDNLayerPartial: NDNLayer where only some of the weights are fit
    
    NDNLayerPartial: NDNLayer with only some of the weights in the layer fit, determined by
        fixed_dims OR num_fixed_filters
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        num_fixed_filters: number of filters to hold constant (default=0)
        fixed_dims: 4-d size of fixed dims, use -1 for not-touching dims. Should have 3 -1s
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

    `layer_dict(fixed_dims=None, num_fixed_filters=0, **kwargs)`
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
    :   Preprocess weights: assemble from real weights + fixed and then NDNLayer Process weights clipped in.
        
        Returns:
            w: torch.Tensor, preprocessed weights

`OriConvLayerPartial(input_dims=None, num_filters=None, filter_width=None, num_fixed_filters=0, fixed_dims=None, bias: bool = False, fixed_num_inh=0, num_inh: int = 0, angles=None, output_norm=None, **kwargs)`
:   ConvLayerPartial: NDNLayer with only some of the weights in the layer fit, determined by
        fixed_dims OR num_fixed_filters
    
    OriConvLayerPartial: OriConvLayer with only some of the weights in the layer fit, determined by
        fixed_dims OR num_fixed_filters
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_dims: width of convolutional kernel (int or list of ints)
        num_fixed_filters: number of filters to hold constant (default=0)
        fixed_dims: 4-d size of fixed dims, use -1 for not-touching dims. Should have 3 -1s
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
        norm_type: int, normalization type
        pos_constraint: int, whether to enforce non-negative weights
        num_inh: int, number of inhibitory filters
        bias: bool, whether to include bias term
        weights_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        output_norm: str, 'batch', 'batchX', or None
        padding: 'same','valid', or 'circular' (default 'same')
        initialize_center: bool, whether to initialize the center
        bias_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        reg_vals: dict, regularization values
        **kwargs: additional arguments to pass to NDNLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.orilayers.OriConvLayer
    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(fixed_dims=None, num_fixed_filters=0, fixed_num_inh=0, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            FIX
            **kwargs: additional arguments to pass to NDNLayer
        
        Returns:
            Ldict: dict, dictionary of layer parameters

    ### Methods

    `preprocess_weights(self)`
    :   Preprocess weights: assemble from real weights + fixed and then NDNLayer Process weights clipped in.
        
        Returns:
            w: torch.Tensor, preprocessed weights