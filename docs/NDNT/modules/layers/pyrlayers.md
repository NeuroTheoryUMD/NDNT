Module NDNT.modules.layers.pyrlayers
====================================

Classes
-------

`MaskLayer(input_dims=None, num_filters=None, NLtype: str = 'lin', norm_type: int = 0, pos_constraint=0, num_inh: int = 0, bias: bool = False, weights_initializer: str = 'xavier_uniform', output_norm=None, initialize_center=False, bias_initializer: str = 'zeros', reg_vals: dict = None, **kwargs)`
:   MaskLayer: Layer with a mask applied to the weights.
    
    MaskLayer: Layer with a mask applied to the weights.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        ## NOT CANT BE PART OF DICTIONARY mask: np.ndarray, mask to apply to the weights
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

    `layer_dict(**kwargs)`
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

    `set_mask(self, mask=None)`
    :   Sets mask -- instead of plugging in by hand. Registers mask as being set, and checks dimensions.
        Can also use to rest to have trivial mask (all 1s)
        Mask will be numpy, leave mask blank if want to set to default mask (all ones)
        
        Args:
            mask: numpy array of size of filters (filter_dims x num_filters)

`MaskSTconvLayer(input_dims=None, num_filters=None, initialize_center=False, output_norm=None, **kwargs)`
:   MaskSTConvLayer: STconvLayer with a mask applied to the weights.
    
    MaskSTconvLayer: STconvLayer with a mask applied to the weights.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        mask: np.ndarray, mask to apply to the weights
        output_norm: str, 'batch', 'batchX', or None
        **kwargs: additional arguments to pass to STConvLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.STconvLayer
    * NDNT.modules.layers.convlayers.TconvLayer
    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(**kwargs)`
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

    `set_mask(self, mask=None)`
    :   Sets mask -- instead of plugging in by hand. Registers mask as being set, and checks dimensions.
        Can also use to rest to have trivial mask (all 1s)
        Mask will be numpy, leave mask blank if want to set to default mask (all ones)
        
        Args:
            mask: numpy array of size of filters (filter_dims x num_filters)