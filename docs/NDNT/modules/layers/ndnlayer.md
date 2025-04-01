Module NDNT.modules.layers.ndnlayer
===================================

Classes
-------

`NDNLayer(input_dims=None, num_filters=None, filter_dims=None, NLtype: str = 'lin', norm_type: int = 0, pos_constraint=0, num_inh: int = 0, bias: bool = False, weights_initializer: str = 'xavier_uniform', initialize_center=False, bias_initializer: str = 'zeros', reg_vals: dict = None, **kwargs)`
:   Base class for NDN layers.
    Handles the weight initialization and regularization
    
    Args:
        input_dims: list of 4 ints, dimensions of input
            [Channels, Height, Width, Lags]
            use 1 if the dimension is not used
        num_filters: int, number of filters in layer
        NLtype: str, type of nonlinearity to use (see activations.py)
        norm: int, type of filter normalization to use (0=None, 1=filters are unit vectors)
        pos_constraint: bool, whether to constrain weights to be positive
        num_inh: int, number of inhibitory units (creates ei_mask and makes "inhibitory" units have negative output)
        bias: bool, whether to include bias term
        weight_init: str, type of weight initialization to use (see reset_parameters, default 'xavier_uniform')
        bias_init: str, type of bias initialization to use (see reset_parameters, default 'zeros')
        reg_vals: dict, regularization values to use (see regularizers.py)
    
    NDNLayers have parameters 'weight' and 'bias' (if bias=True)
    
    The forward has steps:
        1. preprocess_weights
        2. main computation (this is just a linear layer for the base class)
        3. nonlinearity
        4. ei_mask
    
    weight is always flattened to a vector, and then reshaped to the appropriate size
    use get_weights() to get the weights in the correct shape
    
    preprocess_weights() applies positive constraint and normalization if requested
    compute_reg_loss() computes the regularization loss for the regularization specified in reg_vals
    
    Initialize the layer.
    
    Args:
        input_dims: list of 4 ints, dimensions of input
        num_filters: int, number of filters in layer
        NLtype: str, type of nonlinearity to use (see activations.py)
        norm: int, type of filter normalization to use (0=None, 1=filters are unit vectors, 2=maxnorm?)
        pos_constraint: bool, whether to constrain weights to be positive
        num_inh: int, number of inhibitory units (creates ei_mask and makes "inhibitory" units have negative output)
        bias: bool, whether to include bias term
        weight_init: str, type of weight initialization to use (see reset_parameters, default 'xavier_uniform')
        bias_init: str, type of bias initialization to use (see reset_parameters, default 'zeros')
        reg_vals: dict, regularization values to use (see regularizers.py)

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.modules.layers.bilayers.BiSTconv1D
    * NDNT.modules.layers.bilayers.BinocShiftLayer
    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.dimlayers.ChannelLayer
    * NDNT.modules.layers.dimlayers.Dim0Layer
    * NDNT.modules.layers.dimlayers.DimSPLayer
    * NDNT.modules.layers.dimlayers.DimSPTLayer
    * NDNT.modules.layers.laglayers.LagLayer
    * NDNT.modules.layers.lvlayers.LVLayer
    * NDNT.modules.layers.lvlayers.LVLayerOLD
    * NDNT.modules.layers.masklayers.MaskLayer
    * NDNT.modules.layers.normlayers.DivNormLayer
    * NDNT.modules.layers.orilayers.OriLayer
    * NDNT.modules.layers.partiallayers.NDNLayerPartial
    * NDNT.modules.layers.pyrlayers.MaskLayer
    * NDNT.modules.layers.readouts.FixationLayer
    * NDNT.modules.layers.readouts.ReadoutLayer
    * NDNT.modules.layers.specialtylayers.L1convLayer
    * NDNT.modules.layers.specialtylayers.MaskLayer
    * NDNT.modules.layers.specialtylayers.Tlayer
    * NDNT.modules.layers.timelayers.TimeLayer
    * NDNT.modules.layers.timelayers.TimeShiftLayer

    ### Static methods

    `dim_info(input_dims=None, num_filters=None, **kwargs)`
    :   This uses the methods in the init to determine the input_dims, output_dims, filter_dims, and actual size of
        the weight tensor (weight_shape), given inputs, and package in a dictionary. This should be overloaded with each
        child of NDNLayer if want to use -- but this is external to function of actual layer.
        
        Args:
            input_dims: list of 4 ints, dimensions of input
            num_filters: int, number of filters in layer
        
        Returns:
            dinfo: dict, dictionary of dimension information

    `layer_dict(input_dims=None, num_filters=None, bias=False, NLtype='lin', norm_type=0, initialize_center=False, num_inh=0, pos_constraint=False, reg_vals={}, output_norm=None, weights_initializer='xavier_uniform', bias_initializer='zeros', **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            input_dims: list of 4 ints, dimensions of input
            num_filters: int, number of filters in layer
            bias: bool, whether to include bias term
            NLtype: str, type of nonlinearity to use (see activations.py)
            norm_type: int, type of filter normalization to use (0=None, 1=filters are unit vectors, 2=maxnorm?)
            pos_constraint: bool, whether to constrain weights to be positive
            num_inh: int, number of inhibitory units (creates ei_mask and makes "inhibitory" units have negative output)
            initialize_center: bool, whether to initialize the weights to have a Gaussian envelope
            reg_vals: dict, regularization values to use (see regularizers.py)
            output_norm: str, type of output normalization to use
            weights_initializer: str, type of weight initialization to use
            bias_initializer: str, type of bias initialization to use
        
        Returns:
            dict: dictionary of layer parameters

    ### Instance variables

    `num_inh`
    :

    `output_dims`
    :

    ### Methods

    `compute_reg_loss(self)`
    :   Compute the regularization loss for the layer by calling reg_module function on
        the preprocessed weights.
        
        Args:
            None
        
        Returns:
            reg_loss: torch.Tensor, regularization loss

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass for the layer.
        
        Args:
            x: torch.Tensor, input tensor
        
        Returns:
            x: torch.Tensor, output tensor

    `get_weights(self, to_reshape=True, time_reverse=False, num_inh=0)`
    :   num-inh can take into account previous layer inhibition weights.
        
        Args:
            to_reshape: bool, whether to reshape the weights to the original filter shape
            time_reverse: bool, whether to reverse the time dimension
            num_inh: int, number of inhibitory units
        
        Returns:
            ws: np.ndarray, weights of the layer, on the CPU

    `info(self, expand=False, to_output=True)`
    :   This outputs the layer information in abbrev (default) or expanded format

    `initialize_gaussian_envelope(self)`
    :   Initialize the weights to have a Gaussian envelope.
        This is useful for initializing filters to have a spatial-temporal Gaussian shape.

    `list_parameters(self)`
    :

    `plot_filters(self, time_reverse=None, **kwargs)`
    :   Plot the filters in the layer. It first determines whether layer is spatiotemporal (STRF plot)
        or "internal": the different being that spatiotemporal will have lags / dim-3 of filter
        
        Args:
            cmaps: str or colormap, colormap to use for plotting (default 'gray')
            num_cols: int, number of columns to use in plot (default 8)
            row_height: int, number of rows to use in plot (default 2)
            time_reverse: bool, whether to reverse the time dimension (default depends on dimension)
        
        Returns:
            None

    `preprocess_weights(self)`
    :   Preprocess the weights before using them in the forward pass.
        
        Returns:
            w: torch.Tensor, preprocessed weights

    `reset_parameters(self, weights_initializer=None, bias_initializer=None, param=None) ‑> None`
    :   Initialize the weights and bias parameters of the layer.
        
        Args:
            weights_initializer: str, type of weight initialization to use
                options: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal', 'zeros', 'ones'
            bias_initializer: str, type of bias initialization to use
                options: 'uniform', 'normal', 'zeros', 'ones'
            param: float or list of floats, parameter for weight initialization

    `set_parameters(self, name=None, val=None)`
    :   Turn fitting for named params on or off.
        If name is none, do for whole layer.

    `set_reg_val(self, reg_type, reg_val=None)`
    :   Set the regularization value for a given regularization type.
        
        Args:
            reg_type: str, type of regularization to set
            reg_val: float, value to set the regularization to
        
        Returns:
            None