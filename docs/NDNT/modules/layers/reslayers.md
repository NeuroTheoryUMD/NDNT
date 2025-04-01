Module NDNT.modules.layers.reslayers
====================================

Classes
-------

`IterLayer(input_dims=None, num_filters=None, filter_width=None, num_iter=1, output_config='last', temporal_tent_spacing=None, output_norm=None, window=None, res_layer=True, LN_reverse=False, **kwargs)`
:   Residual network composed of many layers (num_iter) with weight-sharing across layers.
    It is based on conv-net setup, and can output from all layers or just last (output_config)
    Also, num_inh is setup for only output layers by default
    
    Args (required):
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_width: width of convolutional kernel (int or list of ints)
    
    Args (optional):
        padding: 'same' or 'valid' (default 'same')
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
    
    Initialize IterLayer with specified parameters.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_width: width of convolutional kernel (int or list of ints)
        num_iter: number of iterations to apply the layer
        output_config: 'last' or 'full' (default 'last')
        temporal_tent_spacing: spacing of temporal tent-basis functions (default None)
        output_norm: 'batch', 'batchX', or None (default None)
        window: 'hamming' or None (default None)
        res_layer: bool, whether to include a residual connection (default True)
        LN_reverse: bool, whether to apply layer normalization after nonlinearity (default False)
        **kwargs: additional arguments to pass to ConvLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.modules.layers.reslayers.ResnetBlock

    ### Static methods

    `layer_dict(filter_width=None, num_iter=1, output_config='last', res_layer=True, LN_reverse=False, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            filter_width: width of convolutional kernel (int or list of ints)
            num_iter: number of iterations to apply the layer
            output_config: 'last' or 'full' (default 'last')
            res_layer: bool, whether to include a residual connection (default True)
            LN_reverse: bool, whether to apply layer normalization after nonlinearity (default False)
            **kwargs: additional arguments to pass to ConvLayer
        
        Returns:
            Ldict: dictionary of layer parameters

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the IterLayer.
        
        Args:
            x: torch.Tensor, input tensor
        
        Returns:
            y: torch.Tensor, output tensor

`IterSTlayer(input_dims=None, num_filters=None, filter_width=None, num_iter=1, num_lags=1, res_layer=True, output_config='last', output_norm=None, **kwargs)`
:   Residual network layer based on conv-net setup but with different forward.
    Namely, the forward includes a skip-connection, and has to be 'same'
    
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
    
    Initialize IterLayer with specified parameters.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_width: width of convolutional kernel (int or list of ints)
        num_iter: number of iterations to apply the layer
        num_lags: number of lags in spatiotemporal filter
        res_layer: bool, whether to include a residual connection (default True)
        output_config: 'last' or 'full' (default 'last')
        output_norm: 'batch', 'batchX', or None (default None)
        **kwargs: additional arguments to pass to ConvLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.STconvLayer
    * NDNT.modules.layers.convlayers.TconvLayer
    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(filter_width=None, num_iter=1, num_lags=1, res_layer=True, output_config='last', **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            filter_width: width of convolutional kernel (int or list of ints)
            num_iter: number of iterations to apply the layer
            num_lags: number of lags in spatiotemporal filter
            res_layer: bool, whether to include a residual connection (default True)
            output_config: 'last' or 'full' (default 'last')
            **kwargs: additional arguments to pass to ConvLayer
        
        Returns:
            Ldict: dictionary of layer parameters

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the IterLayer.
        
        Args:
            x: torch.Tensor, input tensor
        
        Returns:
            y: torch.Tensor, output tensor

`IterTlayer(input_dims=None, num_filters=None, filter_width=None, num_iter=1, num_lags=1, res_layer=True, output_config='last', output_norm=None, **kwargs)`
:   Residual network layer based on conv-net setup but with different forward.
    Namely, the forward includes a skip-connection, and has to be 'same'
    
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
    
    Initialize IterTLayer with specified parameters.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_width: width of convolutional kernel (int or list of ints)
        num_iter: number of iterations to apply the layer
        num_lags: number of lags in spatiotemporal filter
        res_layer: bool, whether to include a residual connection (default True)
        output_config: 'last' or 'full' (default 'last')
        output_norm: 'batch', 'batchX', or None (default None)
        **kwargs: additional arguments to pass to ConvLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.TconvLayer
    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(filter_width=None, num_iter=1, num_lags=1, res_layer=True, output_config='last', **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            filter_width: width of convolutional kernel (int or list of ints)
            num_iter: number of iterations to apply the layer
            num_lags: number of lags in spatiotemporal filter
            res_layer: bool, whether to include a residual connection (default True)
            output_config: 'last' or 'full' (default 'last')
            **kwargs: additional arguments to pass to ConvLayer
        
        Returns:
            Ldict: dictionary of layer parameters

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the IterLayer.
        
        Args:
            x: torch.Tensor, input tensor
        
        Returns:
            y: torch.Tensor, output tensor

`ResnetBlock(input_dims=None, num_filters=None, filter_width=None, num_iter=1, res_layer=True, **kwargs)`
:   Residual network layer based on conv-net setup but with different forward.
    Namely, the forward includes a skip-connection, and has to be 'same'
    
    Args (required):
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_width: width of convolutional kernel (int or list of ints)
    
    Args (optional):
        padding: 'same' or 'valid' (default 'same')
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
    
    Initialize IterLayer with specified parameters.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_width: width of convolutional kernel (int or list of ints)
        num_iter: number of iterations to apply the layer
        output_config: 'last' or 'full' (default 'last')
        temporal_tent_spacing: spacing of temporal tent-basis functions (default None)
        output_norm: 'batch', 'batchX', or None (default None)
        window: 'hamming' or None (default None)
        res_layer: bool, whether to include a residual connection (default True)
        LN_reverse: bool, whether to apply layer normalization after nonlinearity (default False)
        **kwargs: additional arguments to pass to ConvLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.reslayers.IterLayer
    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the ResnetBlock: input filtering (one layer wit filter width) followed
        by some number of filter_width-1 conv layers
        
        Args:
            x: torch.Tensor, input tensor
        
        Returns:
            y: torch.Tensor, output tensor