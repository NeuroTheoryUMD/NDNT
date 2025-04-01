Module NDNT.modules.layers.dimlayers
====================================

Classes
-------

`ChannelLayer(input_dims=None, num_filters=None, **kwargs)`
:   Applies individual filter for each filter (dim0) dimension, preserving separate channels but filtering over other dimensions
    => num_filters equals the channel dimension by definition, otherwise like NDNLayer
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters

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

    `forward(self, x) ‑> Callable[..., Any]`
    :   Args:
            x: torch.Tensor, input tensor of shape [B, C, H, W, L]
        
        Returns:
            y: torch.Tensor, output tensor of shape [B, N, H, W, L]

`Dim0Layer(input_dims=None, num_filters=None, **kwargs)`
:   Filters that act solely on filter-dimension (dim-0).
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `dim_info(input_dims=None, num_filters=None, **kwargs)`
    :   This uses the methods in the init to determine the input_dims, output_dims, filter_dims, and actual size of
        the weight tensor (weight_shape), given inputs, and package in a dictionary. This should be overloaded with each
        child of NDNLayer if want to use -- but this is external to function of actual layer.
        
        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
        
        Returns:
            dinfo: dictionary with keys 'input_dims', 'filter_dims', 'output_dims', 'num_outputs', 'weight_shape'

    `layer_dict(**kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Args:
            x: torch.Tensor, input tensor of shape [B, C, H, W, L]
        
        Returns:
            y: torch.Tensor, output tensor of shape [B, N, H, W, L]

`DimSPLayer(input_dims=None, num_filters=None, transparent=False, **kwargs)`
:   Filters that act solely on spatial-dimensions (dims-1,3)
    transparent=True means that one spatial filter for each channel
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        transparent: if True, one spatial filter for each channel

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(transparent=False, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Args:
            x: torch.Tensor, input tensor of shape [B, C, H, W, L]
        
        Returns:
            y: torch.Tensor, output tensor of shape [B, N, H, W, L]

`DimSPTLayer(input_dims=None, num_filters=None, transparent=False, **kwargs)`
:   Filters that act solely on spatial-dimensions (dims-1,3).
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        transparent: if True, one spatial filter for each channel

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(transparent=False, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Args:
            x: torch.Tensor, input tensor of shape [B, C, H, W, L]
        
        Returns:
            y: torch.Tensor, output tensor of shape [B, N, H, W, L]