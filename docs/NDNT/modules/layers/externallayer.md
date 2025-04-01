Module NDNT.modules.layers.externallayer
========================================

Classes
-------

`ExternalLayer(input_dims=None, num_filters=None, output_dims=None, **kwargs)`
:   This is a dummy 'layer' for the Extenal network that gets filled in by the passed-in network.
    
    ExternalLayer: Dummy layer for the Extenal network that gets filled in by the passed-in network.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        output_dims: tuple or list of ints, (num_channels, height, width, lags)
        **kwargs: additional arguments to pass to NDNLayer

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass for ExternalLayer.
        
        Args:
            x: input tensor
        
        Returns:
            y: output tensor