Module NDNT.modules.layers.normlayers
=====================================

Classes
-------

`DivNormLayer(input_dims=None, num_filters=None, filter_dims=None, pos_constraint=True, bias=True, **kwargs)`
:   Divisive normalization implementation: not explicitly convolutional.
    
    Divisive normalization layer.
    
    Args:
        input_dims: list or tuple of ints, dimensions of input tensor.
        num_filters: int, number of filters to use.
        filter_dims: list or tuple of ints, dimensions of filter tensor.
        pos_constraint: bool, if True, apply a positivity constraint to the weights.
        bias: bool, if True, include a bias term.

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass for divisive normalization layer.
        
        Args:
            x: torch.Tensor, input tensor.
        
        Returns:
            x: torch.Tensor, output tensor.

    `preprocess_weights(self)`
    :   Preprocesses weights by applying positivity constraint and normalization.
        
        Returns:
            w: torch.Tensor, preprocessed weights.