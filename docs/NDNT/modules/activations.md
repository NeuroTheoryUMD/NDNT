Module NDNT.modules.activations
===============================

Functions
---------

`adaptive_elu(x, xshift, yshift, inplace=False)`
:   Exponential Linear Unit shifted by user specified values.
    
    Args:
        x (torch.Tensor): input tensor
        xshift (float): shift in x direction
        yshift (float): shift in y direction
        inplace (bool): whether to modify the input tensor in place
    
    Returns:
        torch.Tensor: output tensor

Classes
-------

`AdaptiveELU(xshift=0.0, yshift=1.0, inplace=True, **kwargs)`
:   Exponential Linear Unit shifted by user specified values.
    This helps to ensure the output to stay positive.
    
    Initialize the AdaptiveELU activation function.
    
    Args:
        xshift (float): shift in x direction
        yshift (float): shift in y direction
        inplace (bool): whether to modify the input tensor in place
        **kwargs: additional keyword arguments
    
    Returns:
        None

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass of the AdaptiveELU activation function.
        
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: output tensor

`Square()`
:   Square activation function.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass of the square activation function.
        
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: output tensor