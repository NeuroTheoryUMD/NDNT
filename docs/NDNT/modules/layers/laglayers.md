Module NDNT.modules.layers.laglayers
====================================

Classes
-------

`LagLayer(input_dims=None, num_filters=None, num_lags=None, **kwargs)`
:   Operates spatiotemporal filter om time-embeeded stimulus
    Spatiotemporal filter should have less lags than stimulus -- then ends up with some lags left
    Filter is full spatial width and the number of lags is explicity specified in initializer
    (so, inherits spatial and chanel dimensions of stimulus input)
    
    Args (required):
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        num_lags: number of lags in spatiotemporal filter
    
    Args (optional):
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
    
    Initialize the layer.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: int, number of output filters
        num_lags: int, number of lags in spatiotemporal filter
        **kwargs: keyword arguments to pass to the parent class

    ### Ancestors (in MRO)

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
            **kwargs: keyword arguments to pass to the parent class
        
        Returns:
            Ldict: dict, dictionary of layer parameters

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the layer.
        
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, *input_dims)
        
        Returns:
            y: torch.Tensor, output tensor of shape (batch_size, *output_dims)

    `plot_filters(self, cmaps='gray', num_cols=8, row_height=2, time_reverse=False)`
    :   Overload plot_filters to automatically time_reverse.
        
        Args:
            cmaps: str or list of str, colormap(s) to use for plotting
            num_cols: int, number of columns in the plot
            row_height: int, height of each row in the plot
            time_reverse: bool, whether to reverse the time axis
        
        Returns:
            fig: matplotlib.figure.Figure, the figure object
            axs: list of matplotlib.axes.Axes, the axes objects