Module NDNT.modules.layers.timelayers
=====================================

Classes
-------

`TimeLayer(start_time=0, end_time=1000, input_dims=None, num_bases=None, num_filters=None, filter_dims=None, pos_constraint=True, **kwargs)`
:   Layer to track experiment time and a weighted output.
    
    TimeLayer: Layer to track experiment time and a weighted output.
    
    Args:
        start_time: float, start time of experiment
        end_time: float, end time of experiment
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_bases: number of tent basis functions
        num_filters: number of output filters
        filter_dims: tuple or list of ints, (num_channels, height, width, lags)
        pos_constraint: bool, whether to enforce non-negative weights
        **kwargs: additional arguments to pass to NDNLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the TimeLayer.
        
        Args:
            x: torch.Tensor, input tensor
        
        Returns:
            y: torch.Tensor, output tensor

`TimeShiftLayer(input_dims=None, num_lags=1, **kwargs)`
:   Layer to shift in time dimension by num_lags
    
    TimeLayer: Layer to track experiment time and a weighted output.
    
    Args:
        input_dims: tuple or list of ints, (num_channels, height, width, lags/angles)
        num_lags: number of lags to shift back by
        **kwargs: additional arguments to pass to NDNLayer

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(input_dims=None, num_lags=1)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Shift in batch dimesnion by num_lags and pad by 0
        
        Args:
            x: torch.Tensor, input tensor
        
        Returns:
            y: torch.Tensor, output tensor