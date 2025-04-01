Module NDNT.modules.layers.lvlayers
===================================

Classes
-------

`LVLayer(num_time_pnts=None, num_lvs=1, num_trials=None, norm_type=0, weights_initializer='normal', **kwargs)`
:   Generates output at based on a number of LVs, sampled by time point.
    Each LV has T weights, and can be many LVs (so weight matrix is T x NLVs.
    Requires specifically formatted input that passes in indices of LVs and relative
    weight, so will linearly interpolate. Also, can have trial-structure so will have 
    temporal reg that does not need to go across trials
    
    Input will specifically be of form of (for each batch point):
    [index1, index2, w1]  and output will be w1*LV(i1) + (1-w1)*LV(i2)
    
    If num_trials is None, then assumes one continuous sequence, otherwise will assume num_time_pnts
    is per-trial, and dimensionality of filter is [num_trials, 1, 1, num_time_pnts].
    
    Args:
        num_time_pnts: int, number of time points
        num_lvs: int, number of latent variables
        num_trials: int, number of trials
        norm_type: int, normalization type
        weights_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        **kwargs: keyword arguments to pass to the parent class

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(num_time_pnts=None, num_lvs=1, num_trials=None, weights_initializer='normal', **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            num_time_pnts: int, number of time points
            num_lvs: int, number of latent variables
            num_trials: int, number of trials
            weights_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            **kwargs: keyword arguments to pass to the parent class

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Assumes input of B x 3 (where three numbers are indexes of 2 surrounding LV and relative weight of LV1).
        
        Args:
            x: torch.Tensor, input tensor.
        
        Returns:
            y: torch.Tensor, output tensor.

    `preprocess_weights(self)`
    :   Preprocesses weights by applying positivity constraint and normalization.
        
        Returns:
            w: torch.Tensor, preprocessed weights

`LVLayerOLD(num_time_pnts=None, num_lvs=1, init_mu_range=0.5, init_sigma=1.0, sigma_shape='lv', input_dims=[1, 1, 1, 1], **kwargs)`
:   No input. Produces LVs sampled from mu/sigma at each time step
    Could make tent functions for smoothness over time as well
    
    Args:
        num_time_points
        num_lvs: default 1
        init_mu_range: default 0.1
        init_sigma: 
        sm_reg: smoothness regularization penalty
        gauss_type: isotropic

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(num_time_pnts=None, init_mu=0.5, init_sigma=1.0, num_lvs=1, sigma_shape='lv', **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            num_time_pnts: int, number of time points
            num_lvs: int, number of latent variables
            init_mu: float, initial mean value
            init_sigma: float, initial sigma value
            sigma_shape: str, 'full', 'lv', or 'time'
            **kwargs: keyword arguments to pass to the parent class
        
        Returns:
            Ldict: dict, dictionary of layer parameters

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Input has to be time-indices of what data is being indexed
        
        Args:
            x: time-indices of LVs
        
        Returns:
            z: output LVs