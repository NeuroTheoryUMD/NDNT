from numpy.lib.arraysetops import isin
import torch
from torch.nn import functional as F
from torch.nn import Parameter
import torch.nn as nn

from .ndnlayer import NDNLayer
import numpy as np


class LVLayer(NDNLayer):
    """
    Generates output at based on a number of LVs, sampled by time point.
    Each LV has T weights, and can be many LVs (so weight matrix is T x NLVs.
    Requires specifically formatted input that passes in indices of LVs and relative
    weight, so will linearly interpolate. Also, can have trial-structure so will have 
    temporal reg that does not need to go across trials
    
    Input will specifically be of form of (for each batch point):
    [index1, index2, w1]  and output will be w1*LV(i1) + (1-w1)*LV(i2)
    """

    def __init__(self, 
        num_time_pnts=None, 
        num_lvs=1,
        num_trials=None,
        norm_type=0,
        weights_initializer='normal',
        **kwargs):
        """
        If num_trials is None, then assumes one continuous sequence, otherwise will assume num_time_pnts
        is per-trial, and dimensionality of filter is [num_trials, 1, 1, num_time_pnts].
        
        Args:
            num_time_pnts: int, number of time points
            num_lvs: int, number of latent variables
            num_trials: int, number of trials
            norm_type: int, normalization type
            weights_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            **kwargs: keyword arguments to pass to the parent class
        """

        if num_trials is None:
            num_trials = 1

        if num_time_pnts is None:
            print('Making trial-level LVs')
            num_time_pnts = num_trials
            num_trials = 1
            self.use_tent_basis = False
        else:
            self.use_tent_basis = True
        
        super().__init__(
            filter_dims=[num_trials, 1, 1, num_time_pnts], norm_type=norm_type,
            num_filters=num_lvs, weights_initializer=weights_initializer,
            bias=False, **kwargs)
    # END LVLayer.__init__()
        
    def preprocess_weights(self):
        """
        Preprocesses weights by applying positivity constraint and normalization.

        Returns:
            w: torch.Tensor, preprocessed weights
        """
        if self.pos_constraint:
            #w = F.relu(self.weight)
            w = torch.square(self.weight)
        else:
            w = self.weight 

        if self.norm_type > 0:
            w = w / torch.std(w, axis=0).clamp(min=1e-6)

        return w
    # END preprocess_weights
            
    def forward( self, x ):
        """
        Assumes input of B x 3 (where three numbers are indexes of 2 surrounding LV and relative weight of LV1).
        
        Args:
            x: torch.Tensor, input tensor.

        Returns:
            y: torch.Tensor, output tensor.
        """
        weights = self.preprocess_weights()

        if self.use_tent_basis:
            w = torch.concat( (x[:, 2][:, None], 1-x[:,2][:, None]), dim=1)  # Relative weights of first LV indexed
            y = torch.einsum('tin,ti->tn', weights[x[:,:2].type(torch.int64), :], w ) 
            #if self.norm_type > 0:
                #y = torch.einsum('tin,ti->tn', self.weight[inds, :], w ) / torch.std(self.weight, axis=0).clamp(min=1e-6)
            #else:
                #y = torch.einsum('tin,ti->tn', self.weight[inds, :], w )
            #y = torch.sum( self.weight[inds, :]*w[:,:,None], axis=1 )  # seems slightly slower
        else:
            y = weights[x.type(torch.int64)]

        if self.NL is not None:
            y = self.NL(y)

        if self._ei_mask is not None:
            y = y * self._ei_mask

        return y
    # END LVLayer.forward()

    @classmethod
    def layer_dict(cls, num_time_pnts=None, num_lvs=1, num_trials=None, weights_initializer='normal', **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
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
        """
        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'LVlayer'
        # delete standard layer info for purposes of constructor
        #del Ldict['input_dims']
        del Ldict['bias_initializer']
        del Ldict['initialize_center']
        del Ldict['num_filters']
        del Ldict['bias']
        # Added arguments
        #Ldict['batch_sample'] = True
        #Ldict['fix_n_index'] = False
        Ldict['num_time_pnts'] = num_time_pnts
        Ldict['num_lvs'] = num_lvs
        Ldict['num_trials'] = num_trials
        Ldict['weights_initializer'] = weights_initializer
        Ldict['input_dims'] = [3, 1, 1, 1]
        return Ldict
# END LVLayer


class LVLayerOLD(NDNLayer):
    """
    No input. Produces LVs sampled from mu/sigma at each time step
    Could make tent functions for smoothness over time as well
    """

    def __init__(self, 
        num_time_pnts=None, 
        num_lvs=1,
        init_mu_range=0.5,
        init_sigma=1.0,
        sigma_shape='lv',  # 'full', 'lv' (one sigma for each LV), or 'time' (one sigma for each time)
        input_dims=[1,1,1,1],  # ignored if not entered, otherwise overwritten
        **kwargs):
        """
        Args:
            num_time_points
            num_lvs: default 1
            init_mu_range: default 0.1
            init_sigma: 
            sm_reg: smoothness regularization penalty
            gauss_type: isotropic
        """
        assert num_time_pnts is not None, "LVLayer: Must specify num_time_pnts explicitly"
        # Determine whether one- or two-dimensional fixation

        super().__init__(
            input_dims=input_dims, filter_dims=[1, 1, 1, num_time_pnts], 
            num_filters=num_lvs,
            bias=False, pos_constraint=False,
            **kwargs)

        self.numLVs = num_lvs
        self.nt = num_time_pnts

        # This makes weights NTxnum_dims     
        self.init_mu_range = init_mu_range
        self.init_sigma = init_sigma
        if sigma_shape == 'full':
            self.sigma_shape = [self.nt, self.numLVs]
        elif sigma_shape == 'lv':
            self.sigma_shape = [1, self.numLVs]
        elif sigma_shape == 'time':
            self.sigma_shape = [self.nt, 1]
        else:
            print( " LVLayer: sigma_shape argument not defined")

        if self.init_mu_range > 1.0 or self.init_mu_range < 0.0 or self.init_sigma <= 0.0:
            raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")
        
        # position grid shape
        #self.grid_shape = (1, 1, 1, self.numLVs)

        # initialize means and spreads
        self.sigma = Parameter(torch.Tensor(*self.sigma_shape))

        # Initialize values
        self.weight.data.uniform_(-self.init_mu_range, self.init_mu_range)
        self.sigma.data.fill_(self.init_sigma)

        self.sm_skips = []
        self.sample = True

        # In case normalizing 
        self.weight_scale = 100.0/self.nt  # average magnitude will be 0.1 over each LV

        # Currently will not use positive-contraint: should use nonlin=ReLu instead
        assert not self.pos_constraint, "LVLayer: should implement positive constraints through nonlinearity"
    # END LVLayer.__init__
            
    def forward(self, x):
        """
        Input has to be time-indices of what data is being indexed

        Args:
            x: time-indices of LVs
        
        Returns:
            z: output LVs
        """

        # Will normalize LVs if useful
        mus = self.preprocess_weights()

        # Sample (adapted from grid_sample)
        with torch.no_grad():
            #self.weight.clamp(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
            self.sigma.clamp(min=0)  # sigma/variance is always a positive quantity

        #sample = self.training if sample is None else self.sample
        #sigs = self.sigma[:, None, None, :]
        #mus = self.weight[:, None, None, :]
        nt = x.shape[0]
        #assert torch.max(x) < self.nt, "LVLayer: not enough LVs"
        x = x[:,0].detach().cpu().numpy().astype(np.int32)

        grid_shape = (nt, self.numLVs)
        if self.sample:
            norm = self.weight.new(*grid_shape).normal_()
        else:
            norm = self.weight.new(*grid_shape).zero_()  # for consistency and CUDA capability

        if self.sigma_shape[0] == 1:
            z = norm * self.sigma + mus[x, :]
        else:
            z = norm * self.sigma[x, :] + mus[x, :]

        if self.NL is not None:
            z = self.NL(z)

        return z


    @classmethod
    def layer_dict(cls, 
        num_time_pnts=None, init_mu=0.5, init_sigma=1.0,
        num_lvs=1, sigma_shape='lv', **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
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
        """
        assert num_time_pnts is not None, "LVLayer: Must specify num_time_pnts"
        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'LVlayer'
        # delete standard layer info for purposes of constructor
        #del Ldict['input_dims']
        del Ldict['weights_initializer']
        del Ldict['bias_initializer']
        del Ldict['initialize_center']
        del Ldict['pos_constraint']
       #del Ldict['input_dims']
        del Ldict['num_filters']
        del Ldict['bias']
        # Added arguments
        #Ldict['batch_sample'] = True
        #Ldict['fix_n_index'] = False
        Ldict['num_time_pnts'] = num_time_pnts
        Ldict['init_mu_range'] = init_mu
        Ldict['init_sigma'] = init_sigma
        Ldict['sigma_shape'] = sigma_shape
        Ldict['num_lvs'] = num_lvs
        Ldict['input_dims'] = [1,1,1,1]
        return Ldict
    # END [classmethod] WithinFixationLayer.layer_dict