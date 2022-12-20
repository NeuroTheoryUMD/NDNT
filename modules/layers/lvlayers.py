from numpy.lib.arraysetops import isin
import torch
from torch.nn import functional as F
from torch.nn import Parameter
import torch.nn as nn

from .ndnlayer import NDNLayer
import numpy as np

class LVLayer(NDNLayer):
    """No input. Produces LVs sampled from mu/sigma at each time step
    Could make tent functions for smoothness over time as well
    Inputs:
        num_time_points
        num_lvs: default 1
        init_mu_range: default 0.1
        init_sigma: 
        sm_reg: smoothness regularization penalty
        gauss_type: isotropic
    """

    def __init__(self, 
        num_time_pnts=None, 
        num_lvs=1,
        init_mu_range=0.5,
        init_sigma=1.0,
        sigma_shape='lv',  # 'full', 'lv' (one sigma for each LV), or 'time' (one sigma for each time)
        **kwargs):

        assert num_time_pnts is not None, "LVLayer: Must specify num_time_pnts explicitly"
        # Determine whether one- or two-dimensional fixation
        self.numLVs = num_lvs
        self.nt = num_time_pnts

        super().__init__(
            filter_dims=[1, 1, 1, self.nt], 
            num_filters=self.numLVs,
            bias=False, pos_constraint=False,
            **kwargs)

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
        self.grid_shape = (1, 1, 1, self.numLVs)

        # initialize means and spreads
        self.sigma = Parameter(torch.Tensor(*self.sigma_shape))

        # Initialize values
        self.weight.data.uniform_(-self.init_mu_range, self.init_mu_range)
        self.sigma.data.fill_(self.init_sigma)

        self.sm_skips = []
        self.sample = True

        # Currently will not use positive-contraint: should use nonlin=ReLu instead
        assert not self.pos_constraint, "LVLayer: should implement positive constraints through nonlinearity"
        assert self.norm_type == 0, "LVLayer: normalization is not currently implemented"
    # END LVLayer.__init__
            
    def forward(self, x):
        """
        Input has to be time-indices of what data is being indexed
            x: time-indices of LVs
            z: output LVs
        """
        #assert x is None, "LVLayer: note this currently takes no input"
        # Ignore input x -- 

        # Sample (adapted from grid_sample)
        with torch.no_grad():
            #self.weight.clamp(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
            self.sigma.clamp(min=0)  # sigma/variance is always a positive quantity

        #sample = self.training if sample is None else self.sample
        #sigs = self.sigma[:, None, None, :]
        #mus = self.weight[:, None, None, :]
        nt = x.shape[0]
        assert torch.max(x) < self.nt, "LVLayer: not enough LVs"
        x = x[:,0].detach().numpy().astype(np.int32)

        grid_shape = (nt, self.numLVs)
        if self.sample:
            norm = self.weight.new(*grid_shape).normal_()
        else:
            norm = self.weight.new(*grid_shape).zero_()  # for consistency and CUDA capability

        if self.sigma_shape[0] == 1:
            z = norm * self.sigma + self.weight[x, :]
        else:
            z = norm * self.sigma[x, :] + self.weight[x, :]

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
        """
        assert num_time_pnts is not None, "LVLayer: Must specify num_time_pnts"
        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'LVlayer'
        # delete standard layer info for purposes of constructor
        #del Ldict['input_dims']
        del Ldict['norm_type']
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