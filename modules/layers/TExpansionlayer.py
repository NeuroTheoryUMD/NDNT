import torch
import torch.nn as nn
from .ndnlayer import NDNLayer
import numpy as np
from torch.nn import functional as F
from torch.nn import Parameter

class TExpansionlayer(NDNLayer):
    def __init__(self, input_dims=None,
        num_filters=None,
        filter_dims=None,
        NLtype:str='lin',
        pos_constraint=0,
        num_inh:int=0,
        bias:bool=False,
        weights_initializer:str='xavier_uniform',
        #output_norm=None,
        initialize_center=False,
        bias_initializer:str='zeros',
        reg_vals:dict=None,
        frac = 1,
        crt = False,
        **kwargs,
        ):
            super().__init__(input_dims=input_dims,num_filters=num_filters,filter_dims=filter_dims, reg_vals=reg_vals,
                              initialize_center=initialize_center, weights_initializer=weights_initializer, num_inh = num_inh,
                              NLtype=NLtype, pos_constraint=pos_constraint, bias=bias, bias_initializer=bias_initializer, **kwargs)
            self.frac = frac
            self.crt = crt


    @classmethod
    def layer_dict(cls, frac = 1, crt = False, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'TEx_layer'
        Ldict['frac'] = frac
        Ldict['crt'] = crt

        return Ldict

    def forward(self,x):
        y = super().forward(x)

        if self.frac > 1:
            if self.crt:
                z = torch.zeros_like(y)
                z = z.repeat_interleave(self.frac, dim=0)
                z[::self.frac] = y
                y = z
            else:
                z = y.repeat_interleave(self.frac, dim=0)
                y = z
                
        return y
    
