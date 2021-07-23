import numpy as np
import torch
from torch import nn

from copy import deepcopy
#from .regularization import reg_setup_ffnet
from NDNlayer import *


class FFnetwork(nn.Module):

    #def __repr__(self):
    #    s = super().__repr__()
    #    # Add information about module to print out

    def __init__(self, ffnet_params):
        """ffnet_params is a dictionary constructed by other utility functions
        reg_params is a dictionary of reg type and list of values for each layer,
        i.e., {'d2xt':[None, 0.2], 'l1':[1e-4,None]}"""
        super(FFnetwork, self).__init__()

        # Format and record inputs into ffnet
        self.layer_list = deepcopy(ffnet_params['layer_list'])
        self.xstim_n = ffnet_params['xstim_n']
        self.ffnets_in = deepcopy(ffnet_params['ffnet_n'])
        self.conv = ffnet_params['conv']    

        num_layers = len(self.layer_list)

        # Check that first layer has matching input dims (to FFnetwork)
        if self.layer_list[0]['input_dims'] is None:
            self.layer_list[0]['input_dims'] = ffnet_params['input_dims']

        # Process regularization into layer-specific list. Will save at this level too
        
        reg_params = self.__reg_setup_ffnet( ffnet_params['reg_list'] )
        # can be saved, but difficult to update. just save reg vals within layers

        # Make each layer as part of an array
        self.layers = nn.ModuleList()
        for ll in range(num_layers):
            self.layers.append(
                NDNlayer(self.layer_list[ll], reg_vals=reg_params[ll]) )
        
    def forward(self, x):        
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __reg_setup_ffnet(self, reg_params=None):
        # Set all default values to none
        num_layers = len(self.layer_list)
        layer_reg_list = []
        for nn in range(num_layers):
            #layer_reg_list.append(deepcopy(_allowed_reg_types))
            layer_reg_list.append(deepcopy({}))  # only put regs in that are there

        # Set specific regularization
        if reg_params is not None:
            for kk, vv in reg_params.items():
                if not isinstance(vv, list):
                    vv = [vv]
                if len(vv) > num_layers:
                    print("Warning: reg params too long for", kk)
                for nn in range(np.minimum(num_layers, len(vv))):
                    layer_reg_list[nn][kk] = vv[nn]
        return layer_reg_list

    def prepare_regularization(self):
        """Makes regularization modules for training"""
        for layer in self.layers:
            layer.reg.build_reg_modules()

    def compute_reg_loss(self):
        rloss = 0
        for layer in self.layers:
            rloss += layer.compute_reg_loss()
        return rloss
