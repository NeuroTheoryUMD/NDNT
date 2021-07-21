import numpy as np
import torch
from torch import nn
from pytorch_lightning import LightningModule
#from torch.nn import functional as F

#from torch import Tensor
#from torch.nn.parameter import Parameter
#from torch.nn import init
#from torch.nn.common_types import _size_2_t, _size_3_t # for conv2,conv3 default
#from torch.nn.modules.utils import _triple # for posconv3

from copy import deepcopy
from NDNlayer import *


class FFnetwork(LightningModule):

    def __repr__(self):
        s = super().__repr__()
        # Add information about module to print out

    def __init__(self, ffnet_params):
        super(FFnetwork, self).__init__()

        # Format and record inputs into ffnet
        self.layer_list = deepcopy(ffnet_params['layer_list'])
        self.xstim = ffnet_params['xstim_n']
        self.ffnets_in = deepcopy(ffnet_params['ffnet_n'])
        self.conv = ffnet_params['conv']    

        num_layers = len(self.layer_list)

        # Check that first layer has matching input dims (to FFnetwork)
        if self.layer_list[0]['input_dims'] is None:
            self.layer_list[0]['input_dims'] = ffnet_params['input_dims']

        # Make each layer as part of an array
        self.layers = nn.ModuleList()
        for ll in range(num_layers):
            self.layers.append(
                NDNlayer(self.layer_list[ll]) )

    def forward(self, x):        
        for ll in range(self.layers):
            x = self.layer[ll](x)
        return x
    
