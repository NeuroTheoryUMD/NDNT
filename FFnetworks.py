import numpy as np
import torch
from torch import nn

from copy import deepcopy
#from .regularization import reg_setup_ffnet
from NDNlayer import *

LayerTypes = {
    'normal': NDNlayer,
    'conv': ConvLayer,
    'readout': ReadoutLayer,
    'divnorm': DivNormLayer,
    'stconv': STconvLayer,
    'external': ExternalLayer
}

_valid_ffnet_types = ['normal', 'add', 'mult', 'readout']

class FFnetwork(nn.Module):

    #def __repr__(self):
    #    s = super().__repr__()
    #    # Add information about module to print out

    def __init__(self, ffnet_params):
        """ffnet_params is a dictionary constructed by other utility functions
        reg_params is a dictionary of reg type and list of values for each layer,
        i.e., {'d2xt':[None, 0.2], 'l1':[1e-4,None]}"""

        super(FFnetwork, self).__init__()
        
        self.network_type = ffnet_params['ffnet_type']
        assert self.network_type in _valid_ffnet_types, "ffnet_type " + self.network_type + " is unknown."

        # Format and record inputs into ffnet
        self.layer_list = deepcopy(ffnet_params['layer_list'])
        self.layer_types = deepcopy(ffnet_params['layer_types'])
        self.xstim_n = ffnet_params['xstim_n']
        self.ffnets_in = deepcopy(ffnet_params['ffnet_n'])
        #if self.network_type == 'mult': 
        #    assert len(self.ffnets_in) == 2, 'mult ffnetwork must receive two input networks.'  

        assert self.determine_input_dims(ffnet_params['input_dims_list'], ffnet_type=ffnet_params['ffnet_type']), 'Invalid network inputs.'
        #self.conv = ffnet_params['conv']    # I don't think this does anything

        num_layers = len(self.layer_list)

        # Check that first layer has matching input dims (to FFnetwork)
        if self.layer_list[0]['input_dims'] is None:
            self.layer_list[0]['input_dims'] = self.input_dims

        # Process regularization into layer-specific list. Will save at this level too
        
        reg_params = self.__reg_setup_ffnet( ffnet_params['reg_list'] )
        # can be saved, but difficult to update. just save reg vals within layers

        # Make each layer as part of an array
        self.layers = nn.ModuleList()
        for ll in range(num_layers):
            layer_type = self.layer_types
            self.layers.append(
                LayerTypes[self.layer_types[ll]](self.layer_list[ll], reg_vals=reg_params[ll]) )
    # END FFnetwork.__init__
 
    def determine_input_dims( self, input_dims_list, ffnet_type='normal' ):
        """
        Sets input_dims given network inputs. Can be overloaded depending on the network type. For this base class, there
        are two types of network input: external stimulus (xstim_n) or a list of internal (ffnet_in) networks:
            For external inputs, it just uses the passed-in input_dims
            For internal network inputs, it will concatenate inputs along the filter dimension, but MUST match other dims
        As currently designed, this can either external or internal, but not both
        
        This sets the following internal FFnetwork properties:
            self.input_dims
            self.input_dims_list
        and returns Boolean whether the passed in input dims are valid
        """

        valid_input_dims = True
        if self.ffnets_in is None:
            # then external input (assume from one source)
            assert len(input_dims_list) == 1, "FFnet constructor: Only one set of input dims can be specified."
            assert input_dims_list[0] is not None, "FFnet constructor: External input dims must be specified."
            self.input_dims = input_dims_list[0]
        else: 
            num_input_networks = len(self.ffnets_in)
            assert len(input_dims_list) == num_input_networks, 'Internal: misspecification of input_dims for FFnetwork.'

            # Go through the input dims of the other ffnetowrks to verify they are valid for the type of network    
            for ii in range(num_input_networks):
                if ii == 0:
                    num_cat_filters = input_dims_list[0][0]
                else:
                    if input_dims_list[ii][1:] != input_dims_list[0][1:]:
                        valid_input_dims = False
                        print("FFnet: invalid concatenation %d:"%ii, input_dims_list[ii][1:], input_dims_list[0][1:] )
                    else:
                        if ffnet_type == 'normal': # then inputs will be concatenated along 'filter' dimension
                            num_cat_filters += input_dims_list[ii][0]
                        else:  # these are combined and input to first layer has same size as one input
                            assert input_dims_list[ii][0] == num_cat_filters, 'Input dims must be the same for' + ffnet_type + 'ffnetwork'
                        
            self.input_dims = [num_cat_filters] + input_dims_list[0][1:]
        
        self.input_dims_list = deepcopy(input_dims_list)
        return valid_input_dims
    # END FFnetwork.determine_input_dims

    def forward(self, inputs):   

        # Combine network inputs (if relevant)
        if isinstance(inputs, list):
            x = inputs[0]
            for mm in range(1, len(inputs)):
                if self.network_type == 'normal': # concatentate inputs
                    x = torch.cat( (x, inputs[mm]), 1 )
                elif self.network_type == 'add': # add inputs
                    x = torch.add( x, inputs[mm] )
                elif self.network_type == 'mult': # multiply: (input1) x (1+input2)
                    x = torch.mult( x, torch.add(inputs[mm], 1.0) )
        else:
            x = inputs

        # Process through layers
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

    def list_params(self, layer_target=None):
        if layer_target is None:
            layer_target = np.arange(len(self.layers), dtype='int32')
        elif not isinstance(layer_target, list):
            layer_target = [layer_target]
        for nn in layer_target:
            assert nn < len(self.layers), '  Invalid layer %d.'%nn
            print("  Layer %d:"%nn)
            self.layers[nn].list_params()

    def set_params(self, layer_target=None, name=None, val=None ):
        """Set parameters for listed layer or for all layers."""
        if layer_target is None:
            layer_target = np.arange(len(self.layers), dtype='int32')
        elif not isinstance(layer_target, list):
            layer_target = [layer_target]
        for nn in layer_target:
            assert nn < len(self.layers), '  Invalid layer %d.'%nn
            self.layers[nn].set_params(name=name, val=val)
    # END FFnetwork class
class ReadoutNetwork(FFnetwork):
    """
    A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
    and covariance of that Gaussian are learned.

    Args:
        in_shape (list, tuple): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]. Default: 0.1
        init_sigma (float): The standard deviation of the Gaussian with `init_sigma` when `gauss_type` is
            'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the
            covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        gauss_type (str): Which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
        shifter (dict): Parameters for a predictor of shfiting grid locations. Has to have a form like
                        {
                        'hidden_layers':1,
                        'hidden_features':20,
                        'final_tanh': False,
                        }
"""
    #def __repr__(self):
    #    s = super().__repr__()
    #    # Add information about module to print out

    def __init__(self, ffnet_params):
        """
        This essentially used the constructor for Point1DGaussian, with dicationary input.
        Currently there is no extra code required at the network level. I think the constructor
        can be left off entirely, but leaving in in case want to add something.
        """
        super(ReadoutNetwork, self).__init__(ffnet_params)
        self.network_type = 'readout'

    def determine_input_dims( self, input_dims_list, ffnet_type='readout' ):
        """
        Sets input_dims given network inputs. Can be overloaded depending on the network type. For this base class, there
        are two types of network input: external stimulus (xstim_n) or a list of internal (ffnet_in) networks:
            For external inputs, it just uses the passed-in input_dims
            For internal network inputs, it will concatenate inputs along the filter dimension, but MUST match other dims
        As currently designed, this can either external or internal, but not both
        
        This sets the following internal FFnetwork properties:
            self.input_dims
            self.input_dims_list
        and returns Boolean whether the passed in input dims are valid
        """

        valid_input_dims = True
        if self.ffnets_in is None:
            # then external input (assume from one source)
            valid_input_dims = False
            print('Readout layer cannot get an external input.') 
            self.input_dims = input_dims_list[0]
        else:             
            assert len(input_dims_list) == len(self.ffnets_in), 'Internal: misspecification of input_dims for FFnetwork.'
            # First dimension is the input network
            self.input_dims = input_dims_list[0]
            # Second dimension would be 
            self.shifter = len(self.ffnets_in) > 1

        return valid_input_dims
    # END ReadoutNetwork.determine_input_dims

    def forward(self, inputs):
        """network inputs correspond to output of conv layer, and (if it exists), a shifter""" 
        if self.shifter:
            y = self.layers[0](inputs[0], shift=inputs[1])
        else:
            y = self.layers[0](inputs[0])
        return y

    def get_readout_locations(self):
        return self.layers[0].get_readout_locations()

    def set_readout_locations(self, locs):
        self.layers[0].set_readout_locations(locs)


class FFnet_external(FFnetwork):
    """This is a 'shell' that lets an external network be plugged into the NDN. It establishes all the basics
    so that information requested to this network from other parts of the NDN will behave correctly.

"""
    #def __repr__(self):
    #    s = super().__repr__()
    #    # Add information about module to print out

    def __init__(self, ffnet_params, external_module_dict):

        # The parent construct will make a 'dummy layer' that will be filled in with module 0 below
        super(FFnet_external, self).__init__(ffnet_params)
        self.network_type = 'external'

        # Extract relevant network fom extenal_module_dict using the ffnet_params['layer_types']
        net_name = ffnet_params['external_module_name']
        assert net_name in external_module_dict, 'External network %s not found in external_modules dict.'%net_name

        # This network will be made to be a layer (so the ffnet forward is the layer forward). Now place external network here
        self.layers[0].external_network = external_module_dict[net_name]
        self.input_dims_reshape = ffnet_params['input_dims_reshape']
    # END FFnet_external.__init__

    def forward(self, inputs):   
        # Leave all heavy lifting to the external module, which is in layers[0]. But concatenate network inputs, as needed
        x = inputs[0]
        for mm in range(1, len(inputs)):
            x = torch.cat( (x, inputs[mm]), 1 )
        batch_size = x.shape[0]

        # Reshape dimensions for layer as needed
        if self.input_dims_reshape is not None:
            x = torch.reshape( x, [-1] + self.input_dims_reshape)
        
        # Pass into external network
        y = self.layers[0](x)

        # Ensure that output is flattened
        return y.reshape((batch_size, -1))
    
    def compute_reg_loss(self):
        # Since we do not implement regularization within the external network, this returns nothing
        return 0

    def list_params(self, layer_target=None):
        assert layer_target is None, 'No ability to directly distinguish layers in the external network.'
        for nm, pp in self.named_parameters(recurse=True):
            if pp.requires_grad:
                print("    %s:"%nm, pp.size())
            else:
                print("    NOT FIT: %s:"%nm, pp.size())

    def set_params(self, layer_target=None, name=None, val=None ):
        assert layer_target is None, 'No ability to directly distinguish layers in the external network.'
        assert isinstance(val, bool), 'val must be set.'
        for nm, pp in self.named_parameters(recurse=True):
            if name is None:
                pp.requires_grad = val
            elif nm == name:
                pp.requires_grad = val
