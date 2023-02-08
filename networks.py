import numpy as np
import torch
from torch import nn

from functools import reduce
from copy import deepcopy

import NDNT.modules.layers as layers

LayerTypes = {
    'normal': layers.NDNLayer,
    'conv': layers.ConvLayer,
    'divnorm': layers.DivNormLayer,
    'tconv': layers.TconvLayer,
    'stconv': layers.STconvLayer,
    'biconv': layers.BiConvLayer1D,
    'readout': layers.ReadoutLayer,
    'fixation': layers.FixationLayer,
    'lag': layers.LagLayer,
    'time': layers.TimeLayer,
    'dim0': layers.Dim0Layer,
    'dimSP': layers.DimSPLayer,
    'channel': layers.ChannelLayer,
    'LVlayer': layers.LVLayer
    # 'external': layers.ExternalLayer,    
}

_valid_ffnet_types = ['normal', 'add', 'mult', 'readout', 'scaffold']
      
class FFnetwork(nn.Module):

    def __init__(self,
            layer_list: list = None,
            ffnet_type: str = 'normal',
            #layer_types: list = None,
            xstim_n: str = 'stim',
            ffnet_n: list = None,
            input_dims_list: list = None,
            reg_list: list = None,
            scaffold_levels: list = None,
            **kwargs,
            ):

        # if len(kwargs) > 0:
        #     print("FFnet: unknown kwargs:", kwargs)
        assert layer_list is not None, "FFnetwork: Must supply a layer_list."
        
        super().__init__()
        
        self.network_type = ffnet_type
        #print("FFnet: network type:", self.network_type)
        assert self.network_type in _valid_ffnet_types, "ffnet_type " + self.network_type + " is unknown."

        # Format and record inputs into ffnet
        self.layer_list = deepcopy(layer_list)
        self.layer_types = []   # read from layer_list (if necessary at all)
        self.xstim_n = xstim_n
        self.ffnets_in = ffnet_n

        num_layers = len(self.layer_list)

        if num_layers == 0:
            self.layers = nn.ModuleList()
            return
            
        # Establish input dims from the network
        if input_dims_list is None:
            # then pull from first layer
            assert layer_list[0]['input_dims'] is not None, "If input_dims is not specified, it must be specified in layer-0"
            input_dims_list = [deepcopy(layer_list[0]['input_dims'])]
        
        # Build input_dims from sources
        assert self.determine_input_dims(input_dims_list, ffnet_type=ffnet_type), 'Invalid network inputs.'

        # Process regularization into layer-specific list. Will save at this level too
        #if reg_list is not None:  # can also be entered into layers directly
        #    reg_params = self.__reg_setup_ffnet( reg_list )

        # Make each layer as part of an array
        self.layers = nn.ModuleList()
        for ll in range(num_layers):
            if self.layer_list[ll]['input_dims'] is None:
                if ll == 0:
                    self.layer_list[ll]['input_dims'] = deepcopy(self.input_dims)
                else:
                    self.layer_list[ll]['input_dims'] = deepcopy(self.layers[ll-1].output_dims)
            Ltype = self.layer_list[ll]['layer_type']
            self.layers.append( LayerTypes[Ltype](**self.layer_list[ll]) )

        # output dims determined by last layer
        self.output_dims = self.layers[-1].output_dims

        # Make scaffold output if requested
        if scaffold_levels is None:
            self.scaffold_levels = [-1] # output last layer only
        else: # output specified layers concatenated together
            self.scaffold_levels = [*range(self.layers)[scaffold_levels:]] if isinstance(scaffold_levels, int) else scaffold_levels
    # END FFnetwork.__init__
 
    @property
    def num_outputs(self):
        n = 0
        for i in self.scaffold_levels:
            n += self.layers[i].output_dims
        return n

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
                        if ffnet_type == 'add':
                            for jj in range(2):
                                #if (input_dims_list[ii][jj+1] > 1) | (input_dims_list[0][jj+1] == 1):
                                if (input_dims_list[ii][jj+1] > 1) & (input_dims_list[0][jj+1] > 1):
                                    valid_input_dims = False
                        else:
                            valid_input_dims = False
                    assert valid_input_dims, print("FFnet: invalid concatenation %d:"%ii, input_dims_list[ii][1:], input_dims_list[0][1:] )

                    if ffnet_type == 'normal': # then inputs will be concatenated along 'filter' dimension
                        num_cat_filters += input_dims_list[ii][0]
                    elif input_dims_list[ii][0] > 1:
                        # these are combined and input to first layer has same size as one input
                        assert input_dims_list[ii][0] == num_cat_filters, 'Input dims must be the same for ' + ffnet_type + ' ffnetwork'
                    
            self.input_dims = [num_cat_filters] + input_dims_list[0][1:]
        
        self.input_dims_list = deepcopy(input_dims_list)
        return valid_input_dims
    # END FFnetwork.determine_input_dims

    def preprocess_input(self, inputs):
        """
        Preprocess input to network.
        """
        # Combine network inputs (if relevant)
        if isinstance(inputs, list):
            if len(inputs) == 1:
                x = inputs[0]
            else:
                x = inputs[0].view([-1]+self.input_dims_list[0])  # this will allow for broadcasting
                nt = inputs[0].shape[0]
                for mm in range(1, len(inputs)):
                    if self.network_type == 'normal': # concatentate inputs
                        x = torch.cat( (x, inputs[mm].view([-1]+self.input_dims_list[mm])), 1 )
                    elif self.network_type == 'add': # add inputs
                        x = torch.add( x, inputs[mm].view([-1]+self.input_dims_list[mm]) )
                    elif self.network_type == 'mult': # multiply: (input1) x (1+input2)
                        x = torch.multiply(
                            x, torch.add(inputs[mm].view([-1]+self.input_dims_list[mm]), 1.0).clamp(min=0.0) )
                        # Make sure multiplication is not negative
                x = x.view([nt, -1])
        else:
            x = inputs
        return x

    def forward(self, inputs):
        if self.layers is None:
            raise ValueError("FFnet: no layers defined.")
        
        out = [] # returned 

        x = self.preprocess_input(inputs)

        for layer in self.layers:
            x = layer(x)
            #out.append(x)

        return x
        #return torch.cat([out[ind] for ind in self.scaffold_levels], dim=1)
    # END FFnetwork.forward

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
        """
        Makes regularization modules with current requested values.
        This is done immediately before training, because it can change during training and tuning.
        """
        for layer in self.layers:
            if hasattr(layer, 'reg'):
                layer.reg.build_reg_modules()
            
    def compute_reg_loss(self):
        rloss = []
        for layer in self.layers:
            rloss.append(layer.compute_reg_loss())
        return reduce(torch.add, rloss)

    def list_parameters(self, layer_target=None):
        if layer_target is None:
            layer_target = np.arange(len(self.layers), dtype='int32')
        elif not isinstance(layer_target, list):
            layer_target = [layer_target]
        for nn in layer_target:
            assert nn < len(self.layers), '  Invalid layer %d.'%nn
            print("  Layer %d:"%nn)
            self.layers[nn].list_parameters()

    def set_parameters(self, layer_target=None, name=None, val=None ):
        """Set parameters for listed layer or for all layers."""
        if layer_target is None:
            layer_target = np.arange(len(self.layers), dtype='int32')
        elif not isinstance(layer_target, list):
            layer_target = [layer_target]
        for nn in layer_target:
            assert nn < len(self.layers), '  Invalid layer %d.'%nn
            self.layers[nn].set_parameters(name=name, val=val)

    def set_reg_val(self, reg_type=None, reg_val=None, layer_target=None ):
        """Set reg_values for listed layer or for all layers."""
        if layer_target is None:
            layer_target = 0
        assert layer_target < len(self.layers), "layer target too large (max = %d)"%len(self.layers)
        self.layers[layer_target].set_reg_val( reg_type=reg_type, reg_val=reg_val )

    def plot_filters(self, layer_target=0, **kwargs):
        self.layers[layer_target].plot_filters(**kwargs)

    def get_weights(self, layer_target=0, **kwargs):
        """passed down to layer call, with optional arguments conveyed"""
        assert layer_target < len(self.layers), "Invalid layer_target %d"%layer_target
        return self.layers[layer_target].get_weights(**kwargs)

    @classmethod
    def ffnet_dict( cls, layer_list=None, xstim_n ='stim', ffnet_n=None, ffnet_type='normal', **kwargs):
        return {'layer_list': deepcopy(layer_list), 'xstim_n':xstim_n, 'ffnet_n':ffnet_n, 'ffnet_type': ffnet_type }
    # END FFnetwork class


class ScaffoldNetwork(FFnetwork):
    """Concatenates output of all layers together in filter dimension, preserving spatial dims  """

    def __repr__(self):
        s = super().__repr__()
        # Add information about module to print out
        s += self.__class__.__name__
        return s

    def __init__(self, scaffold_levels=None, num_lags_out=1, **kwargs):
        """
        This essentially used the constructor for Point1DGaussian, with dicationary input.
        Currently there is no extra code required at the network level. I think the constructor
        can be left off entirely, but leaving in in case want to add something.
        """
        super().__init__(**kwargs)
        self.network_type = 'scaffold'

        self.num_lags_out = num_lags_out

        num_layers = len(self.layers)
        if scaffold_levels is None:
            self.scaffold_levels = np.arange(num_layers)
        else:
            if isinstance(scaffold_levels, list):
                scaffold_levels = np.array(scaffold_levels, dtype=np.int64)
            self.scaffold_levels = scaffold_levels 
        # Determine output dimensions
        #assert self.layers[self.scaffold_levels[0]].output_dims[3] == 1, "Scaffold: cannot currently handle lag dimensions"
        self.spatial_dims = self.layers[self.scaffold_levels[0]].output_dims[1:3]
        self.filter_count = np.zeros(len(self.scaffold_levels))
        self.filter_count[0] = self.layers[self.scaffold_levels[0]].output_dims[0]

        #Tchomps = np.zeros(self.scaffold_levels)
        for ii in range(1, len(self.scaffold_levels)):
            assert self.layers[self.scaffold_levels[ii]].output_dims[1:3] == self.spatial_dims, "Spatial dims problem layer %d"%self.scaffold_levels[ii] 
            #assert self.layers[self.scaffold_levels[ii]].output_dims[3] == 1, "Scaffold: cannot currently handle lag dimensions"
            #if self.layers[self.scaffold_levels[ii]].output_dims[3] > 1:
            #    Tchomps[ii] = self.layers[self.scaffold_levels[ii]].output_dims[3]-1
            self.filter_count[ii] = self.layers[self.scaffold_levels[ii]].output_dims[0]

        # Construct output dimensions
        self.output_dims = [int(np.sum(self.filter_count))] + self.spatial_dims + [self.num_lags_out]
    # END ScaffoldNetwork.__init__

    def forward(self, inputs):
        if self.layers is None:
            raise ValueError("Scaffold: no layers defined.")
        
        out = [] # returned 
        x = self.preprocess_input(inputs)

        for layer in self.layers:
            x = layer(x)
            nt = x.shape[0]
            if layer.output_dims[3] > self.num_lags_out:
                # Need to return just first lag (lag0) -- 'chomp'
                y = x.reshape([nt, -1, layer.output_dims[3]])[..., :(self.num_lags_out)]
                out.append( y.reshape((nt, -1) ))
            else:
                out.append(x)
        
        return torch.cat([out[ind] for ind in self.scaffold_levels], dim=1)
    # END ScaffoldNetwork.forward()

    @classmethod
    def ffnet_dict( cls, scaffold_levels=None, num_lags_out=1, **kwargs):
        ffnet_dict = super().ffnet_dict(**kwargs)
        ffnet_dict['ffnet_type'] = 'scaffold'
        ffnet_dict['scaffold_levels'] = scaffold_levels
        ffnet_dict['num_lags_out'] = num_lags_out
        return ffnet_dict


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
    def __repr__(self):
        s = super().__repr__()
        # Add information about module to print out
        s += self.__class__.__name__
        return s

    def __init__(self, **kwargs):
        """
        This essentially used the constructor for Point1DGaussian, with dicationary input.
        Currently there is no extra code required at the network level. I think the constructor
        can be left off entirely, but leaving in in case want to add something.
        """
        super().__init__(**kwargs)
        self.network_type = 'readout'
        # Make sure first type is readout: important for interpretation of input dims and potential shifter
        assert kwargs['layer_list'][0]['layer_type'] == 'readout', "READOUT NET: Incorrect leading layer type"

    def determine_input_dims( self, input_dims_list, **kwargs):
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

        if not isinstance(inputs, list):
            inputs = [inputs]

        if self.shifter:
            y = self.layers[0](inputs[0], shift=inputs[1])
        else:
            y = self.layers[0](inputs[0])
        return y
    # END ReadoutNetwork.forward

    def get_readout_locations(self):
        return self.layers[0].get_readout_locations()

    def set_readout_locations(self, locs):
        self.layers[0].set_readout_locations(locs)

    @classmethod
    def ffnet_dict( cls, ffnet_n=0, **kwargs):
        ffnet_dict = super().ffnet_dict(xstim_n=None, ffnet_n=ffnet_n, **kwargs)
        ffnet_dict['ffnet_type'] = 'readout'
        return ffnet_dict
    # END ReadoutNetwork


class FFnet_external(FFnetwork):
    """This is a 'shell' that lets an external network be plugged into the NDN. It establishes all the basics
    so that information requested to this network from other parts of the NDN will behave correctly."""
    #def __repr__(self):
    #    s = super().__repr__()
    #    # Add information about module to print out

    def __init__(self, external_module_dict=None, external_module_name=None, input_dims_reshape=None, **kwargs):

        # The parent construct will make a 'dummy layer' that will be filled in with module 0 below
        super(FFnet_external, self).__init__(**kwargs)
        self.network_type = 'external'

        # Extract relevant network fom extenal_module_dict using the ffnet_params['layer_types']
        assert external_module_dict is not None, 'external_module_dict cannot be None.'
        
        net_name = external_module_name
        assert net_name in external_module_dict, 'External network %s not found in external_modules dict.'%net_name

        # This network will be made to be a layer (so the ffnet forward is the layer forward). Now place external network here
        self.layers[0].external_network = external_module_dict[net_name]
        assert input_dims_reshape is not None, 'input_dims_reshape cannot be None. Jake did not know what it is supposed to default to so he used None.'
        self.input_dims_reshape = input_dims_reshape
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

    @classmethod
    def ffnet_dict( cls, **kwargs):
        ffnet_dict = super().ffnet_dict(**kwargs)
        ffnet_dict['ffnet_type'] = 'external'
        return ffnet_dict