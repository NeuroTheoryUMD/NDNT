import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from functools import reduce
from copy import deepcopy

from NDNT.modules import layers


_valid_ffnet_types = ['normal', 'add', 'mult', 'readout', 'scaffold', 'scaffold3d']
      
class FFnetwork(nn.Module):
    """
    Initializes an instance of the network.

    Args:
        layer_list (list, optional): A list of dictionaries representing the layers of the network. Defaults to None.
        ffnet_type (str, optional): The type of the feedforward network. Defaults to 'normal'.
        xstim_n (str, optional): The name of the stimulus input. Defaults to 'stim'.
        ffnet_n (list, optional): A list of feedforward networks. Defaults to None.
        input_dims_list (list, optional): A list of input dimensions for each layer. Defaults to None.
        reg_list (list, optional): A list of regularization parameters. Defaults to None.
        scaffold_levels (list, optional): A list of scaffold levels. Defaults to None.
        **kwargs: Additional keyword arguments.

    Raises:
        AssertionError: If layer_list is not provided.
    """

    def __init__(self,
                layer_list: list = None,
                ffnet_type: str = 'normal',
                #layer_types: list = None,
                xstim_n: str = 'stim',
                ffnet_n: list = None,
                input_dims_list: list = None,
                reg_list: list = None,
                scaffold_levels: list = None,
                stim_dims = None,
                **kwargs,
                ):
        # if len(kwargs) > 0:
        #     print("FFnet: unknown kwargs:", kwargs)
        #print('inside', input_dims_list)
        if ffnet_type == 'mult' and layer_list is None:
            #layer_list = []
            if xstim_n is not None:
                assert stim_dims is not None, "FFnetwork: mult ffnet with xstim_n must have stim_dims specified"
                if layer_list is not None:
                    self.input_dims = layer_list[0]['input_dims']
                else:
                    self.input_dims = stim_dims
                    layer_list = []
        else:
            assert layer_list is not None, "FFnetwork: Must supply a layer_list."
            
        super().__init__()

        self.LayerTypes = {
            'normal': layers.NDNLayer,
            'conv': layers.ConvLayer,
            'divnorm': layers.DivNormLayer,
            'tconv': layers.TconvLayer,
            'stconv': layers.STconvLayer,
            'tlayer': layers.Tlayer,
            'biconv': layers.BiConvLayer1D,
            'bishift': layers.BinocShiftLayer,
            'bistconv': layers.BiSTconv1D,
            #'channelconv': layers.ChannelConvLayer,
            'ori': layers.OriLayer,
            'oriconv': layers.OriConvLayer,
            'oriconvH': layers.HermiteOriConvLayer,
            'conv3d': layers.ConvLayer3D,
            'oolayer': layers.OnOffLayer,
            'masklayer': layers.MaskLayer,
            'maskconv': layers.MaskConvLayer,
            'maskSTClayer': layers.MaskSTconvLayer,
            'maskTlayer': layers.MaskTlayer,
            'iterT': layers.IterTlayer,
            'iterST': layers.IterSTlayer,
            'readout': layers.ReadoutLayer,
            'readout3d': layers.ReadoutLayer3d,
            'readoutQ': layers.ReadoutLayerQsample,
            'fixation': layers.FixationLayer,
            'lag': layers.LagLayer,
            'time': layers.TimeLayer,
            'dim0': layers.Dim0Layer,
            'dimSP': layers.DimSPLayer,
            'dimSPT': layers.DimSPTLayer,
            'channel': layers.ChannelLayer,
            'LVlayer': layers.LVLayer,
            'l1layer': layers.L1convLayer,
            'timeshift': layers.TimeShiftLayer,
            'partial': layers.NDNLayerPartial,
            'partial_conv': layers.ConvLayerPartial,
            'partial_oriconv': layers.OriConvLayerPartial,
            'ptunlayer': layers.ParametricTuneLayer
            #'PyrLayer': layers.PyrLayer,
            #'ConvPyrLayer': layers.ConvPyrLayer
            # 'external': layers.ExternalLayer,    
            #'res': layers.ResLayer,
        }

        self.network_type = ffnet_type
        #print("FFnet: network type:", self.network_type)
        assert self.network_type in _valid_ffnet_types, "ffnet_type " + self.network_type + " is unknown."

        # Format and record inputs into ffnet
        self.layer_list = deepcopy(layer_list)
        self.layer_types = []   # read from layer_list (if necessary at all)
        self.xstim_n = xstim_n
        self.ffnets_in = ffnet_n
        self.shifter = False
        num_layers = len(self.layer_list)

        if num_layers == 0:
            self.layers = nn.ModuleList()
            self.determine_input_dims( input_dims_list=input_dims_list, ffnet_type=ffnet_type )
            self.output_dims = deepcopy(self.input_dims)
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
            #print(Ltype)
            #print(LayerTypes[Ltype])
            self.layers.append( self.LayerTypes[Ltype](**self.layer_list[ll]) )

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

        Args:
            input_dims_list (list): A list of input dimensions for each layer.
            ffnet_type (str): The type of the feedforward network.

        Returns:
            valid_input_dims (bool): Whether the passed in input dims are valid.
        """

        valid_input_dims = True
        if self.ffnets_in is None:
            # then external input (assume from one source)
            assert len(input_dims_list) == 1, "FFnet constructor: Only one set of input dims can be specified."
            assert input_dims_list[0] is not None, "FFnet constructor: External input dims must be specified."
            self.input_dims = input_dims_list[0]
        else: 
            num_input_networks = len(self.ffnets_in)
            if self.xstim_n is not None:
                num_input_networks += 1
            assert len(input_dims_list) == num_input_networks, 'Internal: misspecification of input_dims for FFnetwork.'
            if self.shifter:
                num_skip = 1
            else: 
                num_skip = 0
            # Go through the input dims of the other ffnetowrks to verify they are valid for the type of network
            for ii in range(num_input_networks-num_skip):
                if ii == 0:
                    num_cat_filters = input_dims_list[0][0]
                else:
                    if input_dims_list[ii][1:] != input_dims_list[0][1:]:
                        if ffnet_type in ['add', 'mult']:
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
        Preprocess inputs to the ffnetwork according to the network type. If there
        is only one batch passed in, it does not matter what network type. But mutiple
        inputs (in a list) either:
            'normal': concatenates
            'add': adds inputs together (not must be same size or broadcastable)
            'mult': multiplies x1*(1+x2)*(1+x3+...) with same size req as 'add' 
        Args:
            inputs (list, torch.Tensor): The input to the network.

        Returns:
            x (torch.Tensor): The preprocessed input.

        Raises:
            ValueError: If no layers are defined.
        """
        # Combine network inputs (if relevant)
        if isinstance(inputs, list):
            if len(inputs) == 1:
                x = inputs[0]
            else:
                x = inputs[0].view([-1]+self.input_dims_list[0])  # this will allow for broadcasting
                nt = inputs[0].shape[0]
                for mm in range(1, len(inputs)):
                    #if self.network_type == 'normal': # concatentate inputs
                    #    x = torch.cat( (x, inputs[mm].view([-1]+self.input_dims_list[mm])), 1 )
                    if self.network_type == 'add': # add inputs
                        x = torch.add( x, inputs[mm].view([-1]+self.input_dims_list[mm]) )
                    elif self.network_type == 'mult': # multiply: (input1) x (1+input2)
                        x = torch.multiply(
                            x, torch.add(inputs[mm].view([-1]+self.input_dims_list[mm]), 1.0).clamp(min=0.0) )
                        # Make sure multiplication is not negative
                    else: # especially 'normal'
                        x = torch.cat( (x, inputs[mm].view([-1]+self.input_dims_list[mm])), 1 )
                x = x.view([nt, -1])
        else:
            x = inputs
        return x

    def forward(self, inputs):
        """
        Forward pass through the network.

        Args:
            inputs (list, torch.Tensor): The input to the network.

        Returns:
            x (torch.Tensor): The output of the network.
        """

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

    def proximal_step(self, learning_rate=1.0):
        for layer in self.layers:
            if hasattr(layer, 'proximal_step'):
                layer.proximal_step(learning_rate)
    # END FFnetwork.proximal_step()

    def proxL1list2(self):
        L1list = []
        for layer in self.layers:
            if hasattr(layer, 'proxL1list'):
                L1list += layer.proxL1list2()
        return L1list
        # END FFnetwork.proxL1list2()

    def proxL1list(self):
        regL1_list = []
        for layer in self.layers:
            if hasattr(layer, 'proxL1list2'):
                regL1_list.append(layer.proxL1list2())
        if len(regL1_list) > 0:
            return np.concatenate(regL1_list)
        else:
            return np.zeros(0, dtype=np.float32)
        # END FFnetwork.proxL1list()


    def need_proximal(self):
        for layer in self.layers:
            if hasattr(layer, 'need_proximal'):
                if layer.need_proximal():
                    return True
        return False
    #END FFnetwork.need_proximal()
 
    def __reg_setup_ffnet(self, reg_params=None):
        """
        Sets up the regularization params for the network.

        Args:
            reg_params (dict): The regularization parameters to use.

        Returns:
            layer_reg_list (list): The regularization parameters for each layer.
        """

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

    def prepare_regularization(self, device=None):
        """
        Makes regularization modules with current requested values.
        This is done immediately before training, because it can change during training and tuning.

        Args:
            device (str, optional): The device to use. Defaults to None.
        """

        for layer in self.layers:
            if hasattr(layer, 'reg'):
                layer.reg.build_reg_modules(device=device)
    # END FFnetwork.prepare_regularization()

    def compute_reg_loss(self):
        """
        Computes the regularization loss by summing reg_loss across layers.

        Args:
            None

        Returns:
            rloss (torch.Tensor): The regularization loss.
        """

        rloss = []
        for layer in self.layers:
            rloss.append(layer.compute_reg_loss())
        return reduce(torch.add, rloss)
    # END FFnetwork.compute_reg_loss()

    def list_parameters(self, layer_target=None):
        """
        Lists the (fittable) parameters of the network, calling through each layer

        Args:
            layer_target (int, optional): The layer to list the parameters for. Defaults to None.

        Returns:
            None

        Raises:
            AssertionError: If the layer target is invalid.
        """

        if layer_target is None:
            layer_target = np.arange(len(self.layers), dtype='int32')
        elif not isinstance(layer_target, list):
            layer_target = [layer_target]
        for nn in layer_target:
            assert nn < len(self.layers), '  Invalid layer %d.'%nn
            print("  Layer %d:"%nn)
            self.layers[nn].list_parameters()

    def set_parameters(self, layer_target=None, name=None, val=None ):
        """
        Sets the parameters as either fittable or not (depending on 'val' for the listed 
        layer, with the default being all layers.

        Args:
            layer_target (int, optional): The layer to set the parameters for. Defaults to None.
            name (str): The name of the parameter: default is all parameters.
            val (bool): Whether or not to fit (True) or not fit (False)

        Returns:
            None

        Raises:
            AssertionError: If the layer target is invalid.
        """
        if layer_target is None:
            layer_target = np.arange(len(self.layers), dtype='int32')
        elif not isinstance(layer_target, list):
            layer_target = [layer_target]
        for nn in layer_target:
            assert nn < len(self.layers), '  Invalid layer %d.'%nn
            self.layers[nn].set_parameters(name=name, val=val)
    # END FFnetwork.set_parameters()

    def set_reg_val(self, reg_type=None, reg_val=None, layer_target=None ):
        """
        Set reg_values for listed layer or for all layers.
        
        Args:
            reg_type (str): The type of regularization to set.
            reg_val (float): The value to set the regularization to.
            layer_target (int, optional): The layer to set the regularization for. Defaults to None.

        Returns:
            None

        Raises:
            AssertionError: If the layer target is invalid.
        """
        if layer_target is None:
            layer_target = 0
        assert layer_target < len(self.layers), "layer target too large (max = %d)"%len(self.layers)
        self.layers[layer_target].set_reg_val( reg_type=reg_type, reg_val=reg_val )
    # # END FFnetwork.set_reg_val

    def plot_filters(self, layer_target=0, **kwargs):
        """
        Plots the filters for the listed layer, passed down to layer's call

        Args:
            layer_target (int, optional): The layer to plot the filters for. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.layers[layer_target].plot_filters(**kwargs)
    # END FFnetwork.plot_filters()

    def get_weights(self, layer_target=0, **kwargs):
        """
        Passed down to layer call, with optional arguments conveyed.
        
        Args:
            layer_target (int): The layer to get the weights for.
            **kwargs: Additional keyword arguments.

        Returns:
            The weights for the specified layer.

        Raises:
            AssertionError: If the layer target is invalid.
        """
        assert layer_target < len(self.layers), "Invalid layer_target %d"%layer_target
        return self.layers[layer_target].get_weights(**kwargs)
    # END FFnetwork.get_weights()

    def get_biases(self, layer_target=0, **kwargs):
        """
        Passed down to layer call, with optional arguments conveyed.
        
        Args:
            layer_target (int): The layer to get the weights for.
            **kwargs: Additional keyword arguments.

        Returns:
            The biases for the specified layer.

        Raises:
            AssertionError: If the layer target is invalid.
        """
        assert layer_target < len(self.layers), "Invalid layer_target %d"%layer_target
        return self.layers[layer_target].get_biases(**kwargs)
    # END FFnetwork.get_biases()

    def get_network_info(self, abbrev=False):
        """
        Prints out a description of the network structure.
        """
        if not abbrev:
            for l, L in enumerate(self.layer_list):
                output = f"  L{l}: {L['layer_type']} ({L['num_filters']}"
                if 'conv' in L['layer_type']:
                    output += f", {L['window']}"
                output += f")  |  {L['output_norm']}  |  {L['NLtype']}"
                print(output)
            #if self.network_type == 'mult' and len(self.layer_list) == 0:
            #    print( "  ")
        else: 
            print('not implemented yet')
    # END FFnetwork.get_network_info()

    def info( self, ffnet_n=None, expand=False ):
        """
        This outputs the network information in abbrev (default) or expanded format, including
        information from layers
        """
        info_string = self.generate_info_string()
        # Print ffnet-info
        if ffnet_n is None:
            print(info_string)
        else:
            print( " %2d %s"%(ffnet_n, info_string) )

        # Layer info
        for ii in range(len(self.layers)):
            a, b, c = self.layers[ii].info(expand=expand, to_output=False)
            filler = '\t'*np.maximum((5-int(np.ceil(len(a)+len(b)-1)/8)), 1)
            print( "   %2d %s %s%s%s"%(ii, a, b, filler, c))
            if expand:
                self.layers[ii].list_parameters()
    # END FFnetwork.info()

    def generate_info_string(self):
        info_string = self.network_type + ': Input = '
        if self.xstim_n is not None:
            info_string += "'" + self.xstim_n + "' "
        if self.ffnets_in is not None:
            if self.xstim_n is not None:
                info_string += '+ '        
            info_string += 'ffnet '
            for ii in range(len(self.ffnets_in)):
                info_string += "%d "%self.ffnets_in[ii]
        info_string += '-> ' + str(self.input_dims)
        return info_string
    # END FFnetwork.generate_info_string()

    @classmethod
    def ffnet_dict( cls, layer_list=None, xstim_n ='stim', ffnet_n=None, ffnet_type='normal', scaffold_levels=None, num_lags_out=1, **kwargs):
        """
        Returns a dictionary of the feedforward network.

        Args:
            layer_list (list): A list of dictionaries representing the layers of the network.
            xstim_n (str): The name of the stimulus input.
            ffnet_n (list): A list of feedforward networks.
            ffnet_type (str): The type of the feedforward network.
            scaffold_levels (list): A list of scaffold levels.
            num_lags_out (int): The number of lags out.
            **kwargs: Additional keyword arguments.

        Returns:
            ffnet_dict (dict): The dictionary of the feedforward network.
        """
        dict_out = {
            'ffnet_type': ffnet_type,
            'xstim_n':xstim_n, 'ffnet_n':ffnet_n,
            'layer_list': deepcopy(layer_list),
            'scaffold_levels': scaffold_levels,
            'num_lags_out': num_lags_out}
        if ffnet_type == 'mult' and xstim_n is not None: # let's passively read from xstim_n
            dict_out['stim_dims'] = kwargs.get('stim_dims', None)
        return dict_out
    # END FFnetwork class


class ScaffoldNetwork(FFnetwork):
    """
    Concatenates output of all layers together in filter dimension, preserving spatial dims.

    This essentially used the constructor for Point1DGaussian, with dicationary input.
    Currently there is no extra code required at the network level. I think the constructor
    can be left off entirely, but leaving in in case want to add something.
    """

    def __repr__(self):
        s = super().__repr__()
        # Add information about module to print out
        s += self.__class__.__name__
        return s

    def __init__(self, scaffold_levels=None, num_lags_out=1, **kwargs):
        """
        Args:
            scaffold_levels (list): A list of scaffold levels.
            num_lags_out (int): The number of lags out.

        Raises:
            AssertionError: If the scaffold levels are invalid.
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
        self.spatial_dims = self.layers[self.scaffold_levels[0]].output_dims[1:3]
        self.filter_count = np.zeros(len(self.scaffold_levels))
        self.filter_count[0] = self.layers[self.scaffold_levels[0]].output_dims[0]
        
        for ii in range(1, len(self.scaffold_levels)):
            assert self.layers[self.scaffold_levels[ii]].output_dims[1:3] == self.spatial_dims, "Spatial dims problem layer %d"%self.scaffold_levels[ii] 
            self.filter_count[ii] = self.layers[self.scaffold_levels[ii]].output_dims[0]

        # Construct output dimensions
        if self.num_lags_out is not None:
            self.output_dims = [int(np.sum(self.filter_count))] + self.spatial_dims + [self.num_lags_out]
        else:
            scaffold_lags = [self.layers[self.scaffold_levels[ii]].output_dims[3] for ii in range(len(self.scaffold_levels))]
            # assert that all scaffold_lags are the same
            assert np.all(np.array(scaffold_lags) == scaffold_lags[0]), "Scaffold: cannot currently handle different lag dimensions"
            filter_x_lag = int(np.sum(self.filter_count)) * scaffold_lags[0]
            self.output_dims = [filter_x_lag] + self.spatial_dims + [1]
    # END ScaffoldNetwork.__init__

    def forward(self, inputs):
        """
        Forward pass through the network: passes input sequentially through layers
        and concatenates the based on the self.scaffold_levels argument. Note that if
        there are lags, it will either chomp to the last, or keep number specified
        by self.num_lags_out

        Args:
            inputs (list, torch.Tensor): The input to the network.

        Returns:
            x (torch.Tensor): The output of the network.

        Raises:
            ValueError: If no layers are defined.
        """

        if self.layers is None:
            raise ValueError("Scaffold: no layers defined.")
        
        out = [] # returned 
        x = self.preprocess_input(inputs)

        for layer in self.layers:
            x = layer(x)
            nt = x.shape[0]
            if self.num_lags_out is None and layer.output_dims[3] > 1:
                # reshape y to combine the filters and lags in the second dimension
                # batch x filters x (width x height) x lags
                y = x.reshape([nt, layer.output_dims[0], -1, layer.output_dims[3]])
                # move the lag dimension after the filters (batch, filter, lag, width x height)
                y = y.permute(0, 1, 3, 2)
                # flatten the filter and lag dimensions to be filters x lags
                y = y.reshape([nt, -1])
                out.append(y)
            elif self.num_lags_out is not None and layer.output_dims[3] > self.num_lags_out:
                # Need to return just first lag (lag0) -- 'chomp'
                y = x.reshape([nt, -1, layer.output_dims[3]])[..., :(self.num_lags_out)]
                out.append( y.reshape((nt, -1) ))
            else:
                out.append(x)
        
        # this concatentates across the filter dimension
        return torch.cat([out[ind] for ind in self.scaffold_levels], dim=1)
    # END ScaffoldNetwork.forward()

    def generate_info_string(self):
        info_string = super().generate_info_string()
        #info_string = self.network_type + ": Input = '%s', Output levels "%self.xstim_n
        info_string += ", Scaffold levels ="
        for ii in range(len(self.scaffold_levels)):
            info_string += " %d"%self.scaffold_levels[ii]
        return info_string
    # END ScaffoldNetwork.generate_info_string()

    @classmethod
    def ffnet_dict( cls, scaffold_levels=None, num_lags_out=1, **kwargs):
        """
        Returns a dictionary of the scaffold network.

        Args:
            scaffold_levels (list): A list of scaffold levels.
            num_lags_out (int): The number of lags out.
            **kwargs: Additional keyword arguments.

        Returns:
            ffnet_dict (dict): The dictionary of the scaffold network.

        Raises:
            AssertionError: If the scaffold levels are invalid.
        """
        ffnet_dict = super().ffnet_dict(**kwargs)
        ffnet_dict['ffnet_type'] = 'scaffold'
        ffnet_dict['scaffold_levels'] = scaffold_levels
        ffnet_dict['num_lags_out'] = num_lags_out
        return ffnet_dict
# END ScaffoldNetwork


class ScaffoldNetwork3D(ScaffoldNetwork):
    """
    Like scaffold network above, but preserves the third dimension so in order
    to have shaped filters designed to process in subsequent network components.

    Args:
        num_lags_out (int): The number of lags out.

    Raises:
        AssertionError: If the scaffold levels are invalid.
    """

    def __repr__(self):
        s = super().__repr__()
        # Add information about module to print out
        s += self.__class__.__name__
        return s

    def __init__(self, layer_list=None, num_lags_out=None, **kwargs):
        assert num_lags_out is not None, "should be using num_lags_out with the scaffold3d network"

        # layer_list might have to be modified before passed up the chain
        # but for now, modifying afterwards
        super().__init__(layer_list=layer_list, **kwargs)
        self.network_type = 'scaffold3d'

        self.num_lags_out = num_lags_out  # Makes output equal to number of lags
        self.output_dims[-1] = self.num_lags_out

        # Possibility of broadcasting third dim if those networks don't have it
        #self.broadcast_last_dim = [False]*len(layer_list)
        #for ii in range(len(layer_list)):
        #    if layer_list[ii]['output_dims'][3] == 1:  # could write this boolean-like too
        #        self.broadcast_last_dim[ii] = True
        #        print( "  Scaffold3d: broadcasting layer %d"%ii )
    # END ScaffoldNetwork3d.__init__

    def forward(self, inputs):
        """
        Forward pass through the network.

        Args:
            inputs (list, torch.Tensor): The input to the network.

        Returns:
            x (torch.Tensor): The output of the network.
        
        Raises:
            ValueError: If no layers are defined.
        """

        if self.layers is None:
            raise ValueError("Scaffold: no layers defined.")
        
        out = [] # returned 
        x = self.preprocess_input(inputs)

        for ii in range(len(self.layers)):
        #for layer in self.layers:
            #x = layer(x)
            x = self.layers[ii](x)  # push x through
            
            #if self.broadcast_last_dim[ii]:
            if (self.layers[ii].output_dims[-1] < self.num_lags_out) & (ii in self.scaffold_levels):
                # This has to pad with zeros this output by num_lags_out
                out.append(
                    F.pad( 
                        x[:, :, None], 
                        [0, self.num_lags_out-1] 
                        #[0, x.shape[1]*(self.num_lags_out-1)] 
                        ).reshape([x.shape[0], -1]) ) 
                #print(ii, 'reshaped', out[ii].shape)
            else:
                out.append(x)
        # this concatentates across the filter dimension
        return torch.cat([out[ind] for ind in self.scaffold_levels], dim=1)
    # END ScaffoldNetwork3d.forward()

    @classmethod
    def ffnet_dict( cls, **kwargs):
        """
        Returns a dictionary of the scaffold network.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            ffnet_dict (dict): The dictionary of the scaffold network.
        """
        ffnet_dict = super().ffnet_dict(**kwargs)
        ffnet_dict['ffnet_type'] = 'scaffold3d'
        return ffnet_dict
# END ScaffoldNetwork3d


class ReadoutNetwork(FFnetwork):
    """
    A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
    and covariance of that Gaussian are learned.
    """

    def __repr__(self):
        s = super().__repr__()
        # Add information about module to print out
        s += self.__class__.__name__
        return s

    def __init__(self, shifter=False, **kwargs):
        """
        Same as contructor for regular network, with extra argument to say if there is a shifter coming in. 
        If there is a shifter, it will interpret (in the forward) the last element routing towards the shifter
        """
        super().__init__(**kwargs)
        self.network_type = 'readout'
        self.shifter = shifter
        # Make sure first type is readout: important for interpretation of input dims and potential shifter
        #assert kwargs['layer_list'][0]['layer_type'] == 'readout', "READOUT NET: Incorrect leading layer type"

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

        Args:
            input_dims_list (list): A list of input dimensions for each layer.
            **kwargs: Additional keyword arguments.

        Returns:
            valid_input_dims (bool): Whether the passed in input dims are valid.
        """

        valid_input_dims = True
        if self.ffnets_in is None:
            # then external input (assume from one source)
            valid_input_dims = False
            print('Readout layer cannot get an external input.') 
            self.input_dims = input_dims_list[0]
        else:
            # Check to see if last dimension is shifter
            if np.prod(input_dims_list[-1]) <= 2:
                self.shifter=True
            valid_input_dims = super().determine_input_dims(
                input_dims_list=input_dims_list, ffnet_type='normal')
            assert len(input_dims_list) == len(self.ffnets_in), 'Internal: misspecification of input_dims for FFnetwork.'
        return valid_input_dims
    # END ReadoutNetwork.determine_input_dims

    def forward(self, inputs):
        """
        Network inputs correspond to output of conv layer, and (if it exists), a shifter.
        
        Args:
            inputs (list, torch.Tensor): The input to the network.

        Returns:
            y (torch.Tensor): The output of the network.
        """ 

        if not isinstance(inputs, list):
            inputs = [inputs]

        if self.shifter:
            x = self.preprocess_input(inputs[:-1])
            y = self.layers[0](x, shift=inputs[-1])
        else:
            x = self.preprocess_input(inputs)
            y = self.layers[0](x)
        return y
    # END ReadoutNetwork.forward()

    def get_readout_locations(self):
        """
        Returns the positions in the readout layer (within this network)

        Args:
            None

        Returns:
            The readout locations
        """
        return self.layers[0].get_readout_locations()

    def set_readout_locations(self, locs):
        """
        Sets the readout locations

        Args:
            locs: the readout locations

        Returns:
            None
        """
        self.layers[0].set_readout_locations(locs)
    # END ReadoutNetwork.set_readout_locations()
    
    @classmethod
    def ffnet_dict( cls, ffnet_n=0, shifter=False, **kwargs):
        """
        Returns a dictionary of the readout network.

        Args:
            ffnet_n (int): The feedforward network.

        Returns:
            ffnet_dict (dict): The dictionary of the readout network.
        """
        ffnet_dict = super().ffnet_dict(xstim_n=None, ffnet_n=ffnet_n, **kwargs)
        ffnet_dict['ffnet_type'] = 'readout'
        ffnet_dict['shifter'] = shifter
        return ffnet_dict
    # END ReadoutNetwork


class FFnet_external(FFnetwork):
    """
    This is a 'shell' that lets an external network be plugged into the NDN. It establishes all the basics
    so that information requested to this network from other parts of the NDN will behave correctly.

    Args:
        external_module_dict (dict): A dictionary of external modules.
        external_module_name (str): The name of the external module.
        input_dims_reshape (list): A list of input dimensions to reshape.
        **kwargs: Additional keyword arguments.

    Raises:
        AssertionError: If the external module dictionary is invalid.
    """
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
        """
        Forward pass through the network.

        Args:
            inputs (list, torch.Tensor): The input to the network.

        Returns:
            y (torch.Tensor): The output of the network.

        Raises:
            ValueError: If no layers are defined.
        """
        
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
        """
        Computes the regularization loss.

        Args:
            None

        Returns:
            0
        """
        # Since we do not implement regularization within the external network, this returns nothing
        return 0

    def list_params(self, layer_target=None):
        """
        Lists the parameters for the network.

        Args:
            layer_target (int, optional): The layer to list the parameters for. Defaults to None.

        Returns:
            None

        Raises:
            AssertionError: If the layer target is invalid.
        """
        assert layer_target is None, 'No ability to directly distinguish layers in the external network.'
        for nm, pp in self.named_parameters(recurse=True):
            if pp.requires_grad:
                print("    %s:"%nm, pp.size())
            else:
                print("    NOT FIT: %s:"%nm, pp.size())

    def set_params(self, layer_target=None, name=None, val=None ):
        """
        Sets the parameters for the listed layer or for all layers.

        Args:
            layer_target (int, optional): The layer to set the parameters for. Defaults to None.
            name (str): The name of the parameter.
            val (bool): The value to set the parameter to.

        Returns:
            None

        Raises:
            AssertionError: If the layer target is invalid.
        """
        assert layer_target is None, 'No ability to directly distinguish layers in the external network.'
        assert isinstance(val, bool), 'val must be set.'
        for nm, pp in self.named_parameters(recurse=True):
            if name is None:
                pp.requires_grad = val
            elif nm == name:
                pp.requires_grad = val

    @classmethod
    def ffnet_dict( cls, **kwargs):
        """
        Returns a dictionary of the external network.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            ffnet_dict (dict): The dictionary of the external network.
        """
        ffnet_dict = super().ffnet_dict(**kwargs)
        ffnet_dict['ffnet_type'] = 'external'
        return ffnet_dict


class ScaffoldNetwork3d(ScaffoldNetwork3D):
    """Placeholder so old models are not lonely"""
    def __init__(self, layer_list=None, num_lags_out=None, **kwargs):
        assert num_lags_out is not None, "should be using num_lags_out with the scaffold3d network"
        super().__init__(layer_list=layer_list, **kwargs)
