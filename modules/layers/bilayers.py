import torch
import torch.nn as nn
from .convlayers import ConvLayer, STconvLayer
from .ndnlayer import NDNLayer
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

class BinocLayer1D(ConvLayer):
    """
    Takes a monocular convolutional output over both eyes -- assumes first spatial dimension is doubled
    -- reinterprets input as separate filters from each eye, but keeps them grouped
    -- assumes each 2 input filters (for each eye -- 4 inputs total) are inputs to each output filter
    """ 

    def __init__(self, input_dims=None, num_filters=None, padding=None, **kwargs ):
        """
        Same arguments as ConvLayer, but will reshape output to divide space in half
        
        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            padding: 'same' or 'valid' (default 'same')
        """

        assert padding is None, "Should not enter padding: will be set to 'same'"
        NF, NX2, a, b = input_dims
        assert (a == 1) & (b == 1), "BINOCLAYER: Input dims are screwed up"

        if num_filters is None:
            num_filters = NF//2
        assert num_filters == NF//2, "num_filters must be specified correctly"
        input_dims[0] = 4  # this will be taken care of with groups
        NX = NX2//2
        input_dims[1] = NX

        super().__init__(
            input_dims=input_dims, num_filters=num_filters, padding='same',
            **kwargs)

        self.input_dims[0] = 2*NF  # fix now that weights are specified

        #self.group_filters = group_filters  # I think this is grouped this
    # END BinocLayer1D.__init__

    def forward(self, x):
        """
        Call conv forward, but then option to reshape

        Args:
            x: tensor of shape [batch, num_channels, height, width, lags]

        Returns:
            y: tensor of shape [batch, num_outputs]
        """
        # Reshape weight matrix and inputs (note uses 4-d rep of tensor before combinine dims 0,3)

        s = x.reshape([-1]+self.input_dims).permute(0,1,4,2,3) # B, C, T, X, Y

        w = self.preprocess_weights().reshape(self.filter_dims+[self.num_filters]).permute(4,0,3,1,2)
        # puts output_dims first, as required for conv

        # Collapse over irrelevant dims for dim-specific convs
        #s = torch.reshape( s, (-1, self.folded_dims, self.input_dims[1]) )
        s = torch.reshape( s, (-1, self.input_dims[0], self.input_dims[1]) )
        
        #if self.padding:
        #s = F.pad(s, self._npads, "constant", 0)

        if self._fullpadding:
            #s = F.pad(s, self._npads, "constant", 0)
            y = F.conv1d(
                F.pad(s, self._npads, "constant", 0), 
                w.view([-1, self.folded_dims, self.filter_dims[1]]), 
                bias=self.bias, groups=4,
                stride=self.stride, dilation=self.dilation)
        else:
            y = F.conv1d(
                s,
                w.view([-1, self.folded_dims, self.filter_dims[1]]), 
                bias=self.bias, groups=4,
                padding=self._npads[0],
                stride=self.stride, dilation=self.dilation)

        if not self.res_layer:
            if self.output_norm is not None:
                y = self.output_norm(y)

        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)

        if self._ei_mask is not None:
            if self.is1D:
                y = y * self._ei_mask[None, :, None]
            else:
                y = y * self._ei_mask[None, :, None, None]
            # y = torch.einsum('bchw, c -> bchw* self.ei_mask
            # w = torch.einsum('nctw,tz->nczw', w, self.tent_basis)

        if self.res_layer:
            # s is with dimensions: B, C, T, X, Y 
            y = y + torch.reshape( s, (-1, self.folded_dims, self.input_dims[1]) )
                 
            if self.output_norm is not None:
                y = self.output_norm(y)

        y = torch.reshape(y, (-1, self.num_outputs))

        # store activity regularization to add to loss later
        #self.activity_regularization = self.activity_reg.regularize(y)
        if hasattr(self.reg, 'activity_regmodule'):  # to put buffer in case old model
            self.reg.compute_activity_regularization(y)
        
        return y
    # END BinocLayer1D.forward


    @classmethod
    def layer_dict(cls, **kwargs):
        Ldict = super().layer_dict(**kwargs)
        del Ldict['padding']  # will have 'same' padding automatically
        del Ldict['stride']  # will have 'same' padding automatically
        del Ldict['dilation']  # will have 'same' padding automatically
        # Added arguments
        Ldict['layer_type'] = 'binoc'
        return Ldict
    # END [classmethod] BinocLayer1D.layer_dict


class BiConvLayer1D(ConvLayer):
    """
    Filters that act solely on filter-dimension (dim-0)
    """ 

    def __init__(self, **kwargs ):
        """
        Same arguments as ConvLayer, but will reshape output to divide space in half.

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            filter_dims: width of convolutional kernel (int or list of ints)
            padding: 'same' or 'valid' (default 'same')
            weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            bias: bool, whether to include bias term
            NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
        """

        super().__init__(**kwargs)

        self.output_dims[0] = self.output_dims[0]*2
        self.output_dims[1] = self.output_dims[1]//2
        #self.group_filters = group_filters  # I think this is grouped this
    # END BinConvLayer1D.__init__

    #def forward(self, x):
    #    # Call conv forward, but then option to reshape
    #    super.forward( x )
    #    if self.group_filters:
    #        # Output will be batch x 2 x num_filters x space (with the two corresponding to left and right eyes)


    @classmethod
    def layer_dict(cls, **kwargs):
        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'biconv'
        return Ldict
    # END [classmethod] BinConvLayer1D.layer_dict

class BiSTconv1D(NDNLayer):
#class BiSTconv1D(STconvLayer):
    """
    To be BinocMixLayer (bimix):
    Inputs of size B x C x 2xNX x 1: mixes 2 eyes based on ratio
        filter number is NC -- one for each channel: so infered from input_dims
        filter_dims is [1,1,1,1] -> one number per filter
    ORIGINAL BiSTconv1D: Filters that act solely on filter-dimension (dim-0)
    """ 

    def __init__(self, input_dims=None, num_filters=None, filter_dims=None, norm_type=None,
                 NLtype='relu', **kwargs ):
        """
        Same arguments as ConvLayer, but will reshape output to divide space in half.
        
        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            filter_dims: width of convolutional kernel (int or list of ints)
            padding: 'same' or 'valid' (default 'same')
            weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            bias: bool, whether to include bias term
            NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
        """

        assert input_dims is not None, "BIMIX: Input_dims necessary"
        assert filter_dims is None, "BIMIX: filter_dims invalid"
        assert num_filters is None, "BIMIX: num_filters should not be passed in: preset by input_dims"
        assert norm_type is False, "BIMIX: No normalization allowed"

        super().__init__(
            input_dims=input_dims, num_filters=input_dims[0], filter_dims=[1,1,1,1], 
            **kwargs)

        self.output_dims = deepcopy(input_dims)
        self.output_dims[1] = self.input_dims[0]//2
        self.num_outputs = np.prod(self.output_dims)

# ORIGINAL WAS UNUSED SO USING THE NAME FOR NOW -- to change name if works
#    def __init__(self, **kwargs ):
#        """Same arguments as ConvLayer, but will reshape output to divide space in half"""
#
#        super().__init__(**kwargs)
#
#        self.output_dims[0] = self.output_dims[0]*2
#        self.output_dims[1] = self.output_dims[1]//2
#        #self.group_filters = group_filters  # I think this is grouped this
    # END BinConvLayer1D.__init__

    def forward(self, x):
        """
        Call conv forward, but then option to reshape.

        Args:
            x: tensor of shape [batch, num_channels, height, width, lags]

        Returns:
            y: tensor of shape [batch, num_outputs]
        """
        x = x.reshape([-1, self.input_dims[0], 2, self.output_dims[1]])
        bw = F.sigmoid(self.weight)

        y = bw * x[:,:,0,:] + (1-bw)*x[:,:,1,:] + self.bias[:,None,None]

        if self.NL is not None:
            y = self.NL(y)

        if self._ei_mask is not None:
            y = y * self._ei_mask

        if hasattr(self.reg, 'activity_regmodule'):  # to put buffer in case old model
            self.reg.compute_activity_regularization(y)

        return y.reshape([-1, self.num_outputs])
    #    # Call conv forward, but then option to reshape
    #    super.forward( x )
    #    if self.group_filters:
    #        # Output will be batch x 2 x num_filters x space (with the two corresponding to left and right eyes)

    @classmethod
    def layer_dict(cls, **kwargs):
        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'bistconv'
        return Ldict
    # END [classmethod] BinSTconvLayer1D.layer_dict


class ChannelConvLayer(ConvLayer):
    """
    Channel-Convolutional NDN Layer -- convolutional layer that has each output filter use different
    group of M input filters. So, if there are N output filters, the channel dimension of the input has to be M*N
    
    Args (required):
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_dims: width of convolutional kernel (int or list of ints)

    Args (optional):
        padding: 'same' or 'valid' (default 'same')
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'

    """
    def __init__(self,
            input_dims=None,
            num_filters=None,
            #conv_dims=None,
            #filter_dims=None,
            filter_width=None,
            temporal_tent_spacing=None,
            output_norm=None,
            #stride=None,
            #dilation=1,
            #padding='same',
            #res_layer=False,  # to make a residual layer
            window=None,
            **kwargs,
            ):
        """
        Same arguments as ConvLayer, but will reshape output to divide space in half.

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            filter_width: width of convolutional kernel (int or list of ints)
            temporal_tent_spacing: spacing of tent-basis functions in time
            output_norm: normalization to apply to output
            window: window function to apply to output
        """

        assert input_dims is not None, "ChannelConvLayer: Must specify input_dims"
        assert num_filters is not None, "ChannelConvLayer: Must specify num_filters"
        assert filter_width is not None, "ChannelConvLayer: Must specify filter_width"


        #assert (conv_dims is not None) or (filter_dims is not None), "ConvLayer: conv_dims or filter_dims must be specified"
        if 'conv_dims' in kwargs:
            print("No longer using conv_dims. Use filter_dims instead.")
        
        is1D = input_dims[2] == 1
        # Place filter_dims information in correct place -- maybe overkill, but easy to specify 
        # spatial width in any number of ways using 'filter_dims'

        # Compute filter dims
        assert input_dims[0]%num_filters == 0, "Must have number of filters divide channnel dimensions"
        group_chan = input_dims[0]//num_filters
        filter_dims = [group_chan, filter_width, filter_width, input_dims[-1]]
        if is1D:
            filter_dims[2] = 1

        super().__init__(
            input_dims=input_dims,
            num_filters=num_filters,
            filter_dims=filter_dims,
            temporal_tent_spacing=temporal_tent_spacing,
            output_norm=output_norm,
            window=window,
            **kwargs,  # will pass other arguments through
            )
        
        #self.folded_dims = self.input_dims[0]*self.input_dims[3] <= for stimulus
        self.weights_folded_dims = self.filter_dims[0]*self.filter_dims[3]

    # END ChannelConvLayer.__init__

    @classmethod
    def layer_dict(cls, filter_width=None, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'channelconv'
        Ldict['filter_width'] = filter_width
        del Ldict['filter_dims']
        return Ldict
    # END ChannelConvLayer.layer_dict

    def forward(self, x):
        """
        Call conv forward, but then option to reshape.

        Args:
            x: tensor of shape [batch, num_channels, height, width, lags]

        Returns:
            y: tensor of shape [batch, num_outputs]
        """
        # Reshape weight matrix and inputs (note uses 4-d rep of tensor before combinine dims 0,3)
        s = x.reshape([-1]+self.input_dims).permute(0,1,4,2,3) # B, C, T, X, Y

        w = self.preprocess_weights().reshape(self.filter_dims+[self.num_filters]).permute(4,0,3,1,2)
        # puts output_dims first, as required for conv
        
        # Collapse over irrelevant dims for dim-specific convs
        if self.is1D:
            s = torch.reshape( s, (-1, self.folded_dims, self.input_dims[1]) )
            #if self.padding:
            #s = F.pad(s, self._npads, "constant", 0)

            if self._fullpadding:
                #s = F.pad(s, self._npads, "constant", 0)
                y = F.conv1d(
                    F.pad(s, self._npads, "constant", 0),
                    w.view([-1, self.weights_folded_dims, self.filter_dims[1]]), 
                    groups=self.num_filters,
                    bias=self.bias,
                    stride=self.stride, dilation=self.dilation)
            else:
                y = F.conv1d(
                    s,
                    w.view([-1, self.weights_folded_dims, self.filter_dims[1]]), 
                    groups=self.num_filters,
                    bias=self.bias,
                    padding=self._npads[0],
                    stride=self.stride, dilation=self.dilation)
        else:
            s = torch.reshape( s, (-1, self.folded_dims, self.input_dims[1], self.input_dims[2]) )
            # Alternative location of batch_norm:
            #if self.output_norm is not None:
            #    s = self.output_norm(s)

            if self._fullpadding:
                s = F.pad(s, self._npads, "constant", 0)
                y = F.conv2d(
                    s, # we do our own padding
                    w.view([-1, self.folded_dims, self.filter_dims[1], self.filter_dims[2]]),
                    groups=self.num_filters,
                    bias=self.bias,
                    stride=self.stride,
                    dilation=self.dilation)
            else:
                # functional pads since padding is simple
                y = F.conv2d(
                    s, 
                    w.reshape([-1, self.folded_dims, self.filter_dims[1], self.filter_dims[2]]),
                    groups=self.num_filters,
                    padding=(self._npads[2], self._npads[0]),
                    bias=self.bias,
                    stride=self.stride,
                    dilation=self.dilation)

        if not self.res_layer:
            if self.output_norm is not None:
                y = self.output_norm(y)

        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)

        if self._ei_mask is not None:
            if self.is1D:
                y = y * self._ei_mask[None, :, None]
            else:
                y = y * self._ei_mask[None, :, None, None]
            # y = torch.einsum('bchw, c -> bchw* self.ei_mask
            # w = torch.einsum('nctw,tz->nczw', w, self.tent_basis)

        if self.res_layer:
            # s is with dimensions: B, C, T, X, Y 
            if self.is1D:
                y = y + torch.reshape( s, (-1, self.folded_dims, self.input_dims[1]) )
            else:
                y = y + torch.reshape( s, (-1, self.folded_dims, self.input_dims[1], self.input_dims[2]) )
                 
            if self.output_norm is not None:
                y = self.output_norm(y)

        y = torch.reshape(y, (-1, self.num_outputs))

        # store activity regularization to add to loss later
        #self.activity_regularization = self.activity_reg.regularize(y)
        if hasattr(self.reg, 'activity_regmodule'):  # to put buffer in case old model
            self.reg.compute_activity_regularization(y)
        
        return y
    # END ConvLayer.forward