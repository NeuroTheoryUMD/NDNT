import torch
import torch.nn as nn
from .convlayers import ConvLayer
from torch.nn import functional as F
import numpy as np


class BiConvLayer1D(ConvLayer):
    """
    Filters that act solely on filter-dimension (dim-0)

    """ 

    def __init__(self, **kwargs ):
        """Same arguments as ConvLayer, but will reshape output to divide space in half"""

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