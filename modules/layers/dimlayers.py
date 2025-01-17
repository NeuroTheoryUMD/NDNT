import torch
import torch.nn as nn
from .ndnlayer import NDNLayer
import numpy as np

class Dim0Layer(NDNLayer):
    """
    Filters that act solely on filter-dimension (dim-0).
    """ 

    def __init__(self,
            input_dims=None,
            num_filters=None,
            **kwargs,
        ):
        """
        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
        """
        assert input_dims is not None, "Dim0Layer: Must specify input_dims"
        assert input_dims[0] > 1, "Dim0Layer: Dim-0 of input must be non-trivial"
        assert num_filters is not None, "Dim0Layer: num_filters must be specified"
        
        # Put filter weights in time-lag dimension to allow regularization using d2t etc
        filter_dims = [1, 1, 1, input_dims[0]]

        super().__init__(input_dims,
            num_filters,
            filter_dims=filter_dims,
            bias=False,
            **kwargs)

        self.output_dims = [num_filters] + input_dims[1:]
        self.num_outputs = np.prod(self.output_dims)

        self.num_other_dims = input_dims[1]*input_dims[2]*input_dims[3]
    # END Dim0Layer.__init__

    def forward(self, x):
        """
        Args:
            x: torch.Tensor, input tensor of shape [B, C, H, W, L]

        Returns:
            y: torch.Tensor, output tensor of shape [B, N, H, W, L]
        """
        w = self.preprocess_weights()

        # Reshape input to expose dim0 and put last
        x = x.reshape( [-1, self.input_dims[0], self.num_other_dims] ).permute([0, 2, 1])
        # Simple linear processing of dim0
        x = torch.matmul(x, w).permute([0, 2, 1]).reshape([-1, self.num_outputs])

        # Einsum version
        #x = x.view( [-1, self.input_dims[0], self.num_other_dims] )
        #x = torch.einsum('tcx,cn->tnx', x, w)

        # left over things from NDNLayer forward
        if self.norm_type == 2:
            x = x / self.weight_scale

        #x = x + self.bias

        # Nonlinearity
        if self.NL is not None:
            x = self.NL(x)

        # Constrain output to be signed
        if self._ei_mask is not None:
            x = x * self._ei_mask

        return x 
    # END Dim0Layer.forward
    
    @staticmethod
    def dim_info( input_dims=None, num_filters=None, **kwargs):
        """
        This uses the methods in the init to determine the input_dims, output_dims, filter_dims, and actual size of
        the weight tensor (weight_shape), given inputs, and package in a dictionary. This should be overloaded with each
        child of NDNLayer if want to use -- but this is external to function of actual layer.

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters

        Returns:
            dinfo: dictionary with keys 'input_dims', 'filter_dims', 'output_dims', 'num_outputs', 'weight_shape'
        """
        assert input_dims is not None, "NDNLayer: Must include input_dims."
        assert num_filters is not None, "NDNLayer: Must include num_filters."

        filter_dims = tuple([input_dims[0], 1, 1, 1])
        output_dims = tuple([num_filters] + input_dims[1:])
        num_outputs = np.prod(output_dims) 
        weight_shape = tuple([np.prod(filter_dims), num_filters])  # likewise
        
        dinfo = {
            'input_dims': tuple(input_dims), 'filter_dims': tuple(filter_dims), 
            'output_dims': output_dims, 'num_outputs': num_outputs,
            'weight_shape': weight_shape}

        return dinfo
    # END [static] Dim0Layer.dim_info

    def _layer_abbrev( self ):
        return "dim0lay"

    @classmethod
    def layer_dict(cls, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """
        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'dim0'
        del Ldict['bias']
        return Ldict
    # END [classmethod] Dim0Layer.layer_dict


class ChannelLayer(NDNLayer):
    """
    Applies individual filter for each filter (dim0) dimension, preserving separate channels but filtering over other dimensions
    => num_filters equals the channel dimension by definition, otherwise like NDNLayer
    """ 

    def __init__(self,
            input_dims=None,
            num_filters=None,
            **kwargs,
        ):
        """
        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
        """
        assert input_dims is not None, "ChannelLayer: Must specify input_dims"
        assert input_dims[0] > 1, "ChannelLayer: Dim-0 of input must be non-trivial"
        if num_filters is None:
            num_filters = input_dims[0]
            num_input_filters = 1
        else:
            assert input_dims[0]%num_filters == 0, "ChannelLayer: num_filters must be multiple of number of filter inputs"
            num_input_filters = input_dims[0] // num_filters
 
        filter_dims = [num_input_filters] + input_dims[1:]
        super().__init__(input_dims, num_filters, filter_dims=filter_dims, **kwargs)
    # END ChannelLayer.__init__

    def _layer_abbrev( self ):
        return "channel"

    @classmethod
    def layer_dict(cls, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        #del Ldict['num_filters']
        Ldict['layer_type'] = 'channel'
        return Ldict
    # END [classmethod] ChannelLayer.layer_dict

    def forward(self, x):
        """
        Args:
            x: torch.Tensor, input tensor of shape [B, C, H, W, L]

        Returns:
            y: torch.Tensor, output tensor of shape [B, N, H, W, L]
        """
        # Pull weights and process given pos_constrain and normalization conditions
        w = self.preprocess_weights()

        # Extract filter_dimension 
        x = torch.reshape( x, [-1, self.num_filters, np.prod(self.filter_dims)])

        # matrix multiplication of non-filter dimensions
        y = torch.einsum('anb, bn -> an', x, w)
        if self.norm_type == 2:
            y = y / self.weight_scale

        y = y + self.bias

        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)

        # Constrain output to be signed
        if self._ei_mask is not None:
            y = y * self._ei_mask

        return y 
    # END ChannelLayer.forward


class DimSPLayer(NDNLayer):
    """
    Filters that act solely on spatial-dimensions (dims-1,3)
    transparent=True means that one spatial filter for each channel
    """ 

    def __init__(self,
            input_dims=None,
            num_filters=None,
            transparent=False,
            **kwargs,
        ):
        """
        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            transparent: if True, one spatial filter for each channel
        """
        assert input_dims is not None, "DimSPLayer: Must specify input_dims"
        assert np.prod(input_dims[1:3]) > 1, "DimSPLayer: Spatial dims of input must be non-trivial"
        assert num_filters is not None, "DimSPLayer: num_filters must be specified"
        if transparent:
            assert input_dims[0]==num_filters, "DimSPLayer: transparency requires correct # of filters"

        # Put filter weights in time-lag dimension to allow regularization using d2t etc
        filter_dims = [1, input_dims[1], input_dims[2], 1]
        assert input_dims[3] == 1, "DimSPLayer: Currently does not work with lags"

        super().__init__(input_dims,
            num_filters,
            filter_dims=filter_dims,
            bias=False,
            **kwargs)

        self.transparent = transparent
        if transparent:
            self.output_dims = [num_filters, 1, 1, input_dims[3]]
        else:
            self.output_dims = [num_filters*input_dims[0], 1, 1, input_dims[3]]

        self.num_outputs = np.prod(self.output_dims)
        self.num_sp_dims = input_dims[1]*input_dims[2]
    # END DimSPLayer.__init__

    def forward(self, x):
        """
        Args:
            x: torch.Tensor, input tensor of shape [B, C, H, W, L]

        Returns:
            y: torch.Tensor, output tensor of shape [B, N, H, W, L]
        """
        w = self.preprocess_weights()

        # Reshape input to expose spatial dims -- note a lag-dim will screw this up without a permute
        if self.input_dims[3] == 1:
            x = x.view( [-1, self.input_dims[0], self.num_sp_dims] )
        else:
            print("NEED TO RESHAPE, PERMUTE, and RESHAPE")

        if self.transparent:
            # One spatial filter for each channel
            x = torch.einsum('tcx,xc->tc', x, w)
            #x = torch.sum(torch.multiply(x, w.T), axis=2)
            #x = torch.diagonal(torch.matmul(x, w), dim1=1, dim2=2)
        else:
            # Simple linear processing of dim0
            x = torch.matmul(x, w).reshape([-1, self.num_outputs])

        # left over things from NDNLayer forward
        if self.norm_type == 2:
            x = x / self.weight_scale

        #x = x + self.bias

        # Nonlinearity
        if self.NL is not None:
            x = self.NL(x)

        # Constrain output to be signed
        if self._ei_mask is not None:
            x = x * self._ei_mask

        return x 
    # END DimSPLayer.forward

    @classmethod
    def layer_dict(cls, transparent=False, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'dimSP'
        Ldict['transparent'] = transparent
        del Ldict['bias']
        return Ldict
    # END [classmethod] DimSPLayer.layer_dict

class DimSPTLayer(NDNLayer):
    """
    Filters that act solely on spatial-dimensions (dims-1,3).
    """ 

    def __init__(self,
            input_dims=None,
            num_filters=None,
            transparent=False,
            **kwargs,
        ):
        """
        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            transparent: if True, one spatial filter for each channel
        """
        assert input_dims is not None, "DimSPTLayer: Must specify input_dims"
        assert np.prod(input_dims[1:]) > 1, "DimSPTLayer: Spatial dims of input must be non-trivial"
        assert num_filters is not None, "DimSPTLayer: num_filters must be specified"
        
        # Put filter weights in time-lag dimension to allow regularization using d2t etc
        filter_dims = [1, input_dims[1], input_dims[2], input_dims[3]]

        super().__init__(input_dims,
            num_filters,
            filter_dims=filter_dims,
            bias=False,
            **kwargs)

        self.output_dims = [num_filters*input_dims[0], 1, 1, 1]
        self.num_outputs = np.prod(self.output_dims)
        self.transparent = transparent

        self.num_spt_dims = input_dims[1]*input_dims[2]*input_dims[3]
    # END DimSPLayer.__init__

    def forward(self, x):
        """
        Args:
            x: torch.Tensor, input tensor of shape [B, C, H, W, L]

        Returns:
            y: torch.Tensor, output tensor of shape [B, N, H, W, L]
        """
        w = self.preprocess_weights()

        # Reshape input to expose spatial dims 
        x = x.reshape( [-1, self.input_dims[0], self.num_spt_dims] )

        # Simple linear processing of dim0
        x = torch.matmul(x, w).reshape([-1, self.num_outputs])

        # left over things from NDNLayer forward
        if self.norm_type == 2:
            x = x / self.weight_scale

        #x = x + self.bias

        # Nonlinearity
        if self.NL is not None:
            x = self.NL(x)

        # Constrain output to be signed
        if self._ei_mask is not None:
            x = x * self._ei_mask

        return x 
    # END DimSPTLayer.forward

    @classmethod
    def layer_dict(cls, transparent=False, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'dimSPT'
        Ldict['transparent'] = transparent
        del Ldict['bias']
        return Ldict
    # END [classmethod] DimSPTLayer.layer_dict