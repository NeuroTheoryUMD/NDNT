import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

from .ndnlayer import NDNLayer
from .convlayers import ConvLayer
from torch.nn.parameter import Parameter

class Tlayer(NDNLayer):
    """
    NDN Layer where num_lags is handled convolutionally (but all else is normal)

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
    def __init__(
            self,
            input_dims=None,
            num_filters=None,
            num_lags=None,
            temporal_tent_spacing=None,
            output_norm=None,
            res_layer=False,  # to make a residual layer
            **kwargs):
        """
        Tlayer: NDN Layer where num_lags is handled convolutionally (but all else is normal).

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            num_lags: number of lags in spatiotemporal filter
            temporal_tent_spacing: int, spacing of tent basis functions
            output_norm: str, 'batch', 'batchX', or None
            res_layer: bool, whether to make a residual layer
            **kwargs: additional arguments to pass to NDNLayer
        """
        assert input_dims is not None, "Tlayer: Must specify input_dims"
        assert num_filters is not None, "Tlayer: Must specify num_filters"
        assert num_lags is not None, "Tlayer: Must specify num_lags -- otherwise just use NDNLayer"
        assert input_dims[3] == 1, "Tlayer: input dims must not have lags"

        self.tent_basis = None
        if temporal_tent_spacing is not None and temporal_tent_spacing > 1:
            from NDNT.utils import tent_basis_generate
            num_lags = filter_dims[-1] #conv_dims[2]
            tentctrs = list(np.arange(0, num_lags, temporal_tent_spacing))
            self.tent_basis = tent_basis_generate(tentctrs)
            if self.tent_basis.shape[0] != num_lags:
                print('Warning: tent_basis.shape[0] != num_lags')
                print('tent_basis.shape = ', self.tent_basis.shape)
                print('num_lags = ', num_lags)
                print('Adding zeros or truncating to match')
                if self.tent_basis.shape[0] > num_lags:
                    print('Truncating')
                    self.tent_basis = self.tent_basis[:num_lags,:]
                else:
                    print('Adding zeros')
                    self.tent_basis = np.concatenate(
                        [self.tent_basis, np.zeros((num_lags-self.tent_basis.shape[0], self.tent_basis.shape[1]))],
                        axis=0)
                
            self.tent_basis = self.tent_basis[:num_lags,:]
            num_lag_params = self.tent_basis.shape[1]
            print('ConvLayer temporal tent spacing: num_lag_params =', num_lag_params)
            #conv_dims[2] = num_lag_params
            filter_dims[-1] = num_lag_params

        filter_dims = input_dims[:3] + [num_lags]
        super().__init__(
            input_dims=input_dims,
            num_filters=num_filters,
            filter_dims=filter_dims,
            **kwargs)

        self.res_layer = res_layer

        if self.tent_basis is not None:
            self.register_buffer('tent_basis', torch.Tensor(self.tent_basis.T))
            filter_dims[-1] = self.tent_basis.shape[0]
        else:
            self.tent_basis = None

        self.folded_dims = np.prod(self.input_dims[:3])

        # check if output normalization is specified
        if output_norm in ['batch', 'batchX']:
            if output_norm == 'batchX':
                affine = False
            else:
                affine = True
            if self.is1D:
                self.output_norm = nn.BatchNorm1d(self.num_filters, affine=affine)
            else:
                self.output_norm = nn.BatchNorm2d(self.num_filters, affine=affine)
                #self.output_norm = nn.BatchNorm2d(self.folded_dims, affine=False)
        else:
            self.output_norm = None
    # END Tlayer.__init__()

    def forward(self, x):
        """
        Forward pass through the Tlayer.

        Args:
            x: torch.Tensor, input tensor

        Returns:
            y: torch.Tensor, output tensor
        """
        # Reshape stim matrix LACKING temporal dimension [bcwh] 
        s = (x.T)[None, :, :] # [B,dims]->[1,dims,B]

        w = self.preprocess_weights()
        w = w.reshape([self.folded_dims, self.filter_dims[3], -1]).permute(2,0,1) # [C,T,N]->[N,C,T]

        # pad the batch dimension
        pad = (self.filter_dims[-1]-1, 0)
        s = F.pad(s, pad, "constant", 0)

        y = F.conv1d(
            s,
            w, 
            bias=self.bias,
            stride=1, dilation=1)
        
        y = y.permute(2,1,0)[:, :, 0] # [1,N,B] -> [B,N,1] -> [B, N]
    
        if self.output_norm is not None:
            y = self.output_norm(y)

        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)
        
        if self._ei_mask is not None:
            y = y * self._ei_mask[None,:]
        
        if self.res_layer:
            y = y + x
        #y = y.reshape((-1, self.num_outputs))

        # store activity regularization to add to loss later
        #self.activity_regularization = self.activity_reg.regularize(y)
        if hasattr(self.reg, 'activity_regmodule'):  # to put buffer in case old model
            self.reg.compute_activity_regularization(y)

        return y
    # END Tlayer.forward 

    def plot_filters( self, cmaps='gray', num_cols=8, row_height=2, time_reverse=False):
        """
        Overload plot_filters to automatically time_reverse.

        Args:
            cmaps: str or list of str, colormap(s) to use
            num_cols: int, number of columns to use in plot
            row_height: int, height of each row in plot
            time_reverse: bool, whether to reverse the time dimension

        Returns:
            None
        """
        super().plot_filters( 
            cmaps=cmaps, num_cols=num_cols, row_height=row_height, 
            time_reverse=time_reverse)

    @classmethod
    def layer_dict(cls, num_lags=None, res_layer=False, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'tlayer'
        Ldict['num_lags'] = num_lags
        Ldict['res_layer'] = res_layer
        Ldict['temporal_tent_spacing'] = 1
        Ldict['output_norm'] = None
    
        return Ldict


class L1convLayer(NDNLayer):
    """
    First start with non-convolutional version.

    L1convLayer: Convolutional layer with L1 regularization.
    """
    def __init__(self, **kwargs):  # same as ConvLayer, with some extras built in...
        """
        Set up ConvLayer with L1 regularization.

        Args:
            **kwargs: additional arguments to pass to ConvLayer

        Returns:
            None
        """
        super().__init__(**kwargs)
        # Add second set of weights (corresponding to w-)
        self.weight_minus = Parameter(torch.Tensor(size=self.shape))
        self.weight_minus.data = -deepcopy(self.weight.data)
        self.weight_minus.data[self.weight_minus < 0] = 0.0
        self.weight.data[self.weight < 0] = 0.0
        self.window=False
        self.tent_basis=None
    # END L1convLayer.__init__
        
    def preprocess_weights(self):
        """
        Preprocess weights for L1convLayer.

        Returns:
            w: torch.Tensor, preprocessed weights
        """
        #w = F.relu(self.weight) - F.relu(self.weight_minus)
        w = self.weight**2 - self.weight_minus**2
        # Do all preprocessing for NDNlayer, and then conv-layer below
        #w = super().preprocess_weights()

        if self.window:
            w = w.view(self.filter_dims+[self.num_filters]) # [C, H, W, T, D]
            if self.is1D:
                w = torch.einsum('chwln,h->chwln', w, self.window_function)
            else:
                w = torch.einsum('chwln, hw->chwln', w, self.window_function)
            w = w.reshape(-1, self.num_filters)

        if self.tent_basis is not None:
            wdims = self.tent_basis.shape[0]
            
            w = w.view(self.filter_dims[:3] + [wdims] + [-1]) # [C, H, W, T, D]
            w = torch.einsum('chwtn,tz->chwzn', w, self.tent_basis)
            w = w.reshape(-1, self.num_filters)
        
        return w

    def reset_parameters2(self, weights_initializer=None, bias_initializer=None, param=None) -> None:
        """
        Reset parameters for L1convLayer.

        Args:
            weights_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            bias_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            param: dict, additional parameters to pass to the initializer

        Returns:
            None
        """
        super().reset_parameters(weights_initializer, bias_initializer, param)
        self.weight_minus.data = -deepcopy(self.weight.data)
        self.weight_minus.data[self.weight_minus < 0] = 0.0
        self.weight.data[self.weight < 0] = 0.0


# class L1convLayer(ConvLayer):

#     def __init__(self, **kwargs):  # same as ConvLayer, with some extras built in...
#         # Set up ConvLayer
#         super().__init__(**kwargs)
#         # Add second set of weights (corresponding to w-)
#         self.weight_minus = Parameter(torch.Tensor(size=self.shape))
#     # END L1convLayer.__init__
        
#     def preprocess_weights(self):
#         w = F.relu(self.weight) - F.relu(self.weight_minus)
#         # Do all preprocessing for NDNlayer, and then conv-layer below
#         #w = super().preprocess_weights()

#         if self.window:
#             w = w.view(self.filter_dims+[self.num_filters]) # [C, H, W, T, D]
#             if self.is1D:
#                 w = torch.einsum('chwln,h->chwln', w, self.window_function)
#             else:
#                 w = torch.einsum('chwln, hw->chwln', w, self.window_function)
#             w = w.reshape(-1, self.num_filters)

#         if self.tent_basis is not None:
#             wdims = self.tent_basis.shape[0]
            
#             w = w.view(self.filter_dims[:3] + [wdims] + [-1]) # [C, H, W, T, D]
#             w = torch.einsum('chwtn,tz->chwzn', w, self.tent_basis)
#             w = w.reshape(-1, self.num_filters)
        
        return w

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
        # Added arguments
        Ldict['layer_type'] = 'l1layer'

        return Ldict


class OnOffLayer(Tlayer):
    """
    OnOffLayer: Layer with separate on and off filters.
    """
    
    def __init__(
            self,
            input_dims=None,
            num_filters=None,
            num_lags=None,
            temporal_tent_spacing=None,
            output_norm=None,
            res_layer=False,  # to make a residual layer
            **kwargs):
        """
        Args (required):
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            num_lags: number of lags in spatiotemporal filter
            
        Args (optional):
            temporal_tent_spacing: int, spacing of tent basis functions
            output_norm: str, 'batch', 'batchX', or None
            res_layer: bool, whether to make a residual layer
            **kwargs: additional arguments to pass to NDNLayer
        """
        assert input_dims is not None, "Tlayer: Must specify input_dims"
        assert num_filters is not None, "Tlayer: Must specify num_filters"
        assert num_lags is not None, "Tlayer: Must specify num_lags -- otherwise just use NDNLayer"
        assert input_dims[3] == 1, "Tlayer: input dims must not have lags"

        # Trick Tlayer to make weights with double the channel dimension
        input_dims[0] *= 2

        super().__init__(
            input_dims=input_dims,
            num_filters=num_filters,
            num_lags=num_lags,
            temporal_tent_spacing=temporal_tent_spacing,
            output_norm=output_norm,
            res_layer=res_layer,
            **kwargs)

        # Now change change input-dims back
        self.input_dims[0] = self.input_dims[0]//2
    # END OnOffLayer.__init__

    def plot_filters( self, time_reverse=None, **kwargs):
        """
        Plot the filters for the OnOffLayer.

        Args:
            time_reverse: bool, whether to reverse the time dimension
            **kwargs: additional arguments to pass to the plotting function

        Returns:
            None
        """
        ws = self.get_weights(time_reverse=True)
        for ii in range(2):
            if self.input_dims[2] == 1:
                if self.input_dims[1] == 1:
                    from NDNT.utils import plot_filters_1D
                    plot_filters_1D(ws[ii, ...], **kwargs)
                else:
                    from NDNT.utils import plot_filters_ST1D
                    plot_filters_ST1D(ws[ii, ...], **kwargs)
            else:
                if self.input_dims[0] == 1:
                    from NDNT.utils import plot_filters_ST2D
                    plot_filters_ST2D(ws[ii, ...], **kwargs)
                else:
                    from NDNT.utils import plot_filters_ST3D
                    plot_filters_ST3D(ws[ii, ...], **kwargs)
    # END OnOffLayer.plot_filters()
    
    def forward(self, x):
        """
        Forward pass through the OnOffLayer.

        Args:
            x: torch.Tensor, input tensor

        Returns:
            y: torch.Tensor, output tensor
        """
        # Reshape stim matrix LACKING temporal dimension [bcwh] 
        #x2 = torch.cat( (x, abs(x)), axis=1)

        #### After that (above), it is the same forward as Tlayer (until after the lin-conv)
        s = (x.T)[None, :, :] # [B,dims]->[1,dims,B]

        w = self.preprocess_weights()
        w = w.reshape([2, self.folded_dims//2, self.filter_dims[3], -1]).permute(3,0,1,2) # [C,T,N]->[N,C,T]

        # pad the batch dimension
        pad = (self.filter_dims[-1]-1, 0)
        s = F.pad(s, pad, "constant", 0)

        y = F.conv1d(
            s,
            w[:,0, ...], 
            bias=self.bias,
            stride=1, dilation=1)

        y += F.conv1d(
            abs(s),
            w[:,1, ...], 
            bias=self.bias,
            stride=1, dilation=1)

        y = y.permute(2,1,0)[:, :, 0] # [1,N,B] -> [B,N,1] -> [B, N]
    
        if self.output_norm is not None:
            y = self.output_norm(y)

        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)
        
        if self._ei_mask is not None:
            y = y * self._ei_mask[None,:]
        
        if self.res_layer:
            y = y + x
        #y = y.reshape((-1, self.num_outputs))

        # store activity regularization to add to loss later
        #self.activity_regularization = self.activity_reg.regularize(y)
        if hasattr(self.reg, 'activity_regmodule'):  # to put buffer in case old model
            self.reg.compute_activity_regularization(y)

        return y
    # END OnOffLayer.forward 

    @classmethod
    def layer_dict(cls, num_lags=None, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

        Args:
            num_lags: int, number of lags in spatiotemporal filter
            **kwargs: additional arguments to pass to NDNLayer

        Returns:
            Ldict: dict, dictionary of layer parameters
        """

        Ldict = super().layer_dict(num_lags=num_lags, **kwargs)
        # Added arguments
        Ldict['layer_type'] = 'oolayer'
    
        return Ldict
    # END OnOffLayer.layer_dict()


class MaskLayer(NDNLayer):
    """
    MaskLayer: Layer with a mask applied to the weights.
    """

    def __init__(self, input_dims=None,
                 num_filters=None,
                 #filter_dims=None,  # absorbed by kwargs if necessary
                 mask=None,
                 NLtype:str='lin',
                 norm_type:int=0,
                 pos_constraint=0,
                 num_inh:int=0,
                 bias:bool=False,
                 weights_initializer:str='xavier_uniform',
                 output_norm=None,
                 initialize_center=False,
                 bias_initializer:str='zeros',
                 reg_vals:dict=None,
                 **kwargs,
                 ):
        """
        MaskLayer: Layer with a mask applied to the weights.

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            mask: np.ndarray, mask to apply to the weights
            NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
            norm_type: int, normalization type
            pos_constraint: int, whether to enforce non-negative weights
            num_inh: int, number of inhibitory filters
            bias: bool, whether to include bias term
            weights_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            output_norm: str, 'batch', 'batchX', or None
            initialize_center: bool, whether to initialize the center
            bias_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            reg_vals: dict, regularization values
            **kwargs: additional arguments to pass to NDNLayer
        """
        super().__init__(
            input_dims=input_dims, num_filters=num_filters,
            #filter_dims=None,  # for now, not using so Non is correct
            NLtype=NLtype, norm_type=norm_type,
            pos_constraint=pos_constraint, num_inh=num_inh,
            bias=bias, weights_initializer=weights_initializer,
            output_norm=output_norm, initialize_center=initialize_center,
            bias_initializer=bias_initializer, reg_vals=reg_vals,
            **kwargs)

        # Now make mask
        assert mask is not None, "MASKLAYER: must include mask, dodo"
        assert np.prod(mask.shape) == np.prod(self.filter_dims)*num_filters
        self.register_buffer('mask', torch.tensor(mask.reshape([-1, num_filters]), dtype=torch.float32))
    # END MaskLayer.__init__

    def preprocess_weights( self ):
        """
        Preprocess weights for MaskLayer.

        Returns:
            w: torch.Tensor, preprocessed weights
        """
        w = super().preprocess_weights()
        return w*self.mask
    # END MaskLayer.preprocess_weights()

    @classmethod
    def layer_dict(cls, mask=None, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

        Args:
            mask: np.ndarray, mask to apply to the weights
            **kwargs: additional arguments to pass to NDNLayer

        Returns:
            Ldict: dict, dictionary of layer parameters
        """

        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'masklayer'
        Ldict['mask'] = mask
    
        return Ldict
    # END MaskLayer.layer_dict()