from numpy.lib.arraysetops import isin
import torch
from torch.nn import functional as F
import torch.nn as nn

from .ndnlayer import NDNLayer
import numpy as np

class ConvLayer(NDNLayer):
    """
    Convolutional NDN Layer

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
            filter_dims=None,
            temporal_tent_spacing=None,
            output_norm=None,
            stride=None,
            dilation=1,
            padding='same',
            res_layer=False,  # to make a residual layer
            window=None,
            folded_lags=False,
            **kwargs,
            ):

        assert input_dims is not None, "ConvLayer: Must specify input_dims"
        assert num_filters is not None, "ConvLayer: Must specify num_filters"
        if res_layer:
            assert padding == 'same', "ConvLayer: padding must be set to 'same' for res_layer"

        #assert (conv_dims is not None) or (filter_dims is not None), "ConvLayer: conv_dims or filter_dims must be specified"
        if 'conv_dims' in kwargs:
            print("No longer using conv_dims. Use filter_dims instead.")

        assert filter_dims is not None, "ConvLayer: filter_dims must be specified"

        #if conv_dims is None:
        #    from copy import copy
        #    conv_dims = copy(filter_dims[1:])
        
        is1D = input_dims[2] == 1
        # Place filter_dims information in correct place -- maybe overkill, but easy to specify 
        # spatial width in any number of ways using 'filter_dims'
        if isinstance(filter_dims, int):  # if pass single number (filter_width)
            filter_dims = [filter_dims]
        if len(filter_dims) == 1:
            if is1D:
                #conv_dims = [copy(conv_dims), input_dims[-1]]
                filter_dims = [input_dims[0], filter_dims[0], 1, input_dims[-1]]
            else:
                #conv_dims = [copy(conv_dims), copy(conv_dims), input_dims[-1]]
                filter_dims = [input_dims[0], filter_dims[0], filter_dims[0], input_dims[-1]]
        elif len(filter_dims) == 2:  # Assume 2-d and passing spatial dims
            assert not is1D, "ConvLayer: invalid filter_dims"
            filter_dims = [input_dims[0], filter_dims[0], filter_dims[1], input_dims[-1]]
        elif len(filter_dims) == 3:  # Assume didn't pass in filter dimension
            from copy import copy
            filter_dims = [input_dims[0]] + copy(filter_dims)
        # otherwise filter_dims is correct

        # Best practice for convolutions would have odd-dimension convolutional filters: 
        # will need full_padding and slowdown (possibly non-deterministic behavior) otherwise
        for ii in [1,2]:
            if filter_dims[ii]%2 != 1: print("ConvDim %d should be odd."%ii) 
                
        self.window = False
        # If tent-basis, figure out how many lag-dimensions using tent_basis transform
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

        #if filter_dims is None:
        #    filter_dims = [input_dims[0]] + conv_dims
        #else:            
        #    filter_dims[1:] = conv_dims
        
        super().__init__(
            input_dims=input_dims,
            num_filters=num_filters,
            filter_dims=filter_dims,
            **kwargs,
            )

        self.is1D = is1D
        self.res_layer = res_layer
       
        if self.tent_basis is not None:
            self.register_buffer('tent_basis', torch.Tensor(self.tent_basis.T))
            filter_dims[-1] = self.tent_basis.shape[0]
        else:
            self.tent_basis = None

        self.filter_dims = filter_dims

        # Checks to ensure cuda-bug with large convolutional filters is not activated #1
        # assert self.filter_dims[1] < self.input_dims[1], "Spatial Filter widths must be smaller than input dims."
        # # Check if 1 or 2-d convolution required
        # if self.input_dims[2] > 1:
        #     assert self.filter_dims[2] < self.input_dims[2], "Spatial Filter widths must be smaller than input dims."

        if stride is None:
            self.stride = 1   
        else: 
            self.stride = stride

        if dilation is None:
            self.dilation = 1
        else:
            self.dilation = dilation
        
        assert self.stride == 1, 'Cannot handle greater strides than 1.'
        assert self.dilation == 1, 'Cannot handle greater dilations than 1.'

        self.padding = padding   # self.padding will be a list of integers...    

        # These assignments will be moved to the setter with padding as property and _npads as internal
        # Define padding as "same" or "valid" and then _npads is the number of each edge
        #if self.stride > 1:
            #print('Warning: Manual padding not yet implemented when stride > 1')
            #self.padding = (0,0,0,0)
        #else:
            #w = self.filter_dims[1:3] # handle 2D if necessary
#            if padding == 'same':
#                self.padding = (w[0]//2, (w[0]-1+w[0]%2)//2, w[1]//2, (w[1]-1+w[1]%2)//2)
#            elif padding == 'valid':
#                self.padding = (0, 0, 0, 0)

        # Combine filter and temporal dimensions for conv -- collapses over both
        self.folded_dims = self.input_dims[0]*self.input_dims[3]

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
        else:
            self.output_norm = None

        if window is not None:
            if window == 'hamming':
                win=np.hamming(filter_dims[1])
                if self.is1D:
                    self.register_buffer('window_function', torch.tensor(win).type(torch.float32))
                else:
                    self.register_buffer('window_function', torch.tensor(np.outer(win,win)).type(torch.float32))
                self.window = True
            else:
                print("ConvLayer: unrecognized window")
    # END ConvLayer.__init__

    ## This is now defined in NDNLayer so inherited -- applies to all NDNLayer children
    #@property
    #def output_dims(self):
    #    self.num_outputs = int(np.prod(self._output_dims))
    #    return self._output_dims
    
    #@output_dims.setter
    #def output_dims(self, value):
    #    self._output_dims = value
    #    self.num_outputs = int(np.prod(self._output_dims))

    def batchnorm_convert( self ):
        """Converts layer with batch_norm to have the same output without batch_norm. This involves
        adjusting the weights and adding offsets"""

        from copy import deepcopy

        assert self.output_norm is not None, "Layer does not have batch norm"
        assert self.norm_type == 1, "Layer should have norm type 1"

        W = self.preprocess_weights()
        newW = deepcopy(W.data)
        newB = deepcopy(self.bias.data)
        for ii in range(self.num_filters):
            if self.output_norm.affine:
                gamma = self.output_norm.weight[ii].clone().detach()
                beta = self.output_norm.bias[ii].clone().detach()
            else:
                gamma = 1.0
                beta = 0.0
            E_x =  self.output_norm.running_mean[ii].clone().detach()
            Var_x =  self.output_norm.running_var[ii].clone().detach()
            
            newW.data[:, ii] = gamma/torch.sqrt(Var_x+1e-5)*W[:, ii]
            newB.data[ii] = beta - E_x*(gamma/torch.sqrt(Var_x+1e-5))

        #create new layer
        #layer = deepcopy(self)
        self.norm_type = 0
        self.output_norm = None
        #self.reset_parameters()
        self.weight.data = newW
        self.bias.data = newB
    # END self.batchnorm       

    @property
    def padding(self):
        return self._padding
    
    @padding.setter
    def padding(self, value):
        assert value in ['valid', 'same'], "ConvLayer: incorrect value entered for padding"
        self._padding = value
        self._fullpadding = False

        sz = self.filter_dims[1:3] # handle 2D if necessary
        if self._padding == 'valid':
            self._npads = (0, 0, 0, 0)
        else:
            assert self.stride == 1, "Warning: 'same' padding not yet implemented when stride > 1"
            self._fullpadding = self.filter_dims[1]%2 == 0
            if self.is1D:
                self._npads = (sz[0]//2, (sz[0]-1)//2)
            else:
                self._npads = (sz[1]//2, (sz[1]-1)//2, sz[0]//2, (sz[0]-1)//2)  # F.pad wants things backwards dims
                self._fullpadding = self._fullpadding or (self.filter_dims[2]%2 == 0)

        # Also adjust output dims
        new_output_dims = [
            self.num_filters, 
            self.input_dims[1] - sz[0] + 1 + self._npads[0]+self._npads[1], 
            1, 1]
        if not self.is1D:
            new_output_dims[2] = self.input_dims[2] - sz[1] + 1 + self._npads[2]+self._npads[3]
        
        self.output_dims = new_output_dims
        # This code no longer
        # if isinstance(value, int):
        #    npad = [value, value, value, value]
        #else:
        #    npad = [value[i*2]+value[i*2+1] for i in range(len(value)//2)]
        # handle spatial padding here, time is integrated out in conv base layer

    @classmethod
    def layer_dict(cls, padding='same', filter_dims=None, folded_lags=False, res_layer=False, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'conv'
        Ldict['filter_dims'] = filter_dims
        Ldict['res_layer'] = res_layer
        Ldict['temporal_tent_spacing'] = 1
        Ldict['output_norm'] = None
        Ldict['window'] = None  # could be 'hamming'
        Ldict['stride'] = 1
        Ldict['dilation'] = 1
        Ldict['padding'] = padding
        Ldict['folded_lags'] = folded_lags

        return Ldict
    
    def preprocess_weights(self):
        w = super().preprocess_weights()
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
                    w.view([-1, self.folded_dims, self.filter_dims[1]]), 
                    bias=self.bias,
                    stride=self.stride, dilation=self.dilation)
            else:
                y = F.conv1d(
                    s,
                    #w.view([-1, self.folded_dims, self.filter_dims[1]]), 
                    w.reshape([-1, self.folded_dims, self.filter_dims[1]]), 
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
                    bias=self.bias,
                    stride=self.stride,
                    dilation=self.dilation)
            else:
                # functional pads since padding is simple
                y = F.conv2d(
                    s, 
                    w.reshape([-1, self.folded_dims, self.filter_dims[1], self.filter_dims[2]]),
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

class TconvLayer(ConvLayer):
    """
    Temporal convolutional layer.
    TConv does not integrate out the time dimension and instead treats it as a true convolutional dimension

    Args:
        input_dims (list of ints): input dimensions [C, W, H, T]
        num_filters (int): number of filters
        filter_dims (list of ints): filter dimensions [C, w, h, T]
            w < W, h < H
        stride (int): stride of convolution

    
    """ 

    def __init__(self,
        input_dims=None, # [C, W, H, T]
        num_filters=None, # int
        conv_dims=None, # [w,h,t]
        filter_dims=None, # [C, w, h, t]
        padding='valid',
        output_norm=None,
        **kwargs):
        
        assert input_dims is not None, "TConvLayer: input_dims must be specified"
        assert num_filters is not None, "TConvLayer: num_filters must be specified"
        assert (conv_dims is not None) or (filter_dims is not None), "TConvLayer: conv_dims or filter_dims must be specified"
        
        # Convoluted way to use either filter_dims or conv_dims
        if (filter_dims is not None) and (not isinstance(filter_dims, list)):
            filter_dims = [filter_dims]
        if conv_dims is None:
            conv_dims = filter_dims[1:]
        if filter_dims is None:
            filter_dims = [input_dims[0]] + conv_dims
        else:
            filter_dims[1:] = conv_dims

        # All parameters of filter (weights) should be correctly fit in layer_params
        super().__init__(
            input_dims=input_dims, num_filters=num_filters, 
            filter_dims=filter_dims, padding=padding, output_norm=output_norm, 
            **kwargs)

        self.num_lags = self.input_dims[3]

        # Check if 1 or 2-d convolution required
        self.is1D = (self.input_dims[2] == 1)
        # "1D" means one spatial dimension is singleton

        # padding now property from ConvLayer: set in overload of setter
        self.padding = padding
        
        # check if output normalization is specified
        if output_norm == 'batch':
            if self.is1D:
                self.output_norm = nn.BatchNorm2d(self.num_filters)
            else:
                self.output_norm = nn.BatchNorm3d(self.num_filters)
        else:
            self.output_norm = None
    #END TconvLayer.__init__

    @property
    def padding(self):
        return super().padding

    @padding.setter
    def padding(self, value):
        assert value in ['valid', 'same', 'spatial'], "TconvLayer: incorrect value entered for padding"
        self._padding = value
        self._fullpadding = False
        sz = self.filter_dims[1:] # handle 2D if necessary

        if self.is1D:
            if self._padding == 'valid':
                self._npads = 0
            # TODO: delete this 
            #  elif self._padding == 'same':
            #    self._npads = (self.filter_dims[-1]-1, 0,
            #        self.filter_dims[1]//2, (self.filter_dims[1] - 1 + self.filter_dims[1]%2)//2)
            elif self._padding == 'spatial':
                self._npads = (0, 0,
                    self.filter_dims[1]//2, (self.filter_dims[1] - 1 + self.filter_dims[1]%2)//2)
            self._fullpadding = self.filter_dims[1]%2 == 0
        else:
            if self._padding == 'valid':
                self._npads = (0, 0, 0, 0, 0, 0)
                
            elif self._padding == 'same':
                assert self.stride == 1, "Warning: 'same' padding not yet implemented when stride > 1"
                self._npads = (self.filter_dims[-1]-1, 0,
                    self.filter_dims[1]//2,
                    (self.filter_dims[1] - 1 + self.filter_dims[1]%2)//2,
                    self.filter_dims[2]//2, (self.filter_dims[2] - 1 + self.filter_dims[2]%2)//2)
            elif self._padding == 'spatial':
                self._npads = (0, 0,
                    self.filter_dims[1]//2,
                    (self.filter_dims[1] - 1 + self.filter_dims[1]%2)//2,
                    self.filter_dims[2]//2, (self.filter_dims[2] - 1 + self.filter_dims[2]%2)//2)
            self._fullpadding = self._fullpadding or (self.filter_dims[2]%2 == 0)

        # Also adjust spatial output dims
        new_output_dims = [
            self.num_filters, 
            self.input_dims[1] - sz[0] + 1 + self._npads[2]+self._npads[3], 
            1, 
            self.input_dims[3] - sz[2] + 1 + self._npads[0]+self._npads[1]]
        if not self.is1D:
            new_output_dims[2] = self.input_dims[2] - sz[1] + 1 + self._npads[4]+self._npads[5]
        
        self.output_dims = new_output_dims
        #self.output_dims = self.output_dims # annoying fix for the num_outputs dependency on all output_dims values being updated

    def forward(self, x):

        w = self.preprocess_weights()
        if self.is1D:
            w = w.view(self.filter_dims[:2] + [self.filter_dims[3]] + [-1]).permute(3,0,1,2) # [C,H,T,N]->[N,C,H,T]
            s = x.view([-1] + self.input_dims[:2]+[self.input_dims[3]]) # [B,C,W,T]
            if self.padding:
                s = F.pad(s, self._npads, "constant", 0)

            y = F.conv2d(
                s,
                w, 
                bias=self.bias,
                stride=self.stride, dilation=self.dilation)

        else:

            w = w.view(self.filter_dims + [self.num_filters]).permute(4,0,1,2,3) # [C,H,W,T,N]->[N,C,H,W,T]
            x = x.view([-1] + self.input_dims) # [B,C*W*H*T]->[B,C,W,H,T]

            if self.padding:
                x = F.pad(x, self._npads, "constant", 0)

            y = F.conv3d(
                x,
                w, 
                bias=self.bias,
                stride=self.stride, dilation=self.dilation)
        
        if self.output_norm is not None:
            y = self.output_norm(y)

        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)
        
        if self._ei_mask is not None:
            if self.is1D:
                y = y * self._ei_mask[None,:,None,None]
            else:
                y = y * self._ei_mask[None,:,None,None,None]
        
        y = y.view((-1, self.num_outputs))

        # store activity regularization to add to loss later
        #self.activity_regularization = self.activity_reg.regularize(y)
        if hasattr(self.reg, 'activity_regmodule'):  # to put buffer in case old model
            self.reg.compute_activity_regularization(y)

        return y
    #END TconvLayer.forward

    def plot_filters( self, cmaps='gray', num_cols=8, row_height=2, time_reverse=None):
        # Place-holder: does nothing specific that NDNLayer does not do  

        super().plot_filters( 
            cmaps=cmaps, num_cols=num_cols, row_height=row_height, 
            time_reverse=time_reverse)

    @classmethod
    def layer_dict(cls, padding='spatial', conv_dims=None, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(padding=padding, **kwargs)
        # Added arguments
        Ldict['layer_type'] = 'tconv'
        Ldict['conv_dims'] = conv_dims
        return Ldict
    # END [classmethod] TconvLayer.layer_dict

class STconvLayer(TconvLayer):
    """
    Spatio-temporal convolutional layer.
    STConv Layers overload the batch dimension and assume they are contiguous in time.

    Args:
        input_dims (list of ints): input dimensions [C, W, H, T]
            This is, of course, not the real input dimensions, because the batch dimension is assumed to be contiguous.
            The real input dimensions are [B, C, W, H, 1], T specifies the number of lags
        num_filters (int): number of filters
        filter_dims (list of ints): filter dimensions [C, w, h, T]
            w < W, h < H
        stride (int): stride of convolution

    
    """ 

    def __init__(self,
        input_dims=None, # [C, W, H, T]
        num_filters=None, # int
        output_norm=None,
        **kwargs):

        # conv_dims=None, # [w,h,t]
        # filter_dims=None, # [C, w, h, t]
        
        assert input_dims is not None, "STConvLayer: input_dims must be specified"
        assert num_filters is not None, "STConvLayer: num_filters must be specified"
        # assert (conv_dims is not None) or (filter_dims is not None), "STConvLayer: conv_dims or filter_dims must be specified"
        
        # if conv_dims is None:
        #     conv_dims = filter_dims[1:]

        # All parameters of filter (weights) should be correctly fit in layer_params
        super().__init__(input_dims,
            num_filters, output_norm=output_norm, **kwargs)

        #assert self.input_dims[3] == self.filter_dims[3], "STConvLayer: input_dims[3] must equal filter_dims[3]"    
        self.input_dims[3] = self.filter_dims[3]  # num_lags for convolution specified in filter_dims (not input_dims)
        self.num_lags = self.input_dims[3]
        self.input_dims[3] = 1  # take lag info and use for temporal convolution
        self.output_dims[-1] = 1
        self.output_dims = self.output_dims # annoying fix for the num_outputs dependency on all output_dims values being updated
    # END STconvLayer.__init__

    def forward(self, x):
        # Reshape stim matrix LACKING temporal dimension [bcwh] 
        # and inputs (note uses 4-d rep of tensor before combinine dims 0,3)
        # pytorch likes 3D convolutions to be [B,C,T,W,H].
        # I benchmarked this and it'sd a 20% speedup to put the "Time" dimension first.

        w = self.preprocess_weights()
        if self.is1D:
            s = x.reshape([-1] + self.input_dims[:3]).permute(3,1,0,2) # [B,C,W,1]->[1,C,B,W]
            w = w.reshape(self.filter_dims[:2] + [self.filter_dims[3]] +[-1]).permute(3,0,2,1) # [C,H,T,N]->[N,C,T,W]

            if self.padding:
                # flip order of padding for STconv -- last two are temporal padding
                pad = (self._npads[2], self._npads[3], self.filter_dims[-1]-1, 0)
            else:
                # still need to pad the batch dimension
                pad = (0,0,self.filter_dims[-1]-1,0)

            s = F.pad(s, pad, "constant", 0)

            y = F.conv2d(
                s,
                w, 
                bias=self.bias,
                stride=self.stride, dilation=self.dilation)
            
            y = y.permute(2,1,3,0) # [1,N,B,W] -> [B,N,W,1]

        else:
            s = x.reshape([-1] + self.input_dims).permute(4,1,0,2,3) # [1,C,B,W,H]
            w = w.reshape(self.filter_dims + [-1]).permute(4,0,3,1,2) # [N,C,T,W,H]
            
            if self.padding:
                pad = (self._npads[2], self._npads[3], self._npads[4], self._npads[5], self.filter_dims[-1]-1,0)
            else:
                # still need to pad the batch dimension
                pad = (0,0,0,0,self.filter_dims[-1]-1,0)

            s = F.pad(s, pad, "constant", 0)

            y = F.conv3d(
                s,
                w, 
                bias=self.bias,
                stride=self.stride, dilation=self.dilation)

            y = y.permute(2,1,3,4,0) # [1,N,B,W,H] -> [B,N,W,H,1]
        
        if not self.res_layer:
            if self.output_norm is not None:
                y = self.output_norm(y)

        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)
        
        if self._ei_mask is not None:
            if self.is1D:
                y = y * self._ei_mask[None,:,None,None]
            else:
                y = y * self._ei_mask[None,:,None,None,None]
        
        if self.res_layer:
            # s is with dimensions: B, C, T, X, Y 
            if self.is1D:
                y = y + torch.reshape( s, (-1, self.folded_dims, self.input_dims[1]) )
            else:
                y = y + torch.reshape( s, (-1, self.folded_dims, self.input_dims[1], self.input_dims[2]) )
                 
            if self.output_norm is not None:
                y = self.output_norm(y)
        
        y = y.reshape((-1, self.num_outputs))

        # store activity regularization to add to loss later
        #self.activity_regularization = self.activity_reg.regularize(y)
        if hasattr(self.reg, 'activity_regmodule'):  # to put buffer in case old model
            self.reg.compute_activity_regularization(y)

        return y
    # END STconvLayer.forward 

    def plot_filters( self, cmaps='gray', num_cols=8, row_height=2, time_reverse=False):
        # Overload plot_filters to automatically time_reverse
        super().plot_filters( 
            cmaps=cmaps, num_cols=num_cols, row_height=row_height, 
            time_reverse=time_reverse)

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
        Ldict['layer_type'] = 'stconv'
        return Ldict
    # END [classmethod] STconvLayer.layer_dict
    