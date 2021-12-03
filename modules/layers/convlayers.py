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
        conv_width: width of convolutional kernel (int or list of ints)
    Args (optional):
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'

    """
    def __init__(self,
            input_dims=None,
            num_filters=None,
            conv_dims=None,
            filter_dims=None,
            output_norm=None,
            stride=None,
            dilation=1,
            **kwargs,
            ):

        assert input_dims is not None, "ConvLayer: Must specify input_dims"
        assert num_filters is not None, "ConvLayer: Must specify num_filters"
        assert (conv_dims is not None) or (filter_dims is not None), "ConvLayer: conv_dims or filter_dims must be specified"
        
        if conv_dims is None:
            conv_dims = filter_dims[1:]

        if filter_dims is None:
            if isinstance(conv_dims, int):
                filter_dims = [input_dims[0], conv_dims, conv_dims, input_dims[3]]
            elif isinstance(conv_dims, list):
                if len(conv_dims) == 1:
                    filter_dims = [input_dims[0], conv_dims[0], conv_dims[0], input_dims[3]]
                elif len(conv_dims) == 2:
                    filter_dims = [input_dims[0], conv_dims[0], conv_dims[1], input_dims[3]]
                elif len(conv_dims) == 3:
                    filter_dims = [input_dims[0], conv_dims[0], conv_dims[1], conv_dims[2]]
                else:
                    raise ValueError('conv_width must be int or list of length 1, 2, or 3.')

        super().__init__(input_dims,
            num_filters,
            filter_dims=filter_dims,
            **kwargs,
            )

        output_dims = [num_filters, 1, 1, 1]
        output_dims[1:3] = input_dims[1:3]
        self.output_dims = output_dims

        if input_dims[2] == 1:  # then 1-d spatial
            filter_dims[2] = 1
            self.is1D = (self.input_dims[2] == 1)
        else:
            self.is1D = False

        self.filter_dims = filter_dims
        # Checks to ensure cuda-bug with large convolutional filters is not activated #1
        assert self.filter_dims[1] < self.input_dims[1], "Filter widths must be smaller than input dims."
        # Check if 1 or 2-d convolution required
        if self.input_dims[2] > 1:
            assert self.filter_dims[2] < self.input_dims[2], "Filter widths must be smaller than input dims."

        if stride is None:
            self.stride = 1   
        else: 
            self.stride = stride

        if dilation is None:
            self.dilation = 1
        else:
            self.dilation = dilation

        if self.stride > 1:
            print('Warning: Manual padding not yet implemented when stride > 1')
            self.padding = 0
        else:
            w = self.filter_dims[1:3] # handle 2D if necessary
            self.padding = (w[0]//2, (w[0]-1+w[0]%2)//2, w[1]//2, (w[1]-1+w[1]%2)//2)

        # Combine filter and temporal dimensions for conv -- collapses over both
        self.folded_dims = self.input_dims[0]*self.input_dims[3]
        self.num_outputs = np.prod(self.output_dims)

        # check if output normalization is specified
        if output_norm is 'batch':
            if self.is1D:
                self.output_norm = nn.BatchNorm1d(self.num_filters)
            else:
                self.output_norm = nn.BatchNorm2d(self.num_filters)
        else:
            self.output_norm = None


    def forward(self, x):
        # Reshape weight matrix and inputs (note uses 4-d rep of tensor before combinine dims 0,3)
        s = x.reshape([-1]+self.input_dims).permute(0,1,4,2,3)
        w = self.preprocess_weights().reshape(self.filter_dims+[-1]).permute(4,0,3,1,2)
        # puts output_dims first, as required for conv

        # Collapse over irrelevant dims for dim-specific convs
        if self.is1D:
            s = torch.reshape( s, (-1, self.folded_dims, self.input_dims[1]) )
            y = F.conv1d(
                F.pad(s, self.padding, "constant", 0), # we do our own padding
                torch.reshape( w, (-1, self.folded_dims, self.filter_dims[1]) ), 
                bias=self.bias,
                stride=self.stride, dilation=self.dilation)
        else:
            s = torch.reshape( s, (-1, self.folded_dims, self.input_dims[1], self.input_dims[2]) )
            y = F.conv2d(
                F.pad(s, self.padding, "constant", 0), # we do our own padding
                torch.reshape( w, (-1, self.folded_dims, self.filter_dims[1], self.filter_dims[2]) ), 
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation)

        if self.output_norm is not None:
            y = self.output_norm(y)

        y = torch.reshape(y, (-1, self.num_outputs))
        
        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)
        
        if self.ei_mask is not None:
            y = y * self.ei_mask

        return y




class STconvLayer(ConvLayer):
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
        conv_dims=None, # [w,h,t]
        filter_dims=None, # [C, w, h, t]
        temporal_tent_spacing=None,
        stride=1,
        dilation=1,
        **kwargs):
        
        assert input_dims is not None, "STConvLayer: input_dims must be specified"
        assert num_filters is not None, "STConvLayer: num_filters must be specified"
        assert (conv_dims is not None) or (filter_dims is not None), "STConvLayer: conv_dims or filter_dims must be specified"
        
        if conv_dims is None:
            conv_dims = filter_dims[1:]

        # If tent-basis, figure out how many lag-dimensions using tent_basis transform
        tent_basis = None
        if temporal_tent_spacing is not None:
            from NDNT.utils import tent_basis_generate
            num_lags = conv_dims[2]
            tentctrs = list(np.arange(0, num_lags, temporal_tent_spacing))
            tent_basis = tent_basis_generate(tentctrs)
            if tent_basis.shape[0] != num_lags:
                print('Warning: tent_basis.shape[0] != num_lags')
                print('tent_basis.shape = ', tent_basis.shape)
                print('num_lags = ', num_lags)
                print('Adding zeros or truncating to match')
                if tent_basis.shape[0] > num_lags:
                    print('Truncating')
                    tent_basis = tent_basis[:num_lags,:]
                else:
                    print('Adding zeros')
                    tent_basis = np.concatenate([tent_basis, np.zeros((num_lags-tent_basis.shape[0], tent_basis.shape[1]))], axis=0)
                
            tent_basis = tent_basis[:num_lags,:]
            num_lag_params = tent_basis.shape[1]
            print('STconv: num_lag_params =', num_lag_params)
            conv_dims[2] = num_lag_params

        if filter_dims is None:
            filter_dims = [input_dims[0]] + conv_dims
        else:            
            filter_dims[1:] = conv_dims

        assert stride == 1, 'Cannot handle greater strides than 1.'
        assert dilation == 1, 'Cannot handle greater dilations than 1.'

        # All parameters of filter (weights) should be correctly fit in layer_params
        super().__init__(input_dims,
            num_filters, conv_dims, stride=stride, dilation=dilation, **kwargs)

        self.num_lags = self.input_dims[3]
        self.input_dims[3] = 1  # take lag info and use for temporal convolution

        if tent_basis is not None:
            self.register_buffer('tent_basis', torch.Tensor(tent_basis.T))
        else:
            self.tent_basis = None

        # Check if 1 or 2-d convolution required
        self.is1D = (self.input_dims[2] == 1)
        # "1D" really means a 2D convolution (1D space, 1D time) since time is handled with
        # convolutions instead of embedding lags

        # Do spatial padding by hand -- will want to generalize this for two-ds
        if self.is1D:
            self.padding = (self.filter_dims[1]//2, (self.filter_dims[1] - 1 + self.filter_dims[1]%2)//2,
                self.num_lags-1, 0)
        else:
            # Checks to ensure cuda-bug with large convolutional filters is not activated #2
            assert self.filter_dims[2] < self.input_dims[2], "Filter widths must be smaller than input dims."

            self.padding = (self.filter_dims[1]//2, (self.filter_dims[1] - 1 + self.filter_dims[1]%2)//2,
                self.filter_dims[2]//2, (self.filter_dims[2] - 1 + self.filter_dims[2]%2)//2,
                self.num_lags-1, 0)


        # Combine filter and temporal dimensions for conv -- collapses over both
        self.num_outputs = np.prod(self.output_dims)


    def forward(self, x):
        # Reshape stim matrix LACKING temporal dimension [bcwh] 
        # and inputs (note uses 4-d rep of tensor before combinine dims 0,3)
        # pytorch likes 3D convolutions to be [B,C,T,W,H].
        # I benchmarked this and it'sd a 20% speedup to put the "Time" dimension first.

        w = self.preprocess_weights()
        if self.is1D:
            s = x.reshape([-1] + self.input_dims[:3]).permute(3,1,0,2) # [B,C,W,1]->[1,C,B,W]
            w = w.reshape(self.filter_dims[:2] + [self.filter_dims[3]] +[-1]).permute(3,0,2,1) # [C,H,T,N]->[N,C,T,W]
            # time-expand using tent-basis if it exists
            if self.tent_basis is not None:
                w = torch.einsum('nctw,tz->nczw', w, self.tent_basis)

            y = F.conv2d(
                F.pad(s, self.padding, "constant", 0),
                w, 
                bias=self.bias,
                stride=self.stride, dilation=self.dilation)
            y = y.permute(2,1,3,0) # [1,N,B,W] -> [B,N,W,1]

        else:
            s = x.reshape([-1] + self.input_dims).permute(4,1,0,2,3) # [1,C,B,W,H]
            w = w.reshape(self.filter_dims + [-1]).permute(4,0,3,1,2) # [N,C,T,W,H]
            
            # time-expand using tent-basis if it exists
            if self.tent_basis is not None:
                w = torch.einsum('nctwh,tz->nczwh', w, self.tent_basis)

            y = F.conv3d(
                F.pad(s, self.padding, "constant", 0),
                w, 
                bias=self.bias,
                stride=self.stride, dilation=self.dilation)
            y = y.permute(2,1,3,4,0)    
        
        if self.output_norm is not None:
            y = self.output_norm(y)
        
        y = y.reshape((-1, self.num_outputs))

        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)
        return y

    def plot_filters( self, cmaps='gray', num_cols=8, row_height=2, time_reverse=False):
        # Overload plot_filters to automatically time_reverse
        super().plot_filters( 
            cmaps=cmaps, num_cols=num_cols, row_height=row_height, 
            time_reverse=time_reverse)