import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .ndnlayer import NDNLayer
from .convlayers import ConvLayer
from .convlayers import TconvLayer


class IterLayer(ConvLayer):
    """
    Residual network layer based on conv-net setup but with different forward.
    Namely, the forward includes a skip-connection, and has to be 'same'

    Args (required):
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_width: width of convolutional kernel (int or list of ints)
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
            filter_width=None,
            num_iter=1, # Added
            output_config='last',  # Added: alternative 'full'
            temporal_tent_spacing=None,
            output_norm=None,
            window=None,
            res_layer=True,
            **kwargs,
            ):

        super().__init__(
            input_dims=input_dims,
            num_filters=num_filters,
            filter_dims=filter_width,
            temporal_tent_spacing=temporal_tent_spacing,
            output_norm=None, # do not apply output_norm within layer
            stride=None,
            padding='same',
            dilation=1,
            window=window,
            **kwargs,
            )
        
        self.num_iter = num_iter

        self.output_config = output_config
        if self.output_config != 'last':  # then 'full'
            self.output_dims[0] *= num_iter
            self.num_outputs *= num_iter

        self.res_layer = res_layer

        if output_norm in ['batch', 'batchX']:
            if output_norm == 'batchX':
                affine = False
            else:
                affine = True
            self.output_norm = nn.ModuleList()
            for iter in range(self.num_iter):
                if self.is1D:
                    self.output_norm.append(nn.BatchNorm1d(self.num_filters, affine=affine))
                else:
                    self.output_norm.append(nn.BatchNorm2d(self.num_filters, affine=affine))
        else:
            self.output_norm = None

    # END IterLayer.Init

    @classmethod
    def layer_dict(cls, filter_width=None, num_iter=1, output_config='last', res_layer=True, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'iter'
        Ldict['filter_width'] = filter_width
        Ldict['num_iter'] = num_iter
        Ldict['temporal_tent_spacing'] = 1
        Ldict['output_norm'] = None
        Ldict['window'] = None  # could be 'hamming'
        Ldict['res_layer'] = res_layer
        #Ldict['stride'] = 1
        #Ldict['dilation'] = 1
        Ldict['output_config'] = output_config 
        # These are options we are taking away
        del Ldict['stride']
        del Ldict['dilation']
        del Ldict['padding']
        del Ldict['filter_dims']

        return Ldict
    
    def forward(self, x):
        # Reshape weight matrix and inputs (note uses 4-d rep of tensor before combinine dims 0,3)

        #return x + super().forward(x)
        ## THIS IS CUT-AND-PASTE FROM ConvLayer -- with mods

        # Prepare stimulus
        x = x.reshape([-1]+self.input_dims).permute(0,1,4,2,3)
        if self.is1D:
            x = torch.reshape( x, (-1, self.folded_dims, self.input_dims[1]) )
        else:
            x = torch.reshape( x, (-1, self.folded_dims, self.input_dims[1], self.input_dims[2]) )

        # Prepare weights
        w = self.preprocess_weights().reshape(self.filter_dims+[self.num_filters]).permute(4,0,3,1,2)
        # puts output_dims first, as required for conv

        outputs = None
        for iter in range(self.num_iter):

            if self.is1D:

                if self._fullpadding:
                    #s = F.pad(s, self._npads, "constant", 0)
                    y = F.conv1d(
                        F.pad(s, self._npads, "constant", 0),
                        w.view([-1, self.folded_dims, self.filter_dims[1]]), 
                        bias=self.bias,
                        stride=self.stride, dilation=self.dilation)
                else:
                    y = F.conv1d(
                        x,
                        w.view([-1, self.folded_dims, self.filter_dims[1]]), 
                        bias=self.bias,
                        padding=self._npads[0],
                        stride=self.stride, dilation=self.dilation)

            else:

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
                        x, 
                        w.reshape([-1, self.folded_dims, self.filter_dims[1], self.filter_dims[2]]),
                        padding=(self._npads[2], self._npads[0]),
                        bias=self.bias,
                        stride=self.stride,
                        dilation=self.dilation)


            # Nonlinearity
            if self.NL is not None:
                y = self.NL(y)

            if self.res_layer:
                y = y + x  # Add input back in

            if self._ei_mask is not None:
                if self.is1D:
                    y = y * self._ei_mask[None, :, None]
                else:
                    y = y * self._ei_mask[None, :, None, None]

            if self.output_norm is not None:
                y = self.output_norm[iter](y)

            if (self.output_config == 'full') | (iter == self.num_iter-1):
                if outputs is None:
                    outputs = y
                else:
                    outputs = torch.cat( (outputs, y), dim=1)
            x = y
        # end of iteration loop

        return torch.reshape(outputs, (-1, self.num_outputs))
    # END IterLayer.forward


class IterTlayer(TconvLayer):
    """
    Residual network layer based on conv-net setup but with different forward.
    Namely, the forward includes a skip-connection, and has to be 'same'

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
            input_dims=None,  # [C, W, H, T]
            num_filters=None, # int
            filter_width=None, # use to assemble filter_dims[C, w, h, t]
            num_iter=1, # Added
            num_lags=1, # Added
            res_layer=True,
            output_config='last',  # Added: alternative 'full'
            #temporal_tent_spacing=None,
            output_norm=None,
            #window=None,
            **kwargs,
            ):

        filter_dims = [input_dims[0], filter_width, filter_width, num_lags]
        if input_dims[2] == 1:
            filter_dims[2] = 1

        super().__init__(
            input_dims=input_dims, 
            num_filters=num_filters,
            filter_dims=filter_dims,
            #temporal_tent_spacing=temporal_tent_spacing,
            output_norm=output_norm, 
            padding='spatial',
            #window=window,
            **kwargs,
            )
              
        self.num_iter = num_iter
        self.output_config = output_config
        self.res_layer = res_layer

        # Remaing lags after layer are output
        self.output_dims[3] = input_dims[3]-(num_lags-1)*num_iter

        if self.output_config != 'last':  # then 'full'
            self.output_dims[0] *= num_iter
        self.num_outputs = np.prod(self.output_dims)

        if output_norm in ['batch', 'batchX']:
            if output_norm == 'batchX':
                affine = False
                print('actually not using this right here')
            else:
                affine = True
            self.output_norm = nn.ModuleList()
            for iter in range(self.num_iter):
                if self.is1D:
                    self.output_norm.append(nn.BatchNorm2d(self.num_filters))
                else:
                    self.output_norm.append(nn.BatchNorm3d(self.num_filters))
        else:
            self.output_norm = None

    # END IterLayerT.Init

    @classmethod
    def layer_dict(cls, filter_width=None, num_iter=1, num_lags=1, res_layer=True, output_config='last', **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'iterT'
        Ldict['filter_width'] = filter_width
        Ldict['num_iter'] = num_iter
        Ldict['num_lags'] = num_lags
        Ldict['res_layer'] = res_layer
        #Ldict['temporal_tent_spacing'] = 1
        Ldict['output_norm'] = None
        Ldict['window'] = None  # could be 'hamming'
        Ldict['output_config'] = output_config 
        # These are options we are taking away
        del Ldict['stride']
        del Ldict['filter_dims']
        del Ldict['dilation']
        del Ldict['padding']

        return Ldict
    # END IterLayerT.LayerDict
    
    def forward(self, x):
        # Reshape weight matrix and inputs (note uses 4-d rep of tensor before combinine dims 0,3)

        w = self.preprocess_weights()

        if self.is1D:
            x = x.view([-1] + self.input_dims[:2]+[self.input_dims[3]]) # [B,C,W,T]
            w = w.reshape(self.filter_dims[:2] + [self.filter_dims[3]] + [-1]).permute(3,0,1,2) # [C,H,T,N]->[N,C,H,T]

        else:
            w = w.view(self.filter_dims + [self.num_filters]).permute(4,0,1,2,3) # [C,H,W,T,N]->[N,C,H,W,T]
            x = x.view([-1] + self.input_dims) # [B,C*W*H*T]->[B,C,W,H,T]

        # Prepare weights
        #w = self.preprocess_weights().reshape(self.filter_dims+[self.num_filters]).permute(4,0,3,1,2)
        # puts output_dims first, as required for conv

        reduce_lag = self.filter_dims[3]-1

        outputs = None
        for iter in range(self.num_iter):

            if self.padding:
                s = F.pad(x, self._npads, "constant", 0)
            else:
                s = x

            if self.is1D:
                #w = w.view(self.filter_dims[:2] + [self.filter_dims[3]] + [-1]).permute(3,0,1,2) # [C,H,T,N]->[N,C,H,T] 
                y = F.conv2d(
                    s,
                    w, 
                    bias=self.bias,
                    stride=self.stride, dilation=self.dilation)

            else:

                y = F.conv3d(
                    s,
                    w, 
                    bias=self.bias,
                    stride=self.stride, dilation=self.dilation)

            # Nonlinearity
            if self.NL is not None:
                y = self.NL(y)

            if self.res_layer:
                y = y + x[..., :-reduce_lag]  # Add input back in (without oldest lag)

            if self._ei_mask is not None:
                if self.is1D:
                    y = y * self._ei_mask[None, :, None, None]
                else:
                    y = y * self._ei_mask[None, :, None, None, None]

            if self.output_norm is not None:
                y = self.output_norm[iter](y)

            if (self.output_config == 'full') | (iter == self.num_iter-1):
                if outputs is None:
                    outputs = y[..., 0]
                else:
                    outputs = torch.cat( (outputs, y[..., 0]), dim=1)
            x = y
        # end of iteration loop
        return torch.reshape(outputs, (-1, self.num_outputs))
    # END IterLayerT.forward


##### THIS IS EASIER IMPLEMENTED IN ConvLayer Directly -- so not used so far
class ResLayer(ConvLayer):
    """
    Residual network layer based on conv-net setup but with different forward.
    Namely, the forward includes a skip-connection, and has to be 'same'

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
            filter_dims=None,
            temporal_tent_spacing=None,
            output_norm=None,
            window=None,
            **kwargs,
            ):

        super().__init__(
            input_dims=input_dims,
            num_filters=num_filters,
            filter_dims=filter_dims,
            temporal_tent_spacing=temporal_tent_spacing,
            output_norm=output_norm,
            stride=None,
            padding='same',
            dilation=1,
            window=window,
            **kwargs,
            )

        # This has exactly the same initializer and variables as ConvLayer
        # but will just have the res-net twist


    @classmethod
    def layer_dict(cls, filter_dims=None, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'res'
        Ldict['filter_dims'] = filter_dims
        Ldict['temporal_tent_spacing'] = 1
        Ldict['output_norm'] = None
        Ldict['window'] = None  # could be 'hamming'
        Ldict['stride'] = 1
        Ldict['dilation'] = 1

        return Ldict
    # END IterLayerT
    
    def forward(self, x):
        # Reshape weight matrix and inputs (note uses 4-d rep of tensor before combinine dims 0,3)

        s = x.reshape([-1]+self.input_dims).permute(0,1,4,2,3)

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
                    w.view([-1, self.folded_dims, self.filter_dims[1]]), 
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

        y = torch.reshape(y, (-1, self.num_outputs))
        
        return y
    # END ConvLayer.forward


