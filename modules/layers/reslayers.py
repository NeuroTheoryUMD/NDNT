import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .ndnlayer import NDNLayer
from .convlayers import ConvLayer
from .convlayers import TconvLayer
from .convlayers import STconvLayer


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
            LN_reverse=False,
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
        self.LN_reverse = LN_reverse
        if LN_reverse and self.pos_constraint:
            print('WARNING: should not use pos_constraint with LN_reverse')
            
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
    def layer_dict(cls, filter_width=None, num_iter=1, output_config='last', res_layer=True, LN_reverse=False, **kwargs):
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
        Ldict['LN_reverse'] = LN_reverse
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

            if self.LN_reverse:
                if self.NL is not None:
                    x = self.NL(x)

            if self.is1D:

                if self._fullpadding:
                    #s = F.pad(s, self._npads, "constant", 0)
                    y = F.conv1d(
                        F.pad(x, self._npads, "constant", 0),
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
                    x = F.pad(x, self._npads, "constant", 0)
                    y = F.conv2d(
                        x, # we do our own padding
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
            if (self.NL is not None) &  (not self.LN_reverse):
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
                    outputs = y[..., :self.output_dims[3]]
                else:
                    outputs = torch.cat( (outputs, y[..., :self.output_dims[3]]), dim=1)
            x = y
        # end of iteration loop
        return torch.reshape(outputs, (-1, self.num_outputs))
    # END IterLayerT.forward


class IterSTlayer(STconvLayer):
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
        self.output_dims[3] = 1 #input_dims[3]-(num_lags-1)*num_iter

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
        Ldict['layer_type'] = 'iterST'
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
    # END IterSTlayer.LayerDict
    
    def forward(self, x):
        # Reshape weight matrix and inputs (note uses 4-d rep of tensor before combinine dims 0,3)

        w = self.preprocess_weights()

        if self.is1D:
            x = x.view([-1] + self.input_dims[:3])
            w = w.view(self.filter_dims[:2] + [self.filter_dims[3]] +[-1]).permute(3,0,2,1) # [C,H,T,N]->[N,C,T,W]

            #x = x.view([-1] + self.input_dims[:2]+[self.input_dims[3]]) # [B,C,W,T]
            #w = w.reshape(self.filter_dims[:2] + [self.filter_dims[3]] + [-1]).permute(3,0,1,2) # [C,H,T,N]->[N,C,H,T]

            if self.padding:
                # flip order of padding for STconv -- last two are temporal padding
                pad = (self._npads[2], self._npads[3], self.filter_dims[-1]-1, 0)
            else:
                # still need to pad the batch dimension
                pad = (0,0,self.filter_dims[-1]-1,0)

        else:
            #w = w.view(self.filter_dims + [self.num_filters]).permute(4,0,1,2,3) # [C,H,W,T,N]->[N,C,H,W,T]
            #x = x.view([-1] + self.input_dims) # [B,C*W*H*T]->[B,C,W,H,T]

            x = x.reshape([-1] + self.input_dims)
            w = w.reshape(self.filter_dims + [-1]).permute(4,0,3,1,2) # [N,C,T,W,H]
            
            if self.padding:
                pad = (self._npads[2], self._npads[3], self._npads[4], self._npads[5], self.filter_dims[-1]-1,0)
            else:
                # still need to pad the batch dimension
                pad = (0,0,0,0,self.filter_dims[-1]-1,0)

        # Prepare weights
        #w = self.preprocess_weights().reshape(self.filter_dims+[self.num_filters]).permute(4,0,3,1,2)
        # puts output_dims first, as required for conv

        #reduce_lag = self.filter_dims[3]-1

        outputs = None
        for iter in range(self.num_iter):

            # Pad at least the temporal dimensions if not the spatial dimension too
            #x = F.pad(x, pad, "constant", 0)
            # Rotate stimulus into proper convolutonal form

            if self.is1D:
                s = x.permute(3,1,0,2) # [B,C,W,1]->[1,C,B,W]
                #w = w.view(self.filter_dims[:2] + [self.filter_dims[3]] + [-1]).permute(3,0,1,2) # [C,H,T,N]->[N,C,H,T] 
                y = F.conv2d(
                    F.pad(s, pad, "constant", 0),
                    w, 
                    bias=self.bias,
                    stride=self.stride, dilation=self.dilation)
            
                #y = y.permute(2,1,3,0) # [1,N,B,W] -> [B,N,W,1] -- do this after adding x

            else:
                s = x.permute(4,1,0,2,3) # -> [1,C,B,W,H]
                y = F.conv3d(
                    F.pad(s, pad, "constant", 0),
                    w, 
                    bias=self.bias,
                    stride=self.stride, dilation=self.dilation)

                # y = y.permute(2,1,3,4,0) # [1,N,B,W,H] -> [B,N,W,H,1] -- do this after adding x

            # Rotate back to add and process
            if self.is1D:
                y = y.permute(2,1,3,0) # [1,N,B,W] -> [B,N,W,1] 
            else:
                y = y.permute(2,1,3,4,0) # [1,N,B,W,H] -> [B,N,W,H,1]

            # Nonlinearity
            if self.NL is not None:
                y = self.NL(y)

            if self.res_layer:
                #y = y + x[..., :-reduce_lag]  # Add input back in (without oldest lag)
                y = y + x  # same shape [1,C,B,W,H] or [1,C,B,W] 

            # Rotate back for regular processing

            if self._ei_mask is not None:
                if self.is1D:
                    y = y * self._ei_mask[None, :, None, None]
                else:
                    y = y * self._ei_mask[None, :, None, None, None]

            if self.output_norm is not None:
                y = self.output_norm[iter](y)

            if (self.output_config == 'full') | (iter == self.num_iter-1):
                if outputs is None:
                    outputs = y
                else:
                    outputs = torch.cat( (outputs, y), dim=1)
            # Rotate batch dimension back

            if self.is1D:
                x = x.permute(3,1,0,2) # [B,C,W,1]->[1,C,B,W]
            else:
                x = y.permute(4,1,0,2,3) # [1,C,B,W,H]

            x = y
        # end of iteration loop

        return torch.reshape(outputs, (-1, self.num_outputs))
    # END IterSTlayer.forward


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
        self.activity_regularization = self.activity_reg.regularize(y)

        return y
    # END Tlayer.forward 

    def plot_filters( self, cmaps='gray', num_cols=8, row_height=2, time_reverse=False):
        # Overload plot_filters to automatically time_reverse
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


    