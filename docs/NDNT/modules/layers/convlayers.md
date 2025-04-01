Module NDNT.modules.layers.convlayers
=====================================

Classes
-------

`ConvLayer(input_dims=None, num_filters=None, filter_dims=None, temporal_tent_spacing=None, output_norm=None, stride=None, dilation=1, padding='same', res_layer=False, window=None, **kwargs)`
:   Convolutional NDN Layer
    
    Args (required):
        input_dims: tuple or list of ints, (num_channels, height, width, lags)
        num_filters: number of output filters
        filter_dims: width of convolutional kernel (int or list of ints)
    Args (optional):
        padding: 'same','valid', or 'circular' (default 'same')
        weight_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias_init: str, 'uniform', 'normal', 'xavier', 'zeros', or None
        bias: bool, whether to include bias term
        NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
    
    Initialize the layer.
    
    Args:
        input_dims: list of 4 ints, dimensions of input
        num_filters: int, number of filters in layer
        NLtype: str, type of nonlinearity to use (see activations.py)
        norm: int, type of filter normalization to use (0=None, 1=filters are unit vectors, 2=maxnorm?)
        pos_constraint: bool, whether to constrain weights to be positive
        num_inh: int, number of inhibitory units (creates ei_mask and makes "inhibitory" units have negative output)
        bias: bool, whether to include bias term
        weight_init: str, type of weight initialization to use (see reset_parameters, default 'xavier_uniform')
        bias_init: str, type of bias initialization to use (see reset_parameters, default 'zeros')
        reg_vals: dict, regularization values to use (see regularizers.py)

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.modules.layers.bilayers.BiConvLayer1D
    * NDNT.modules.layers.bilayers.BinocLayer1D
    * NDNT.modules.layers.bilayers.BinocShiftLayerOld
    * NDNT.modules.layers.bilayers.ChannelConvLayer
    * NDNT.modules.layers.convlayers.TconvLayer
    * NDNT.modules.layers.orilayers.ConvLayer3D
    * NDNT.modules.layers.orilayers.HermiteOriConvLayer
    * NDNT.modules.layers.orilayers.OriConvLayer
    * NDNT.modules.layers.partiallayers.ConvLayerPartial
    * NDNT.modules.layers.reslayers.IterLayer

    ### Static methods

    `layer_dict(padding='same', filter_dims=None, res_layer=False, window=None, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

    ### Instance variables

    `padding`
    :

    ### Methods

    `batchnorm_clone(self, bn_orig)`
    :

    `batchnorm_convert(self)`
    :   Converts layer with batch_norm to have the same output without batch_norm. This involves
        adjusting the weights and adding offsets

`STconvLayer(input_dims=None, num_filters=None, output_norm=None, **kwargs)`
:   Spatio-temporal convolutional layer.
    STConv Layers overload the batch dimension and assume they are contiguous in time.
    
    Args:
        input_dims (list of ints): input dimensions [C, W, H, T]
            This is, of course, not the real input dimensions, because the batch dimension is assumed to be contiguous.
            The real input dimensions are [B, C, W, H, 1], T specifies the number of lags
        num_filters (int): number of filters
        filter_dims (list of ints): filter dimensions [C, w, h, T]
            w < W, h < H
        stride (int): stride of convolution    
    
    Initialize the layer.
    
    Args:
        input_dims: list of 4 ints, dimensions of input
        num_filters: int, number of filters in layer
        NLtype: str, type of nonlinearity to use (see activations.py)
        norm: int, type of filter normalization to use (0=None, 1=filters are unit vectors, 2=maxnorm?)
        pos_constraint: bool, whether to constrain weights to be positive
        num_inh: int, number of inhibitory units (creates ei_mask and makes "inhibitory" units have negative output)
        bias: bool, whether to include bias term
        weight_init: str, type of weight initialization to use (see reset_parameters, default 'xavier_uniform')
        bias_init: str, type of bias initialization to use (see reset_parameters, default 'zeros')
        reg_vals: dict, regularization values to use (see regularizers.py)

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.TconvLayer
    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.modules.layers.masklayers.MaskSTconvLayer
    * NDNT.modules.layers.pyrlayers.MaskSTconvLayer
    * NDNT.modules.layers.reslayers.IterSTlayer

`TconvLayer(input_dims=None, num_filters=None, conv_dims=None, filter_dims=None, padding='spatial', output_norm=None, **kwargs)`
:   Temporal convolutional layer.
    TConv does not integrate out the time dimension and instead treats it as a true convolutional dimension
    
    Args:
        input_dims (list of ints): input dimensions [C, W, H, T]
        num_filters (int): number of filters
        filter_dims (list of ints): filter dimensions [C, w, h, T]
            w < W, h < H
        stride (int): stride of convolution
    
    Initialize the layer.
    
    Args:
        input_dims: list of 4 ints, dimensions of input
        num_filters: int, number of filters in layer
        NLtype: str, type of nonlinearity to use (see activations.py)
        norm: int, type of filter normalization to use (0=None, 1=filters are unit vectors, 2=maxnorm?)
        pos_constraint: bool, whether to constrain weights to be positive
        num_inh: int, number of inhibitory units (creates ei_mask and makes "inhibitory" units have negative output)
        bias: bool, whether to include bias term
        weight_init: str, type of weight initialization to use (see reset_parameters, default 'xavier_uniform')
        bias_init: str, type of bias initialization to use (see reset_parameters, default 'zeros')
        reg_vals: dict, regularization values to use (see regularizers.py)

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.modules.layers.convlayers.STconvLayer
    * NDNT.modules.layers.reslayers.IterTlayer

    ### Instance variables

    `padding`
    :