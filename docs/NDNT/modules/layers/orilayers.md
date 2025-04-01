Module NDNT.modules.layers.orilayers
====================================

Classes
-------

`ConvLayer3D(input_dims: list = None, filter_width: int = None, ori_filter_width: int = 1, num_filters: int = None, output_norm: int = None, **kwargs)`
:   3D convolutional layer.
    
    Initialize 3D convolutional layer.
    
    Args:
        input_dims: input dimensions
        filter_width: filter width
        ori_filter_width: orientation filter width
        num_filters: number of filters
        output_norm: output normalization

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(filter_width=None, ori_filter_width=1, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            filter_width: filter width
            ori_filter_width: orientation filter width

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the layer.
        
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, *input_dims)
        
        Returns:
            y: torch.Tensor, output tensor of shape (batch_size, *output_dims)

    `plot_filters(self, cmaps='gray', num_cols=8, row_height=2, time_reverse=False)`
    :   Plot the filters.
        
        Args:
            cmaps: color map
            num_cols: number of columns
            row_height: row height
            time_reverse: time reverse
        
        Returns:
            fig: figure
            axs: axes

`HermiteOriConvLayer(input_dims=None, num_filters=None, hermite_rank=None, filter_width=None, output_norm=None, filter_dims=None, angles=None, **kwargs)`
:   HermiteOriConv Layer: An OriConvLayer whose filters are expressed in Hermite basis functions. From Ecker et al (2019)
    
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

    ### Static methods

    `layer_dict(hermite_rank=None, filter_width=None, basis=None, angles=None, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        
        Args:
            num_angles: number of rotations 
            **kwargs: additional arguments to pass to NDNLayer
        
        Returns:
            Ldict: dict, dictionary of layer parameters

    ### Instance variables

    `padding`
    :

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the layer.
        
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, *input_dims)
        
        Returns:
            y: torch.Tensor, output tensor of shape (batch_size, *output_dims)

    `get_filters(self, num_inh=0)`
    :   num-inh can take into account previous layer inhibition weights.
        
        Args:
            to_reshape: bool, whether to reshape the weights to the original filter shape
            time_reverse: bool, whether to reverse the time dimension
            num_inh: int, number of inhibitory units
        
        Returns:
            ws: np.ndarray, weights of the layer, on the CPU

    `hermcgen(self, mu, nu)`
    :   Generate coefficients of 2D Hermite functions

    `hermite_2d(self, N, npts, xvalmax=None)`
    :   Generate 2D Hermite function basis
        
        Arguments:
        N           -- the maximum rank.
        npts        -- the number of points in x and y
        
        Keyword arguments:
        xvalmax     -- the maximum x and y value (default: 2.5 * sqrt(N))
        
        Returns:
        H           -- Basis set of size N*(N+1)/2 x npts x npts
        desc        -- List of descriptors specifying for each
                       basis function whether it is:
                            'z': rotationally symmetric
                            'r': real part of quadrature pair
                            'i': imaginary part of quadrature pair

`OriConvLayer(input_dims=None, num_filters=None, res_layer=False, filter_width=None, padding='valid', output_norm=None, angles=None, **kwargs)`
:   Orientation-Convolutional layer.
    
    Will detect if needs to expand to multiple orientations (first layer, original OriConv) or use group-convolutions
    
    2-d conv layer that creates and maintains a third convolutional dimension through grouping and weight sharing. In
    other words, a third convolutional dimension is passed in, but the filters here act on each element of that third
    dimension, and weight-share between different groups. 
    
    Initialize orientation layer.
    
    Args:
        input_dims: input dimensions
        num_filters: number of filters
        filter_width: filter spatial width -- the rest of filter dims is determined
        angles: angles for rotation (in degrees)

    ### Ancestors (in MRO)

    * NDNT.modules.layers.convlayers.ConvLayer
    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.modules.layers.partiallayers.OriConvLayerPartial

    ### Static methods

    `layer_dict(filter_width=None, angles=None, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        
        Args:
            angles: list of angles for rotation (in degrees)
        
        Returns:
            Ldict: dict, dictionary of layer parameters

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the layer.
        
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, *input_dims)
        
        Returns:
            y: torch.Tensor, output tensor of shape (batch_size, *output_dims)

`OriLayer(input_dims=None, num_filters=None, filter_dims=None, angles=None, **kwargs)`
:   Orientation layer.
    
    Initialize orientation layer.
    
    Args:
        input_dims: input dimensions
        num_filters: number of filters
        filter_dims: filter dimensions
        angles: angles for rotation (in degrees)

    ### Ancestors (in MRO)

    * NDNT.modules.layers.ndnlayer.NDNLayer
    * torch.nn.modules.module.Module

    ### Static methods

    `layer_dict(angles=None, **kwargs)`
    :   This outputs a dictionary of parameters that need to input into the layer to completely specify.
        
        Args:
            angles: list of angles for rotation (in degrees)
        
        Returns:
            Ldict: dict, dictionary of layer parameters

    ### Methods

    `forward(self, x) ‑> Callable[..., Any]`
    :   Forward pass through the layer.
        
        Args:
            x: torch.Tensor, input tensor of shape (batch_size, *input_dims)
        
        Returns:
            y: torch.Tensor, output tensor of shape (batch_size, *output_dims)

    `rotation_matrix_tensor(self, filter_dims, theta_list)`
    :   Create a rotation matrix tensor for each angle in theta_list.
        
        Args:
            filter_dims: filter dimensions
            theta_list: list of angles in degrees
        
        Returns:
            rotation_matrix_tensor: rotation matrix tensor