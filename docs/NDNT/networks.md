Module NDNT.networks
====================

Classes
-------

`FFnet_external(external_module_dict=None, external_module_name=None, input_dims_reshape=None, **kwargs)`
:   This is a 'shell' that lets an external network be plugged into the NDN. It establishes all the basics
    so that information requested to this network from other parts of the NDN will behave correctly.
    
    Args:
        external_module_dict (dict): A dictionary of external modules.
        external_module_name (str): The name of the external module.
        input_dims_reshape (list): A list of input dimensions to reshape.
        **kwargs: Additional keyword arguments.
    
    Raises:
        AssertionError: If the external module dictionary is invalid.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * NDNT.networks.FFnetwork
    * torch.nn.modules.module.Module

    ### Static methods

    `ffnet_dict(**kwargs)`
    :   Returns a dictionary of the external network.
        
        Args:
            **kwargs: Additional keyword arguments.
        
        Returns:
            ffnet_dict (dict): The dictionary of the external network.

    ### Methods

    `compute_reg_loss(self)`
    :   Computes the regularization loss.
        
        Args:
            None
        
        Returns:
            0

    `forward(self, inputs) ‑> Callable[..., Any]`
    :   Forward pass through the network.
        
        Args:
            inputs (list, torch.Tensor): The input to the network.
        
        Returns:
            y (torch.Tensor): The output of the network.
        
        Raises:
            ValueError: If no layers are defined.

    `list_params(self, layer_target=None)`
    :   Lists the parameters for the network.
        
        Args:
            layer_target (int, optional): The layer to list the parameters for. Defaults to None.
        
        Returns:
            None
        
        Raises:
            AssertionError: If the layer target is invalid.

    `set_params(self, layer_target=None, name=None, val=None)`
    :   Sets the parameters for the listed layer or for all layers.
        
        Args:
            layer_target (int, optional): The layer to set the parameters for. Defaults to None.
            name (str): The name of the parameter.
            val (bool): The value to set the parameter to.
        
        Returns:
            None
        
        Raises:
            AssertionError: If the layer target is invalid.

`FFnetwork(layer_list: list = None, ffnet_type: str = 'normal', xstim_n: str = 'stim', ffnet_n: list = None, input_dims_list: list = None, reg_list: list = None, scaffold_levels: list = None, **kwargs)`
:   Initializes an instance of the network.
    
    Args:
        layer_list (list, optional): A list of dictionaries representing the layers of the network. Defaults to None.
        ffnet_type (str, optional): The type of the feedforward network. Defaults to 'normal'.
        xstim_n (str, optional): The name of the stimulus input. Defaults to 'stim'.
        ffnet_n (list, optional): A list of feedforward networks. Defaults to None.
        input_dims_list (list, optional): A list of input dimensions for each layer. Defaults to None.
        reg_list (list, optional): A list of regularization parameters. Defaults to None.
        scaffold_levels (list, optional): A list of scaffold levels. Defaults to None.
        **kwargs: Additional keyword arguments.
    
    Raises:
        AssertionError: If layer_list is not provided.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.networks.FFnet_external
    * NDNT.networks.ReadoutNetwork
    * NDNT.networks.ScaffoldNetwork

    ### Static methods

    `ffnet_dict(layer_list=None, xstim_n='stim', ffnet_n=None, ffnet_type='normal', scaffold_levels=None, num_lags_out=1, **kwargs)`
    :   Returns a dictionary of the feedforward network.
        
        Args:
            layer_list (list): A list of dictionaries representing the layers of the network.
            xstim_n (str): The name of the stimulus input.
            ffnet_n (list): A list of feedforward networks.
            ffnet_type (str): The type of the feedforward network.
            scaffold_levels (list): A list of scaffold levels.
            num_lags_out (int): The number of lags out.
            **kwargs: Additional keyword arguments.
        
        Returns:
            ffnet_dict (dict): The dictionary of the feedforward network.

    ### Instance variables

    `num_outputs`
    :

    ### Methods

    `compute_reg_loss(self)`
    :   Computes the regularization loss by summing reg_loss across layers.
        
        Args:
            None
        
        Returns:
            rloss (torch.Tensor): The regularization loss.

    `determine_input_dims(self, input_dims_list, ffnet_type='normal')`
    :   Sets input_dims given network inputs. Can be overloaded depending on the network type. For this base class, there
        are two types of network input: external stimulus (xstim_n) or a list of internal (ffnet_in) networks:
            For external inputs, it just uses the passed-in input_dims
            For internal network inputs, it will concatenate inputs along the filter dimension, but MUST match other dims
        As currently designed, this can either external or internal, but not both
        
        This sets the following internal FFnetwork properties:
            self.input_dims
            self.input_dims_list
        and returns Boolean whether the passed in input dims are valid
        
        Args:
            input_dims_list (list): A list of input dimensions for each layer.
            ffnet_type (str): The type of the feedforward network.
        
        Returns:
            valid_input_dims (bool): Whether the passed in input dims are valid.

    `forward(self, inputs) ‑> Callable[..., Any]`
    :   Forward pass through the network.
        
        Args:
            inputs (list, torch.Tensor): The input to the network.
        
        Returns:
            x (torch.Tensor): The output of the network.

    `generate_info_string(self)`
    :

    `get_network_info(self, abbrev=False)`
    :   Prints out a description of the network structure.

    `get_weights(self, layer_target=0, **kwargs)`
    :   Passed down to layer call, with optional arguments conveyed.
        
        Args:
            layer_target (int): The layer to get the weights for.
            **kwargs: Additional keyword arguments.
        
        Returns:
            The weights for the specified layer.
        
        Raises:
            AssertionError: If the layer target is invalid.

    `info(self, ffnet_n=None, expand=False)`
    :   This outputs the network information in abbrev (default) or expanded format, including
        information from layers

    `list_parameters(self, layer_target=None)`
    :   Lists the (fittable) parameters of the network, calling through each layer
        
        Args:
            layer_target (int, optional): The layer to list the parameters for. Defaults to None.
        
        Returns:
            None
        
        Raises:
            AssertionError: If the layer target is invalid.

    `plot_filters(self, layer_target=0, **kwargs)`
    :   Plots the filters for the listed layer, passed down to layer's call
        
        Args:
            layer_target (int, optional): The layer to plot the filters for. Defaults to 0.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None

    `prepare_regularization(self, device=None)`
    :   Makes regularization modules with current requested values.
        This is done immediately before training, because it can change during training and tuning.
        
        Args:
            device (str, optional): The device to use. Defaults to None.

    `preprocess_input(self, inputs)`
    :   Preprocess inputs to the ffnetwork according to the network type. If there
        is only one batch passed in, it does not matter what network type. But mutiple
        inputs (in a list) either:
            'normal': concatenates
            'add': adds inputs together (not must be same size or broadcastable)
            'mult': multiplies x1*(1+x2)*(1+x3+...) with same size req as 'add' 
        Args:
            inputs (list, torch.Tensor): The input to the network.
        
        Returns:
            x (torch.Tensor): The preprocessed input.
        
        Raises:
            ValueError: If no layers are defined.

    `set_parameters(self, layer_target=None, name=None, val=None)`
    :   Sets the parameters as either fittable or not (depending on 'val' for the listed 
        layer, with the default being all layers.
        
        Args:
            layer_target (int, optional): The layer to set the parameters for. Defaults to None.
            name (str): The name of the parameter: default is all parameters.
            val (bool): Whether or not to fit (True) or not fit (False)
        
        Returns:
            None
        
        Raises:
            AssertionError: If the layer target is invalid.

    `set_reg_val(self, reg_type=None, reg_val=None, layer_target=None)`
    :   Set reg_values for listed layer or for all layers.
        
        Args:
            reg_type (str): The type of regularization to set.
            reg_val (float): The value to set the regularization to.
            layer_target (int, optional): The layer to set the regularization for. Defaults to None.
        
        Returns:
            None
        
        Raises:
            AssertionError: If the layer target is invalid.

`ReadoutNetwork(**kwargs)`
:   A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
    and covariance of that Gaussian are learned.
    
    This essentially used the constructor for Point1DGaussian, with dicationary input.
    Currently there is no extra code required at the network level. I think the constructor
    can be left off entirely, but leaving in in case want to add something.
    
    Args:
        in_shape (list, tuple): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]. Default: 0.1
        init_sigma (float): The standard deviation of the Gaussian with `init_sigma` when `gauss_type` is
            'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the
            covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        gauss_type (str): Which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
        shifter (dict): Parameters for a predictor of shfiting grid locations. Has to have a form like
                        {
                        'hidden_layers':1,
                        'hidden_features':20,
                        'final_tanh': False,
                        }
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * NDNT.networks.FFnetwork
    * torch.nn.modules.module.Module

    ### Static methods

    `ffnet_dict(ffnet_n=0, **kwargs)`
    :   Returns a dictionary of the readout network.
        
        Args:
            ffnet_n (int): The feedforward network.
        
        Returns:
            ffnet_dict (dict): The dictionary of the readout network.

    ### Methods

    `determine_input_dims(self, input_dims_list, **kwargs)`
    :   Sets input_dims given network inputs. Can be overloaded depending on the network type. For this base class, there
        are two types of network input: external stimulus (xstim_n) or a list of internal (ffnet_in) networks:
            For external inputs, it just uses the passed-in input_dims
            For internal network inputs, it will concatenate inputs along the filter dimension, but MUST match other dims
        As currently designed, this can either external or internal, but not both
        
        This sets the following internal FFnetwork properties:
            self.input_dims
            self.input_dims_list
        and returns Boolean whether the passed in input dims are valid
        
        Args:
            input_dims_list (list): A list of input dimensions for each layer.
            **kwargs: Additional keyword arguments.
        
        Returns:
            valid_input_dims (bool): Whether the passed in input dims are valid.

    `forward(self, inputs) ‑> Callable[..., Any]`
    :   Network inputs correspond to output of conv layer, and (if it exists), a shifter.
        
        Args:
            inputs (list, torch.Tensor): The input to the network.
        
        Returns:
            y (torch.Tensor): The output of the network.

    `get_readout_locations(self)`
    :   Returns the positions in the readout layer (within this network)
        
        Args:
            None
        
        Returns:
            The readout locations

    `set_readout_locations(self, locs)`
    :   Sets the readout locations
        
        Args:
            locs: the readout locations
        
        Returns:
            None

`ScaffoldNetwork(scaffold_levels=None, num_lags_out=1, **kwargs)`
:   Concatenates output of all layers together in filter dimension, preserving spatial dims.
    
    This essentially used the constructor for Point1DGaussian, with dicationary input.
    Currently there is no extra code required at the network level. I think the constructor
    can be left off entirely, but leaving in in case want to add something.
    
    Args:
        scaffold_levels (list): A list of scaffold levels.
        num_lags_out (int): The number of lags out.
    
    Raises:
        AssertionError: If the scaffold levels are invalid.

    ### Ancestors (in MRO)

    * NDNT.networks.FFnetwork
    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.networks.ScaffoldNetwork3D

    ### Static methods

    `ffnet_dict(scaffold_levels=None, num_lags_out=1, **kwargs)`
    :   Returns a dictionary of the scaffold network.
        
        Args:
            scaffold_levels (list): A list of scaffold levels.
            num_lags_out (int): The number of lags out.
            **kwargs: Additional keyword arguments.
        
        Returns:
            ffnet_dict (dict): The dictionary of the scaffold network.
        
        Raises:
            AssertionError: If the scaffold levels are invalid.

    ### Methods

    `forward(self, inputs) ‑> Callable[..., Any]`
    :   Forward pass through the network: passes input sequentially through layers
        and concatenates the based on the self.scaffold_levels argument. Note that if
        there are lags, it will either chomp to the last, or keep number specified
        by self.num_lags_out
        
        Args:
            inputs (list, torch.Tensor): The input to the network.
        
        Returns:
            x (torch.Tensor): The output of the network.
        
        Raises:
            ValueError: If no layers are defined.

    `generate_info_string(self)`
    :

`ScaffoldNetwork3D(layer_list=None, num_lags_out=None, **kwargs)`
:   Like scaffold network above, but preserves the third dimension so in order
    to have shaped filters designed to process in subsequent network components.
    
    Args:
        num_lags_out (int): The number of lags out.
    
    Raises:
        AssertionError: If the scaffold levels are invalid.
    
    Args:
        scaffold_levels (list): A list of scaffold levels.
        num_lags_out (int): The number of lags out.
    
    Raises:
        AssertionError: If the scaffold levels are invalid.

    ### Ancestors (in MRO)

    * NDNT.networks.ScaffoldNetwork
    * NDNT.networks.FFnetwork
    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.networks.ScaffoldNetwork3d

    ### Static methods

    `ffnet_dict(**kwargs)`
    :   Returns a dictionary of the scaffold network.
        
        Args:
            **kwargs: Additional keyword arguments.
        
        Returns:
            ffnet_dict (dict): The dictionary of the scaffold network.

    ### Methods

    `forward(self, inputs) ‑> Callable[..., Any]`
    :   Forward pass through the network.
        
        Args:
            inputs (list, torch.Tensor): The input to the network.
        
        Returns:
            x (torch.Tensor): The output of the network.
        
        Raises:
            ValueError: If no layers are defined.

`ScaffoldNetwork3d(layer_list=None, num_lags_out=None, **kwargs)`
:   Placeholder so old models are not lonely
    
    Args:
        scaffold_levels (list): A list of scaffold levels.
        num_lags_out (int): The number of lags out.
    
    Raises:
        AssertionError: If the scaffold levels are invalid.

    ### Ancestors (in MRO)

    * NDNT.networks.ScaffoldNetwork3D
    * NDNT.networks.ScaffoldNetwork
    * NDNT.networks.FFnetwork
    * torch.nn.modules.module.Module