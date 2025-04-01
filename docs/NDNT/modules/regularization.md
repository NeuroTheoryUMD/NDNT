Module NDNT.modules.regularization
==================================

Classes
-------

`ActivityReg(reg_type=None, reg_val=None, input_dims=None, num_dims=0, **kwargs)`
:   Regularization to penalize activity separably for each dimension
    Note that the penalty needs to be computed elsewhere and just stored here
    
    Constructor for LocalityReg class

    ### Ancestors (in MRO)

    * NDNT.modules.regularization.RegModule
    * torch.nn.modules.module.Module

    ### Methods

    `compute_activity_penalty(self, acts)`
    :   Computes activity penalty from activations -- called in layer forward

    `compute_reg_penalty(self, weights)`
    :   Compute regularization penalty for locality

`ConvReg(reg_type=None, reg_val=None, input_dims=None, bc_val=1, **kwargs)`
:   Regularization module for convolutional penalties
    
    Constructor for Reg_module class

    ### Ancestors (in MRO)

    * NDNT.modules.regularization.RegModule
    * torch.nn.modules.module.Module

    ### Methods

    `compute_reg_penalty(self, weights)`
    :   I'm separating the code for more complicated regularization penalties in the simplest possible way here, but
        this can be done more (or less) elaborately in the future. The challenge here is that the dimension of the weight
        vector (1-D, 2-D, 3-D) determines what sort of Laplacian matrix, and convolution, to do
        Note that all convolutions are implicitly 'valid' so no boundary conditions

`DiagonalReg(reg_type=None, reg_val=None, input_dims=None, **kwargs)`
:   Regularization module for diagonal penalties
    
    Constructor for Reg_module class

    ### Ancestors (in MRO)

    * NDNT.modules.regularization.RegModule
    * torch.nn.modules.module.Module

    ### Methods

    `build_reg_mats(self, reg_type)`
    :

    `compute_reg_penalty(self, weights)`
    :   Compute regularization loss

`InlineReg(reg_type=None, reg_val=None, input_dims=None, **kwargs)`
:   Regularization module for inline penalties
    
    Constructor for Reg_module class

    ### Ancestors (in MRO)

    * NDNT.modules.regularization.RegModule
    * torch.nn.modules.module.Module

    ### Methods

    `compute_reg_penalty(self, weights)`
    :   Calculate regularization penalty for various reg types

`LocalityReg(reg_type=None, reg_val=None, input_dims=None, num_dims=0, **kwargs)`
:   Regularization to penalize locality separably for each dimension
    
    Constructor for LocalityReg class

    ### Ancestors (in MRO)

    * NDNT.modules.regularization.RegModule
    * torch.nn.modules.module.Module

    ### Methods

    `build_reg_mats(self)`
    :

    `compute_reg_penalty(self, weights)`
    :   Compute regularization penalty for locality

`RegModule(reg_type=None, reg_val=None, input_dims=None, num_dims=0, unit_reg=False, folded_lags=False, pos_constraint=False, **kwargs)`
:   Base class for regularization modules 
    
    Constructor for Reg_module class

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * NDNT.modules.regularization.ActivityReg
    * NDNT.modules.regularization.ConvReg
    * NDNT.modules.regularization.DiagonalReg
    * NDNT.modules.regularization.InlineReg
    * NDNT.modules.regularization.LocalityReg
    * NDNT.modules.regularization.Tikhanov
    * NDNT.modules.regularization.TikhanovC

    ### Methods

    `forward(self, weights) ‑> Callable[..., Any]`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`Regularization(filter_dims=None, vals=None, num_outputs=None, normalize=False, pos_constraint=False, folded_lags=False, **kwargs)`
:   Class for handling layer-wise regularization. 
    
    This class stores all info for regularization, and sets up regularization modules for training, 
    and returns reg_penalty from layer when passed in weights. Note that boundary conditions for a 
    given regularization is specified by a regularization 'bc', and passing that a dictionary. By 
    default, boundary conditions are on for a given regularization if not explicitly turned off.
    
    Args:
        vals (dict): values for different types of regularization stored as
            floats
        vals_ph (dict): placeholders for different types of regularization to
            simplify the tf Graph when experimenting with different reg vals
        vals_var (dict): values for different types of regularization stored as
            (un-trainable) tf.Variables
        mats (dict): matrices for different types of regularization stored as
            tf constants
        penalties (dict): tf ops for evaluating different regularization 
            penalties
        input_dims (list): dimensions of layer input size; for constructing reg 
            matrices
        num_outputs (int): dimension of layer output size; for generating 
            target weights in norm2
    
    Constructor for Regularization class. This stores all info for regularization, and 
    sets up regularization modules for training, and returns reg_penalty from layer when passed in weights
    
    Args:
        input_dims (list of ints): dimension of input size (for building reg mats)
        vals (dict, optional): key-value pairs specifying value for each type of regularization 
        Note: to pass in boundary_condition information, use a dict in vals with the values corresponding to
        particular regularization, e,g., 'BCs':{'d2t':1, 'd2x':0} (1=on, 0=off)
        
    Raises:
        TypeError: If `input_dims` is not specified
        TypeError: If `num_outputs` is not specified
    
    Notes:
        I'm using my old regularization matrices, which is made for the following 3-dimensional 
        weights with dimensions ordered in [NX, NY, num_lags]. Currently, filter_dims is 4-d: 
        [num_filters, NX, NY, num_lags] so this will need to rearrage so num_filters gets folded into last 
        dimension so that it will work with d2t regularization, if reshape is necessary]

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `get_reg_class(reg_type=None)`
    :

    ### Instance variables

    `unit_reg`
    :

    ### Methods

    `build_reg_modules(self, device=None)`
    :   Prepares regularization modules in train based on current regularization values

    `compute_activity_regularization(self, layer_output)`
    :

    `compute_reg_loss(self, weights)`
    :

    `reg_copy(self)`
    :   Copy regularization to new structure

    `set_reg_val(self, reg_type, reg_val=None)`
    :   Set regularization value in self.vals dict. Secondarily, it will also determine whether unit-reg
        applies or not, based on whether any of the reg vals are lists or arrays.
        
        Args:
            reg_type (str): see `_allowed_reg_types` for options
            reg_val (float or array): value of regularization parameter, or list of values if unit_reg
            
        Returns:
            bool: True if `reg_type` has not been previously set
            
        Note: can also pass in a dictionary with boundary condition information addressed by reg_type
        
        Raises:
            ValueError: If `reg_type` is not a valid regularization type
            ValueError: If `reg_val` is less than 0.0

    `unit_reg_convert(self, unit_reg=True, num_outputs=None)`
    :   Can convert reg object to turn on or off unit_reg (default turns it on

`Tikhanov(reg_type=None, reg_val=None, input_dims=None, bc_val=0, **kwargs)`
:   Regularization module for Tikhanov regularization
    
    Constructor for Tikhanov class

    ### Ancestors (in MRO)

    * NDNT.modules.regularization.RegModule
    * torch.nn.modules.module.Module

    ### Methods

    `compute_reg_penalty(self, weights)`
    :   Calculate regularization penalty for Tikhanov reg types

`TikhanovC(reg_type=None, reg_val=None, input_dims=None, bc_val=0, pos_constraint=True, **kwargs)`
:   Regularization module for Tikhanov-collapse regularization -- where all but one
    dimension is marginalized over first.
    
    Constructor for Tikhanov-Collapse class: where dimensions are summed over before Tikhanov
    matrix is applied. In case where weights are not pos-constrained, be sure to pass in a 
    boundary condition of '1' and it will square weights before summing. Otherwise, you will
    get a lot of -1s in the output.

    ### Ancestors (in MRO)

    * NDNT.modules.regularization.RegModule
    * torch.nn.modules.module.Module

    ### Methods

    `compute_reg_penalty(self, weights)`
    :   Calculate regularization penalty for Tikhanov reg types