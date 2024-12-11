import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

from .ndnlayer import NDNLayer
from .convlayers import ConvLayer
from .orilayers import OriConvLayer
#from torch.nn.parameter import Parameter


class NDNLayerPartial(NDNLayer):
    """
    NDNLayerPartial: NDNLayer where only some of the weights are fit
    """

    def __init__(self, input_dims=None,
                 num_filters=None,
                 #filter_dims=None,
                 num_fixed_filters=0,
                 fixed_dims=None,
                 fixed_num_inh=0,
                 NLtype:str='lin',
                 norm_type:int=0,
                 pos_constraint=0,
                 num_inh:int=0,  # this should be total (fixed and non-fixed)
                 bias:bool=False,
                 weights_initializer:str='xavier_uniform',
                 output_norm=None,
                 initialize_center=False,
                 bias_initializer:str='zeros',
                 reg_vals:dict=None,
                 **kwargs,
                 ):

        """
        NDNLayerPartial: NDNLayer with only some of the weights in the layer fit, determined by
            fixed_dims OR num_fixed_filters

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            num_fixed_filters: number of filters to hold constant (default=0)
            fixed_dims: 4-d size of fixed dims, use -1 for not-touching dims. Should have 3 -1s
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
        
        input_dims_all = deepcopy(input_dims)
        
        if fixed_dims is None:
            dims_mod = False
            assert num_fixed_filters > 0, "Need to make something fixed"
        else:
            dims_mod = True
            a = np.where(fixed_dims > 0)[0]
            assert len(a) == 1, "fixed_dims aint cool"
            fixed_axis = a[0]
            num_fixed = fixed_dims[fixed_axis]
            # Modify fixed input dimensions for purposes of making weights
            input_dims[fixed_axis] += -num_fixed
            assert num_fixed_filters == 0, "Can't set dims and num_fixed_filters at the same time"
            num_fixed_filters = num_filters

        num_filters_tot = num_filters
        #num_inh_tot = num_inh
        if num_fixed_filters > 0:
            assert num_fixed_filters < num_filters_tot, "Too many fixed filters"
            num_filters += -num_fixed_filters
            num_inh += -fixed_num_inh

        super().__init__(
            input_dims=input_dims, num_filters=num_filters,
            #filter_dims=None,  # for now, not using so None is correct
            NLtype=NLtype, norm_type=norm_type,
            pos_constraint=pos_constraint, num_inh=0,
            bias=False, weights_initializer=weights_initializer,
            output_norm=output_norm, initialize_center=initialize_center,
            bias_initializer=bias_initializer, reg_vals=reg_vals,
            **kwargs)

        # Redo bias -- assume it will always be fit, but needs to be the full input size
        if bias:
            del self.bias
            from torch.nn.parameter import Parameter
            self.bias = Parameter(torch.Tensor(num_filters_tot))
        else:
            self.register_buffer('bias', torch.zeros(num_filters_tot))

        #assert mask is not None, "MASKLAYER: must include mask, dodo"
        self.dims_mod = dims_mod
        self.num_fixed_filters = num_fixed_filters
        self.num_filters = num_filters_tot
        self.output_dims[0] = num_filters_tot

        if dims_mod:
            self.input_dims = deepcopy(input_dims_all)
            self.fixed_dims = deepcopy(self.filter_dims) + [num_filters_tot]
            self.fitted_dims = deepcopy(self.fixed_dims)
            self.fixed_dims[fixed_axis] = fixed_dims[fixed_axis]
            self.fitted_dims[fixed_axis] = self.filter_dims[fixed_axis]-fixed_dims[fixed_axis]
            self.fixed_axis = fixed_axis
        else:
            self.fixed_dims = [np.prod(self.filter_dims)]

        # Set num_inhibition
        self.num_inh = num_inh  # this will be reduced already and make weights
        if fixed_num_inh > 0:
            inh_range = range( self.num_fixed_filters-self.fixed_num_inh, self.num_fixed_filters)
            if num_inh == 0:
                # Then need to make buffer since canceled out
                self.num_inh = fixed_num_inh  # this calls setter, but will put weight in wrong place)
                self._ei_mask[:] = 1
            self._ei_mask[inh_range] = -1

        self.register_buffer('fixed_weight', torch.zeros( self.fixed_dims + [self.num_fixed_filters], dtype=torch.float32))
    # END NDNLayerPartial.__init__

    def preprocess_weights( self ):
        """
        Preprocess weights: assemble from real weights + fixed and then NDNLayer Process weights clipped in.
        
        Returns:
            w: torch.Tensor, preprocessed weights
        """

        # Make weights through concatenation
        if self.dims_mod:
            # Need to reshape to combine, and then flatten
            w = torch.cat( (self.fixed_weight, self.weight.reshape(self.fitted_dims)), 
                          axis=4).reshape([-1, self.num_filters])
        else:
            w = torch.cat( (self.fixed_weight, self.weight), axis=1 )

        # Apply positive constraints
        if self.pos_constraint > 0:
            w = torch.square(w) # to promote continuous gradients around 0
            #w = torch.maximum(self.weight, self.minval)
            #w = self.weight.clamp(min=0)
        elif self.pos_constraint < 0:
            w = -torch.square(w)  # to promote continuous gradients around 0
 
        # Add normalization
        if self.norm_type == 1: # so far just standard filter-specific normalization
            w = F.normalize( w, dim=0 ) / self.weight_scale
        return w
    # END NDNLayerPartial.preprocess_weights()

    #def forward( self, x):
    #    assert self.mask_is_set > 0, "ERROR: Must set mask before using MaskLayer"
    #    return super().forward(x)
    # END NDNLayerPartial.forward()

    @classmethod
    def layer_dict(cls, fixed_dims=None, num_fixed_filters=0, **kwargs):
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
        Ldict['layer_type'] = 'partial'
        Ldict['num_fixed_filters'] = num_fixed_filters
        Ldict['fixed_dims'] = fixed_dims
        return Ldict
    # END NDNLayerPartial.layer_dict()


class ConvLayerPartial(ConvLayer):
    """
    ConvLayerPartial: NDNLayer with only some of the weights in the layer fit, determined by
        fixed_dims OR num_fixed_filters
    """

    def __init__(self, input_dims=None,
                 num_filters=None,
                 filter_width=None,  # Note this is a change from full filter_dims
                 num_fixed_filters=0,
                 fixed_dims=None,
                 bias:bool=False,
                 fixed_num_inh=0,
                 num_inh:int=0,
                 # Pass through in kwargs anything that doesnt affect additional processing
                 #NLtype:str='lin',
                 #norm_type:int=0,
                 #pos_constraint=0,
                 #weights_initializer:str='xavier_uniform',
                 #output_norm=None,
                 #initialize_center=False,
                 #bias_initializer:str='zeros',
                 #reg_vals:dict=None,
                 **kwargs,
                 ):

        """
        ConvLayerPartial: NDNLayer with only some of the weights in the layer fit, determined by
            fixed_dims OR num_fixed_filters

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            filter_dims: width of convolutional kernel (int or list of ints)
            num_fixed_filters: number of filters to hold constant (default=0)
            fixed_dims: 4-d size of fixed dims, use -1 for not-touching dims. Should have 3 -1s
            NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
            norm_type: int, normalization type
            pos_constraint: int, whether to enforce non-negative weights
            num_inh: int, number of inhibitory filters
            bias: bool, whether to include bias term
            weights_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            output_norm: str, 'batch', 'batchX', or None
            padding: 'same','valid', or 'circular' (default 'same')
            initialize_center: bool, whether to initialize the center
            bias_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            reg_vals: dict, regularization values
            **kwargs: additional arguments to pass to NDNLayer
        """

        assert filter_width is not None, "filter_width (not filter_dims) is correct argument here"
        # Process filter_dims -- make 4-d from filter_width
        if input_dims[2] == 1: # then its 1-d filter
            filter_dims = [input_dims[0], filter_width, 1, input_dims[3]]
        else:
            filter_dims = [input_dims[0], filter_width, filter_width, input_dims[3]]

        if fixed_dims is None:
            dims_mod = False
            assert num_fixed_filters > 0, "Need to make something fixed"
        else:
            dims_mod = True
            a = np.where(fixed_dims > 0)[0]
            assert len(a) == 1, "fixed_dims aint cool"
            fixed_axis = a[0]
            assert fixed_axis in [0, 3], "Can't use spatial for fixed_axis"
            num_fixed = fixed_dims[fixed_axis]
            filter_dims[fixed_axis] += -num_fixed
            assert num_fixed_filters == 0, "Can't set dims and num_fixed_filters at the same time"
            num_fixed_filters = num_filters

        num_filters_tot = num_filters
        num_inh_tot = num_inh
        if num_fixed_filters > 0:
            assert num_fixed_filters < num_filters_tot, "Too many fixed filters"
            num_filters += -num_fixed_filters
            num_inh += -fixed_num_inh

        super().__init__(
            input_dims=input_dims, num_filters=num_filters, bias=False, 
            filter_dims=filter_dims, num_inh=0,
            #NLtype=NLtype, norm_type=norm_type,
            #pos_constraint=pos_constraint, num_inh=num_inh,
            #weights_initializer=weights_initializer,
            #output_norm=output_norm, initialize_center=initialize_center,
            #bias_initializer=bias_initializer, reg_vals=reg_vals,
            **kwargs)

        # Redo bias -- assume it will always be fit, but needs to be the full input size
        if bias:
            from torch.nn.parameter import Parameter
            self.bias = Parameter(torch.Tensor(num_filters_tot, dtype=torch.float32))
        else:
            #self.register_buffer('bias', torch.zeros(num_filters_tot))  # should not register twice
            self.bias = torch.zeros(num_filters_tot, dtype=torch.float32)

        # Redo EI-mask to be larger size -- buffer already created
        self._ei_mask = torch.ones(num_filters_tot, dtype=torch.float32)
        self.num_inh = num_inh  # this will be reduced already and make weights
        if fixed_num_inh > 0:
            inh_range = range( self.num_fixed_filters-self.fixed_num_inh, self.num_fixed_filters)
            if num_inh == 0:
                # Then need to make buffer since canceled out
                self.num_inh = fixed_num_inh  # this calls setter, but will put weight in wrong place)
                self._ei_mask[:] = 1
            self._ei_mask[inh_range] = -1

        #assert mask is not None, "MASKLAYER: must include mask, dodo"
        self.dims_mod = dims_mod
        self.num_fixed_filters = num_fixed_filters
        self.num_filters = num_filters_tot
        self.output_dims[0] = num_filters_tot
        self.num_outputs = np.prod(self.output_dims)

        if dims_mod:
            self.fixed_dims = deepcopy(self.filter_dims) + [num_filters_tot]
            self.fitted_dims = deepcopy(self.fixed_dims)
            self.fixed_dims[fixed_axis] = fixed_dims[fixed_axis]
            self.fitted_dims[fixed_axis] = self.filter_dims[fixed_axis]-fixed_dims[fixed_axis]
            self.fixed_axis = fixed_axis
        else:
            self.fixed_dims = [np.prod(self.filter_dims)]
        #if num_fixed_filters > 0:
        #    self.register_buffer('fixed_filters', torch.ones( [np.prod(self.filter_dims), self.num_fixed_filters], dtype=torch.float32))

        self.register_buffer('fixed_weight', torch.zeros( self.fixed_dims + [self.num_fixed_filters], dtype=torch.float32))
    # END ConvLayerPartial.__init__

    def preprocess_weights( self ):
        """
        Preprocess weights: assemble from real weights + fixed and then NDNLayer Process weights clipped in.
        
        Returns:
            w: torch.Tensor, preprocessed weights
        """

        # Make weights through concatenation
        if self.dims_mod:
            # Need to reshape to combine, and then flatten
            w = torch.cat( (self.fixed_weight, self.weight.reshape(self.fitted_dims)), 
                          axis=4).reshape([-1, self.num_filters])
        else:
            w = torch.cat( (self.fixed_weight, self.weight), axis=1 )

        # Once joined, can call regular pre-process weights
        return super().preprocess_weights(mod_weight=w)
    # END ConvLayerPartial.preprocess_weights()

    #def forward( self, x):
    #    assert self.mask_is_set > 0, "ERROR: Must set mask before using MaskLayer"
    #    return super().forward(x)
    # END ConvLayerPartial.forward()

    @classmethod
    def layer_dict(cls, fixed_dims=None, num_fixed_filters=0, fixed_num_inh=0,
                   filter_width=None, **kwargs):
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
        Ldict['layer_type'] = 'partial_conv'
        # Added arguments
        Ldict['num_fixed_filters'] = num_fixed_filters
        Ldict['fixed_dims'] = fixed_dims
        # Swap filter-width for filter_dims
        Ldict['filter_width'] = filter_width
        Ldict['fixed_num_inh'] = fixed_num_inh
        del Ldict['filter_dims']
        return Ldict
    # END ConvLayerPartial.layer_dict()


class OriConvLayerPartial(OriConvLayer):
    """
    ConvLayerPartial: NDNLayer with only some of the weights in the layer fit, determined by
        fixed_dims OR num_fixed_filters
    """

    def __init__(self, input_dims=None,
                 num_filters=None,
                 filter_width=None,  # Note this is a change from full filter_dims
                 num_fixed_filters=0,
                 fixed_dims=None,
                 bias:bool=False,
                 fixed_num_inh=0,
                 num_inh:int=0,
                 angles=None,
                 # Pass through in kwargs anything that doesnt affect additional processing
                 #NLtype:str='lin',
                 #norm_type:int=0,
                 #pos_constraint=0,
                 #weights_initializer:str='xavier_uniform',
                 #output_norm=None,
                 #initialize_center=False,
                 #bias_initializer:str='zeros',
                 #reg_vals:dict=None,
                 **kwargs,
                 ):
        """
        OriConvLayerPartial: OriConvLayer with only some of the weights in the layer fit, determined by
            fixed_dims OR num_fixed_filters

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            filter_dims: width of convolutional kernel (int or list of ints)
            num_fixed_filters: number of filters to hold constant (default=0)
            fixed_dims: 4-d size of fixed dims, use -1 for not-touching dims. Should have 3 -1s
            NLtype: str, 'lin', 'relu', 'tanh', 'sigmoid', 'elu', 'none'
            norm_type: int, normalization type
            pos_constraint: int, whether to enforce non-negative weights
            num_inh: int, number of inhibitory filters
            bias: bool, whether to include bias term
            weights_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            output_norm: str, 'batch', 'batchX', or None
            padding: 'same','valid', or 'circular' (default 'same')
            initialize_center: bool, whether to initialize the center
            bias_initializer: str, 'uniform', 'normal', 'xavier', 'zeros', or None
            reg_vals: dict, regularization values
            **kwargs: additional arguments to pass to NDNLayer
        """
        assert angles is not None, "need to put in the angles"
        assert filter_width is not None, "filter_width (not filter_dims) is correct argument here"
        # Process filter_dims -- make 4-d from filter_width
        if input_dims[2] == 1: # then its 1-d filter
            filter_dims = [input_dims[0], filter_width, 1, input_dims[3]]
        else:
            filter_dims = [input_dims[0], filter_width, filter_width, input_dims[3]]

        if fixed_dims is None:
            dims_mod = False
            assert num_fixed_filters > 0, "Need to make something fixed"
        else:
            dims_mod = True
            a = np.where(fixed_dims > 0)[0]
            assert len(a) == 1, "fixed_dims aint cool"
            fixed_axis = a[0]
            assert fixed_axis in [0, 3], "Can't use spatial for fixed_axis"
            num_fixed = fixed_dims[fixed_axis]
            filter_dims[fixed_axis] += -num_fixed
            assert num_fixed_filters == 0, "Can't set dims and num_fixed_filters at the same time"
            num_fixed_filters = num_filters

        num_filters_tot = num_filters
        num_inh_tot = num_inh
        if num_fixed_filters > 0:
            assert num_fixed_filters < num_filters_tot, "Too many fixed filters"
            num_filters += -num_fixed_filters
            num_inh += -fixed_num_inh

        super().__init__(
            input_dims=input_dims, num_filters=num_filters, bias=False, 
            filter_width=filter_width, num_inh=0, angles=angles,
            **kwargs)

        # Redo bias -- assume it will always be fit, but needs to be the full input size
        if bias:
            from torch.nn.parameter import Parameter
            self.bias = Parameter(torch.Tensor(num_filters_tot, dtype=torch.float32))
        else:
            #self.register_buffer('bias', torch.zeros(num_filters_tot))  # should not register twice
            self.bias = torch.zeros(num_filters_tot, dtype=torch.float32)

        # Redo EI-mask to be larger size -- buffer already created
        self._ei_mask = torch.ones(num_filters_tot, dtype=torch.float32)
        self.num_inh = num_inh  # this will be reduced already and make weights
        if fixed_num_inh > 0:
            inh_range = range( self.num_fixed_filters-self.fixed_num_inh, self.num_fixed_filters)
            if num_inh == 0:
                # Then need to make buffer since canceled out
                self.num_inh = fixed_num_inh  # this calls setter, but will put weight in wrong place)
                self._ei_mask[:] = 1
            self._ei_mask[inh_range] = -1

        self.dims_mod = dims_mod
        self.num_fixed_filters = num_fixed_filters
        self.num_filters = num_filters_tot
        self.output_dims[0] = num_filters_tot
        self.num_outputs = np.prod(self.output_dims)

        if dims_mod:
            self.fixed_dims = deepcopy(self.filter_dims) + [num_filters_tot]
            self.fitted_dims = deepcopy(self.fixed_dims)
            self.fixed_dims[fixed_axis] = fixed_dims[fixed_axis]
            self.fitted_dims[fixed_axis] = self.filter_dims[fixed_axis]-fixed_dims[fixed_axis]
            self.fixed_axis = fixed_axis
        else:
            self.fixed_dims = [np.prod(self.filter_dims)]
        #if num_fixed_filters > 0:
        #    self.register_buffer('fixed_filters', torch.ones( [np.prod(self.filter_dims), self.num_fixed_filters], dtype=torch.float32))

        self.register_buffer('fixed_weight', torch.zeros( self.fixed_dims + [self.num_fixed_filters], dtype=torch.float32))
    # END OriConvLayerPartial.__init__

    def preprocess_weights( self ):
        """
        Preprocess weights: assemble from real weights + fixed and then NDNLayer Process weights clipped in.
        
        Returns:
            w: torch.Tensor, preprocessed weights
        """

        # Make weights through concatenation
        if self.dims_mod:
            # Need to reshape to combine, and then flatten
            w = torch.cat( (self.fixed_weight, self.weight.reshape(self.fitted_dims)), 
                          axis=4).reshape([-1, self.num_filters])
        else:
            w = torch.cat( (self.fixed_weight, self.weight), axis=1 )

        # Once joined, can call regular pre-process weights
        return super().preprocess_weights(mod_weight=w)
    # END ConvLayerPartial.preprocess_weights()

    # END OriConvLayerPartial.forward()

    @classmethod
    def layer_dict(cls, fixed_dims=None, num_fixed_filters=0, fixed_num_inh=0, #filter_width=None, 
                   **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults

        Args:
            FIX
            **kwargs: additional arguments to pass to NDNLayer

        Returns:
            Ldict: dict, dictionary of layer parameters
        """

        Ldict = super().layer_dict(**kwargs)
        Ldict['layer_type'] = 'partial_oriconv'
        # Added arguments
        Ldict['num_fixed_filters'] = num_fixed_filters
        Ldict['fixed_dims'] = fixed_dims
        # Swap filter-width for filter_dims
        #Ldict['filter_width'] = filter_width
        Ldict['fixed_num_inh'] = fixed_num_inh
        #del Ldict['filter_dims']
        return Ldict
    # END OriConvLayerPartial.layer_dict()