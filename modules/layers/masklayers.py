import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

from .ndnlayer import NDNLayer
from .convlayers import STconvLayer
from torch.nn.parameter import Parameter


class MaskLayer(NDNLayer):
    """
    MaskLayer: Layer with a mask applied to the weights.
    """

    def __init__(self, input_dims=None,
                 num_filters=None,
                 #filter_dims=None,  # absorbed by kwargs if necessary
                 #mask=None,  # cant pass in as part of dictionary
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
            ## NOT CANT BE PART OF DICTIONARY mask: np.ndarray, mask to apply to the weights
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
            output_norm=output_norm, initialize_center=False,
            bias_initializer=bias_initializer, reg_vals=reg_vals,
            **kwargs)

        #assert mask is not None, "MASKLAYER: must include mask, dodo"
        self.register_buffer('mask', torch.ones( [np.prod(self.filter_dims), self.num_filters], dtype=torch.float32))
        #self.register_buffer('mask_is_set', torch.zeros(1, dtype=torch.int8))  # have to convert to save in state_dict
        #self.mask_is_set = False

        # Must be done after mask is filled in
        if initialize_center:
            self.initialize_gaussian_envelope()
    # END MaskLayer.__init__

    def set_mask( self, mask=None ):
        """
        Sets mask -- instead of plugging in by hand. Registers mask as being set, and checks dimensions.
        Can also use to rest to have trivial mask (all 1s)
        Mask will be numpy, leave mask blank if want to set to default mask (all ones)
        
        Args:
            mask: numpy array of size of filters (filter_dims x num_filters)
        """
        #self.mask_is_set = 1
        if mask is None:
            self.mask[:,:] = 1.0
        else:
            assert np.prod(mask.shape) == np.prod(self.filter_dims)*self.num_filters
            self.mask = torch.tensor( mask.reshape([-1, self.num_filters]), dtype=torch.float32)
    # END MaskLayer.set_mask()

    def preprocess_weights( self ):
        """
        Preprocess weights for MaskLayer.

        Returns:
            w: torch.Tensor, preprocessed weights
        """
        w = super().preprocess_weights()
        return w*self.mask
    # END MaskLayer.preprocess_weights()

    def _layer_abbrev( self ):
        return "normalM"

    #def forward( self, x):
    #    assert self.mask_is_set > 0, "ERROR: Must set mask before using MaskLayer"
    #    return super().forward(x)
    # END MaskLayer.forward()

    @classmethod
    def layer_dict(cls, **kwargs):
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
        #Ldict['mask'] = mask
        return Ldict
    # END MaskLayer.layer_dict()


class MaskSTconvLayer(STconvLayer):
    """
    MaskSTConvLayer: STconvLayer with a mask applied to the weights.
    """

    def __init__(self, input_dims=None,
                 num_filters=None,
                 #filter_dims=None,  # absorbed by kwargs if necessary
                 #mask=None,
                 initialize_center=False,
                 output_norm=None,
                 **kwargs,
                 ):
        """
        MaskSTconvLayer: STconvLayer with a mask applied to the weights.

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            mask: np.ndarray, mask to apply to the weights
            output_norm: str, 'batch', 'batchX', or None
            **kwargs: additional arguments to pass to STConvLayer
        """

        assert input_dims is not None, "MaskSTconvLayer: input_dims must be specified"
        assert num_filters is not None, "MaskSTconvLayer: num_filters must be specified"

        super().__init__(
            input_dims=input_dims, num_filters=num_filters,
            output_norm=output_norm, initialize_center=False,
            **kwargs)

        # Now make mask
        #assert mask is not None, "MaskSTConvLayer: must include mask!"
        self.register_buffer('mask', torch.ones( [np.prod(self.filter_dims), self.num_filters], dtype=torch.float32))
        #self.register_buffer('mask_is_set', torch.zeros(1, dtype=torch.int8))     

        if initialize_center:
            self.initialize_gaussian_envelope()
    # END MaskSTconvLayer.__init__

    def set_mask( self, mask=None ):
        """
        Sets mask -- instead of plugging in by hand. Registers mask as being set, and checks dimensions.
        Can also use to rest to have trivial mask (all 1s)
        Mask will be numpy, leave mask blank if want to set to default mask (all ones)
        
        Args:
            mask: numpy array of size of filters (filter_dims x num_filters)
        """
        #self.mask_is_set = torch.ones()
        if mask is None:
            self.mask[:,:] = 1.0
        else:
            assert np.prod(mask.shape) == np.prod(self.filter_dims)*self.num_filters
            self.mask = torch.tensor( mask.reshape([-1, self.num_filters]), dtype=torch.float32)
    # END MaskSTconvLayer.set_mask()

    def preprocess_weights( self ):
        """
        Preprocess weights for MaskLayer.

        Returns:
            w: torch.Tensor, preprocessed weights
        """
        w = super().preprocess_weights()
        return w*self.mask
    # END MaskSTconvLayer.preprocess_weights()

    # DONT CHECK TO SEE IF MASK SET -- will work either way
    #def forward( self, x):
    #    assert self.mask_is_set > 0, "ERROR: Must set mask before using MaskSTconvLayer"
    #    return super().forward(x)
    # END MaskSTconvLayer.forward()

    def _layer_abbrev( self ):
        return "STconvM"

    @classmethod
    def layer_dict(cls, **kwargs):
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
        Ldict['layer_type'] = 'maskSTClayer'
        #Ldict['mask'] = mask    
        return Ldict
    # END MaskSTconvLayer.layer_dict()