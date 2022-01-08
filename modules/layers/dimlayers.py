import torch
import torch.nn as nn
from .ndnlayer import NDNLayer
import numpy as np

class Dim0Layer(NDNLayer):
    """
    Filters that act solely on filter-dimension (dim-0)

    """ 

    def __init__(self,
            input_dims=None,
            num_filters=None,
            bias=False,
            **kwargs,
        ):

        assert input_dims is not None, "Dim0Layer: Must specify input_dims"
        assert input_dims[0] > 1, "Dim0Layer: Dim-0 of input must be non-trivial"
        assert num_filters is not None, "Dim0Layer: num_filters must be specified"
        
        # Put filter weights in time-lag dimension to allow regularization using d2t etc
        filter_dims = [1, 1, 1, input_dims[0]]

        super().__init__(input_dims,
            num_filters,
            filter_dims=filter_dims,
            bias=bias,
            **kwargs)

        self.output_dims = [num_filters] + input_dims[1:]
        self.num_outputs = np.prod(self.output_dims)

        self.num_other_dims = input_dims[1]*input_dims[2]*input_dims[3]

    # END Dim0Layer.__init__

    def forward(self, x):

        w = self.preprocess_weights()

        # Reshape input to expose dim0 and put last
        x = x.reshape( [-1, self.input_dims[0], self.num_other_dims] ).permute([0, 2, 1])

        # Simple linear processing of dim0
        x = torch.matmul(x, w).permute([0, 2, 1]).reshape([-1, self.num_outputs])

        # left over things from NDNLayer forward
        if self.norm_type == 2:
            x = x / self.weight_scale

        x = x + self.bias

        # Nonlinearity
        if self.NL is not None:
            x = self.NL(x)

        # Constrain output to be signed
        if self.ei_mask is not None:
            x = x * self.ei_mask

        return x 
    # END Dim0Layer.forward
    
    @staticmethod
    def dim_info( input_dims=None, num_filters=None, **kwargs):
        """
        This uses the methods in the init to determine the input_dims, output_dims, filter_dims, and actual size of
        the weight tensor (weight_shape), given inputs, and package in a dictionary. This should be overloaded with each
        child of NDNLayer if want to use -- but this is external to function of actual layer.
        """
        assert input_dims is not None, "NDNLayer: Must include input_dims."
        assert num_filters is not None, "NDNLayer: Must include num_filters."

        filter_dims = tuple([input_dims[0], 1, 1, 1])
        output_dims = tuple([num_filters] + input_dims[1:])
        num_outputs = np.prod(output_dims) 
        weight_shape = tuple([np.prod(filter_dims), num_filters])  # likewise
        
        dinfo = {
            'input_dims': tuple(input_dims), 'filter_dims': tuple(filter_dims), 
            'output_dims': output_dims, 'num_outputs': num_outputs,
            'weight_shape': weight_shape}

        return dinfo
    # END [static] Dim0Layer.dim_info

    @classmethod
    def layer_dict(cls):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- Values that are fixed (not settable) will be set to None or not included
        -- Values that are needed will be lists or strings with default values
        -- Required inputs will be set to empty lists
        """

        Ldict = super().layer_dict()
        # Added arguments -- none
        # subtracted arguments
        del Ldict['filter_dims']

        return Ldict
    # END [classmethod] Dim0Layer.layer_dict
