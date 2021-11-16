import torch
from .ndnlayer import NDNLayer
import numpy as np
class DivNormLayer(NDNLayer):
    """
    Divisive normalization implementation: not explicitly convolutional
    

    """ 

    def __init__(self,
            input_dims=None,
            num_filters=None,
            filter_dims=None,
            pos_constraint=True,
            **kwargs,
        ):

        assert (input_dims is not None) or (num_filters is not None), "DivNormLayer: Must specify either input_dims or num_filters"
        
        if not pos_constraint:
            pos_constraint = True

        if num_filters is None:            
            num_filters = input_dims[0]

        if input_dims is None:
            input_dims = [num_filters, 1, 1, 1] # assume non-convolutional

        # number of filters (and size of filters) is set by channel dims on the input
        
        filter_dims = [num_filters, 1, 1, 1]

        super().__init__(input_dims,
            num_filters,
            filter_dims=filter_dims,
            pos_constraint=pos_constraint, **kwargs)

        self.output_dims = self.input_dims
        self.num_outputs = int(np.prod(self.output_dims))

        # self.weight.data.fill_(1/self.num_outputs)
        # self.bias.data.fill_(0.5)

    def forward(self, x):
        # TODO: make if statement for 1-d or 2-d, 3-d indpendent of x, should be specified by a property

        # Pull weights and process given pos_constrain and normalization conditions
        w = self.preprocess_weights()

        # Nonlinearity (apply first)
        if self.NL is not None:
            x = self.NL(x)

        x = x.reshape([-1] + self.input_dims)

        # Linear processing to create divisive drive
        xdiv = torch.einsum('nc...,ck->nk...', x, w)

        if len(x.shape)==2:
            xdiv = xdiv + self.bias[None,:]
        elif len(x.shape)==3:
            xdiv = xdiv + self.bias[None,:,None] # is 1D convolutional
        elif len(x.shape)==4:
            xdiv = xdiv + self.bias[None,:,None,None] # is 2D convolutional
        elif len(x.shape)==5:
            xdiv = xdiv + self.bias[None,:,None,None,None] # is 2D convolutional
        else:
            raise NotImplementedError('DivNormLayer only supports 2D, 3D, and 4D tensors')
            
        # apply divisive drive
        x = x / xdiv.clamp_(0.001) # divide
        
        x = x.reshape((-1, self.num_outputs))
        return x