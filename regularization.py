### regularization.py: managing regularization
import torch
from torch import nn

from copy import deepcopy

class Regularization(nn.Module):
    """Class for handling layer-wise regularization
    
    Attributes:
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

    """

    _allowed_reg_types = ['l1', 'l2', 'norm2', 'norm2_space', 'norm2_filt',
                          'd2t', 'd2x', 'd2xt', 'local', 'glocal', 'center',
                          'max', 'max_filt', 'max_space', 'orth']

    def __init__(self, filter_dims=None, num_filters=None, vals=None):
        """Constructor for Regularization class. This stores all info for regularization, and 
        sets up regularization modules for training, and returns reg_penalty from layer when passed in weights
        
        Args:
            input_dims (list of ints): dimension of input size (for building reg mats)
            vals (dict, optional): key-value pairs specifying value for each type of regularization 

        Raises:
            TypeError: If `input_dims` is not specified
            TypeError: If `num_outputs` is not specified
        
        Note that I'm using my old regularization matrices, which is made for the following 3-dimensional
        weights with dimensions ordered in [NX, NY, num_lags]. Currently, filter_dims is 4-d: 
        [num_filters, NX, NY, num_lags] so this will need to rearrage so num_filters gets folded into last 
        dimension so that it will work with d2t regularization, if reshape is necessary]
        """

        super(Regularization, self).__init__()

        # check input
        assert filter_dims is not None, "Must specify `input_dims`"
        self.input_dims_original = filter_dims
        # the "input_dims" are ordered differently for matrix implementations,
        # so this is a temporary fix to be able to use old functions
        self.input_dims = [
            filter_dims[0]*filter_dims[3], # non-spatial
            filter_dims[1], filter_dims[2]]  # spatial 
 
        # will this need to combine first and last input dims? (or just ignore first)
        if filter_dims[0] == 1:
            self.need_reshape = False
        else:
            self.need_reshape = True

        self.vals = {}
        self.reg_modules = nn.ModuleList() 

        # read user input
        if vals is not None:
            #for reg_type, reg_val in vals.iteritems():  # python3 mod
            for reg_type, reg_val in vals.items():
                if reg_val is not None:
                    self.set_reg_val(reg_type, reg_val)
    # END Regularization.__init__

    #def __repr__(self):
    #    s = super().__repr__()
        # Add other details for printing out if we want

    def set_reg_val(self, reg_type, reg_val=None):
        """Set regularization value in self.vals dict (doesn't affect a tf 
        Graph until a session is run and `assign_reg_vals` is called)
        
        Args:
            reg_type (str): see `_allowed_reg_types` for options
            reg_val (float): value of regularization parameter
            
        Returns:
            bool: True if `reg_type` has not been previously set
            
        Raises:
            ValueError: If `reg_type` is not a valid regularization type
            ValueError: If `reg_val` is less than 0.0
            
        """

        # check inputs
        if reg_type not in self._allowed_reg_types:
            raise ValueError('Invalid regularization type ''%s''' % reg_type)

        if reg_val is None:  # then eliminate reg_type
            if reg_type in self.vals:
                del self.vals[reg_type]
        else:  # add or modify reg_val
            if reg_val < 0.0:
                raise ValueError('`reg_val` must be greater than or equal to zero')

            self.vals[reg_type] = reg_val

    # END Regularization.set_reg_val

    def build_reg_modules(self):
        """Prepares regularization modules in train based on current regularization values"""
        self.reg_modules = nn.ModuleList()  # this clears old modules (better way?)
        #self.reg_modules.clear()  # no 'clear' exists for module lists
        for kk, vv in self.vals.items():
            # Awkwardly, old regularization (inherited uses 3-d input_dims, but new reg uses the 4-d)
            if kk in ['d2xt', 'd2x', 'd2t']:
                self.reg_modules.append( RegModule(reg_type=kk, reg_val=vv, input_dims=self.input_dims_original) )
            else: # old-way: 3-d input_dims
                self.reg_modules.append( RegModule(reg_type=kk, reg_val=vv, input_dims=self.input_dims) )

    def compute_reg_loss(self, weights):  # this could also be a forward?
        """Define regularization loss. Will reshape weights as needed"""
        
        if len(self.reg_modules) == 0:
            return 0.0

        if self.need_reshape:
            wsize = weights.size()
            num_filters = wsize[-1]
            w = torch.reshape(
                    torch.reshape(
                        weights, 
                        self.input_dims_original + [wsize[-1]]
                        ).permute(1,2,0,3,4), 
                    wsize)
        else:
            w = weights

        rloss = 0
        for regmod in self.reg_modules:
            rloss += regmod( weights )
        return rloss
    # END Regularization.define_reg_loss

    def reg_copy(self):
        """Copy regularization to new structure"""

        reg_target = Regularization(input_dims=self.input_dims)
        reg_target.vals = deepcopy(self.val)

        return reg_target
    # END Regularization.reg_copy

class RegModule(nn.Module):

    def __init__(self, reg_type=None, reg_val=None, input_dims=None):
        """Constructor for Reg_module class"""

        assert reg_type is not None, 'Need reg_type.'
        assert reg_val is not None, 'Need reg_val'
        assert input_dims is not None, 'Need input dims'

        super(RegModule, self).__init__()

        self.reg_type = reg_type
        self.register_buffer( 'val', torch.tensor(reg_val))
        self.input_dims = input_dims
        self.num_dims = 0 # this is the relevant number of dimensions for some filters -- will be set within functions

        # Make appropriate reg_matrix as buffer (non-fit parameter)
        reg_tensor = self._build_reg_mats( reg_type)
        if reg_tensor is None:  # some reg dont need rmat 
            self.rmat = None
        else:
            self.register_buffer( 'rmat', reg_tensor)

    # END RegModule.__init__

    #def __repr__(self):
    #    s = super().__repr__()
        # Add other details for printing out if we want

    def _build_reg_mats(self, reg_type):
        """Build regularization matrices in default tf Graph

        Args:
            reg_type (str): see `_allowed_reg_types` for options
        """
        import create_reg_matrices as get_rmats

        if (reg_type == 'd2t') or (reg_type == 'd2x') or (reg_type == 'd2xt'):
            #reg_mat = get_rmats.create_tikhonov_matrix(self.input_dims, reg_type)
            #name = reg_type + '_laplacian'
            reg_mat = self.make_laplacian(reg_type)
        elif (reg_type == 'max') or (reg_type == 'max_filt') or (reg_type == 'max_space'):
            reg_mat = get_rmats.create_maxpenalty_matrix(self.input_dims, reg_type)
            #name = reg_type + '_reg'
        elif reg_type == 'center':
            reg_mat = get_rmats.create_maxpenalty_matrix(self.input_dims, reg_type)
            #name = reg_type + '_reg'
        elif reg_type == 'local':
            reg_mat = get_rmats.create_localpenalty_matrix(
                self.input_dims, separable=False)
            #name = reg_type + '_reg'
        elif reg_type == 'glocal':
            reg_mat = get_rmats.create_localpenalty_matrix(
                self.input_dims, separable=False, spatial_global=True)
            #name = reg_type + '_reg'
        else:
            reg_mat = None

        if reg_mat is None:
            return None
        else:
            return torch.Tensor(reg_mat)
    # END RegModule._build_reg_mats

    def forward(self, weights):
        """Calculate regularization penalty for various reg types"""

        if self.reg_type == 'l1':
            reg_pen = torch.sum(torch.abs(weights))

        elif self.reg_type == 'l2':
            reg_pen = torch.sum(torch.square(weights))

        elif self.reg_type in ['d2t', 'd2x', 'd2xt']:
            #reg_pen = torch.sum( torch.square( torch.matmul(self.rmat, weights) ) )
            reg_pen = self.d2xt( weights )

        elif self.reg_type == 'norm2':  # [custom] convex (I think) soft-normalization regularization
            reg_pen = torch.square( torch.mean(torch.square(weights))-1 )

        elif self.reg_type in ['max', 'max_filt', 'max_space', 'local', 'glocal', 'center']:  # [custom]
            # my old implementation didnt square weights before passing into center. should it? I think....
            w2 = torch.square(weights)
            reg_pen = torch.trace(
                torch.matmul( w2.T, torch.matmul(self.rmat, w2) ))
        # ORTH MORE COMPLICATED: needs another buffer?
        #elif self.reg_type == 'orth':  # [custom]
        #    diagonal = np.ones(weights.shape[1], dtype='float32')
        #    # sum( (W^TW - I).^2)
        #    reg_pen = tf.multiply(self.vals_var['orth'],
        #        tf.reduce_sum(tf.square(tf.math.subtract(tf.matmul(tf.transpose(weights), weights),tf.linalg.diag(diagonal)))))
        else:
            reg_pen = 0.0

        return self.val*reg_pen
    # END RegModule.forward

    def make_laplacian( self, reg_type ):
        """This will make the Laplacian of the right dimensionality depending on d2xt, d2t, d2x"""

        import numpy as np

        # Determine number of dimensions for laplacian matrix
        dim_mask = np.array(self.input_dims[1:]) > 1  # filter dimension will be ignored
        if reg_type == 'd2t':
            dim_mask[:2] = False  # zeros out spatial dimensions
        elif reg_type == 'd2x':
            dim_mask[2] = False # zeros out temporal dimensions
        self.num_dims = np.sum(dim_mask)
        
        # note all the extra brackets are so the first two dims [out_chan, in_chan] are 1,1
        if self.num_dims == 1:
            rmat = np.array([[[-1, 2, -1]]])
        elif self.num_dims == 2:
            rmat = np.array([[[[0,-1,0],[-1, 4, -1], [0,-1,0]]]])
        elif self.num_dims == 3:
            rmat = np.array(
                [[[[[0, 0, 0],[0, -1, 0], [0, 0, 0]],
                [[0, -1, 0],[-1, 6, -1], [0, -1, 0]],
                [[0, 0, 0],[0, -1, 0], [0, 0, 0]]]]])
        else:
            rmat = np.array([1])
            print("Warning: %s regularization does not have the necessary filter dimensions.")
        return rmat

    def d2xt(self, weights):
        """I'm separating the code for more complicated regularization penalties in the simplest possible way here, but
        this can be done more (or less) elaborately in the future. The challenge here is that the dimension of the weight
        vector (1-D, 2-D, 3-D) determines what sort of Laplacian matrix, and convolution, to do
        Note that all convolutions are implicitly 'valid' so no boundary conditions"""
        from torch.nn import functional as F
        
        weight_dims = self.input_dims + [weights.shape[1]]
        w = weights.view(weight_dims)
        # puts in [C, W, H, T, num_filters]: to reorder depending on reg type
        # default reg_dims
        if self.reg_type == 'd2t':
            w = w.permute(4,0,1,2,3) # needs temporal dimension last so only convolved
            reg_dims = [weight_dims[3]]
        elif self.reg_type == 'd2xt':
            w = w.permute(4,0,1,2,3) 
            # Reg-dims will depend on whether space is one- or two-dimensional
            if weight_dims[2] > 1:
                reg_dims = [weight_dims[1], weight_dims[2], weight_dims[3]]
            else:
                reg_dims = [weight_dims[1], weight_dims[3]]
        else:  # then d2x
            w = w.permute(4,0,3,1,2)  # rotate temporal dimensions next to filter dims
            reg_dims = [weight_dims[1], weight_dims[2]]

        if self.num_dims == 1:
            # prepare for 1-d convolve
            rpen = torch.sum(F.conv1d( 
                w.reshape( [-1, 1] + reg_dims[:1] ),  # [batch_dim, all non-conv dims, conv_dim] reshape needed, not view
                self.rmat ).pow(2))
        elif self.num_dims == 2:
            rpen = torch.sum(F.conv2d( 
                w.view( [-1, 1] + reg_dims[:2] ), 
                self.rmat, padding=(0,0) ).pow(2))
        elif self.num_dims == 3:
            rpen = torch.sum(F.conv3d( 
                w.view( [-1,1] + reg_dims),
                self.rmat ).pow(2))
        return rpen
