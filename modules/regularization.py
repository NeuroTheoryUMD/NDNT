### regularization.py: managing regularization
import torch
from torch import nn
from torch.nn import functional as F

from copy import deepcopy
import numpy as np

class Regularization(nn.Module):
    """
    Class for handling layer-wise regularization. 
    
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
    """

    def __init__(self, filter_dims=None, vals=None, num_outputs=None, 
                 normalize=False, pos_constraint=False, folded_lags=False, **kwargs):
        """
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
        """

        super(Regularization, self).__init__()

        # check input
        assert filter_dims is not None, "Must specify `input_dims`"
        self.input_dims = filter_dims

        self.vals = {}
        self.reg_modules = nn.ModuleList() 
        self.normalize = normalize

        self.folded_lags = folded_lags
        self.num_outputs = num_outputs
        self.pos_constraint=pos_constraint
        self.boundary_conditions = None
        self.activity_regmodule = None

        self.register_buffer( '_unit_reg', torch.zeros(1, dtype=torch.int8) )
        self.unit_reg = False

        # read user input
        if vals is not None:
            for reg_type, reg_val in vals.items():
                if reg_val is not None:
                    self.set_reg_val(reg_type, reg_val)
    # END Regularization.__init__

    def set_reg_val(self, reg_type, reg_val=None):
        """
        Set regularization value in self.vals dict. Secondarily, it will also determine whether unit-reg
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
        """

        # check inputs
        if reg_type not in self.get_reg_class():
            if reg_type in ['bc', 'bcs', 'BC', 'BCs']:
                assert isinstance(reg_val, dict), "Regularization: boundary conditions must be a dictionary."
                self.boundary_conditions = deepcopy(reg_val)  # set potential boundary conditions
                return
            else:
                print("Regularization: %s not recognized as a valid regularization type" % reg_type)
                # skip this regularization type
                return

        if reg_val is None:  # then eliminate reg_type
            if reg_type in self.vals:
                del self.vals[reg_type]
        else:  # add or modify reg_val
            if isinstance(reg_val, np.ndarray) | isinstance(reg_val, list):  # then must be unit_reg
                assert np.min(reg_val) >= 0, "UNIT_REG: all reg_values must be non-negative"
                if not self.unit_reg:
                    # then intent to have unit_conver set
                    self.unit_reg = True
                    print("%s: implementing unit_reg"%reg_type)
            else:
                if reg_val < 0.0:
                    raise ValueError('`reg_val` must be greater than or equal to zero')

            if self.unit_reg:
                assert self.num_outputs is not None, "UNITREG initialization problems"
                if isinstance(reg_val, list) or isinstance(reg_val, np.ndarray):
                    assert len(reg_val) == self.num_outputs, "UNITREG: unmatched size list of reg values"
                    self.vals[reg_type] = np.array(deepcopy(reg_val), dtype=np.float32)
                else:
                    self.vals[reg_type] = np.ones(self.num_outputs, dtype=np.float32)*reg_val
            else:  # Normal reg val setting
                self.vals[reg_type] = reg_val
    # END Regularization.set_reg_val()

    @property
    def unit_reg(self):
        if self._unit_reg > 0:
            return True
        else:
            return False

    @unit_reg.setter
    def unit_reg(self, value):
        assert isinstance(value, bool), "Regularization.unit_reg: boolean value required"
        if value:
            self._unit_reg.data[:] = 1
        else:
            self._unit_reg.data[:] = 0

    def unit_reg_convert( self, unit_reg=True, num_outputs=None ):
        """Can convert reg object to turn on or off unit_reg (default turns it on"""

        if unit_reg == self.unit_reg:
            print("UNIT_REG: No conversion necessary")
            return
        if unit_reg:
            assert self.num_outputs is not None, "UNIT_REG: must set num_ouputs"

        self.unit_reg = unit_reg 

        for reg_type, val in self.vals.items():
            if not unit_reg:  # turn off unit reg -- convert from list to single number via mean
                new_val = np.mean(val) 
            else:
                new_val = val
            self.set_reg_val(reg_type, reg_val=new_val)
    # END Regularization.unit_reg_convert

    def build_reg_modules(self, device=None):
        """Prepares regularization modules in train based on current regularization values"""
        
        self.reg_modules = nn.ModuleList()  # this clears old modules (better way?)
        if device is None:
            device=torch.device('cpu')
            
        for reg, val in self.vals.items():
            # check for boundary conditions
            BC = 1 # default padding for boundary conditions
            if self.boundary_conditions is not None:
                if reg in self.boundary_conditions:
                    BC = self.boundary_conditions[reg]
                    #print('  Setting boundary conditions for', reg, '=', BC)

            reg_obj = self.get_reg_class(reg)(
                reg_type=reg, reg_val=val, 
                input_dims=self.input_dims, pos_constraint=self.pos_constraint,
                folded_lags=self.folded_lags, unit_reg=self.unit_reg, bc_val=BC)
            self.reg_modules.append(reg_obj.to(device))

            if reg == 'activity':
                self.activity_regmodule = self.reg_modules[-1]  # hoping this acts as a pointer (otherwise use explicit indexing)
                # note this currently will only handle one activity reg type at once, otherwise make list 
    # END Regularization.build_reg_modules

    def compute_reg_loss(self, weights):
        # I don't think this flag is yet used anywhere -- weights are passed in normalized, if relevant
        if self.normalize: 
            weights = F.normalize(weights, dim=0)

        rloss = torch.tensor(0, dtype=torch.float32, device=weights.device)
        for regmod in self.reg_modules:
            rloss += regmod( weights )

        return rloss
    # END Regularization.define_reg_loss

    def compute_activity_regularization(self, layer_output):
        # first put wrapper so wont throw bug with older model
        if self.activity_regmodule is not None:
            self.activity_regmodule.compute_activity_penalty(layer_output)

    def reg_copy(self):
        """Copy regularization to new structure"""

        reg_target = Regularization(input_dims=self.input_dims)
        reg_target.vals = deepcopy(self.val)

        return reg_target

    @staticmethod
    def get_reg_class(reg_type=None):

        reg_index = {'d2xt': ConvReg,
                     'd2x': ConvReg,
                     'd2t': ConvReg,
                     'l1': InlineReg,
                     'l2': InlineReg,
                     'norm2': InlineReg, # weight regularization
                     'norm': InlineReg,
                     'pos': InlineReg,
                     'neg': InlineReg,
                     'orth': InlineReg,
                     'bi_t': InlineReg,
                     'glocalx': LocalityReg,
                     'glocalt': LocalityReg,
                     'localx': LocalityReg,
                     'localt': LocalityReg,
                     'trd': LocalityReg,
                     'local': Tikhanov,
                     'glocal': Tikhanov,
                     'max': Tikhanov,
                     'max_filt': Tikhanov,
                     'max_space': Tikhanov,
                     'gmax_t': TikhanovC,
                     'gmax_space': TikhanovC,
                     'gmax_filter': TikhanovC,
                     'center': DiagonalReg,
                     'edge_t': DiagonalReg,
                     'edge_t0': DiagonalReg,
                     'edge_x': DiagonalReg,
                     'activity': ActivityReg}
        
        if reg_type is None:
            ret = reg_index.keys()
        else:
            ret = reg_index[reg_type]

        return ret
    # END Regularization.reg_copy

class RegModule(nn.Module): 
    """ Base class for regularization modules """

    def __init__(self, reg_type=None, reg_val=None, 
                 input_dims=None, num_dims=0, unit_reg=False, folded_lags=False, 
                 pos_constraint=False, **kwargs):
        """Constructor for Reg_module class"""

        assert reg_type is not None, 'Need reg_type.'
        assert reg_val is not None, 'Need reg_val'
        assert input_dims is not None, 'Need input dims'

        super().__init__()

        self.reg_type = reg_type
        self.unit_reg = unit_reg
        self.register_buffer( 'val', torch.tensor(reg_val))
        self.input_dims = input_dims
        self.num_dims = num_dims # this is the relevant number of dimensions for some filters -- will be set within functions
        self.folded_lags = folded_lags
        self.pos_constraint = pos_constraint

    def forward(self, weights):
        rpen = self.compute_reg_penalty(weights)
        rpen = self.val * rpen
        
        return rpen.mean()

class LocalityReg(RegModule):
    """ Regularization to penalize locality separably for each dimension"""
    
    def __init__(self, reg_type=None, reg_val=None, input_dims=None, num_dims=0, **kwargs):
        """Constructor for LocalityReg class"""

        _valid_reg_types = ['localx', 'localt', 'glocalx', 'glocalt', 'trd']
        assert reg_type in _valid_reg_types, '{} is not a valid Locality Reg type'.format(reg_type)

        super().__init__(reg_type, reg_val, input_dims, num_dims, **kwargs)

        self.build_reg_mats()
        # normalized weights mean w ~ 1/sqrt(N), so \sum w^2 = 1. but w^4 => 1/N^2 so multiply by N
        #self.multiplier = np.prod(input_dims)

    def compute_reg_penalty(self, weights):
        """Compute regularization penalty for locality"""

        rpen = 0
        if self.pos_constraint:
            w = weights.reshape(self.input_dims + [-1]) #[C, NX, NY, num_lags, num_filters]
        else:
            w = weights.reshape(self.input_dims + [-1])**2 #[C, NX, NY, num_lags, num_filters]

        if self.reg_type == 'glocalx':
            
            # glocal                
            wx = w.sum(dim=(0,2,3)) # sum over y and t (note this works for singular dim y too, i.e., 1d)

            penx = torch.einsum('xn,xw->wn', wx, self.localx_pen)
            penx = torch.einsum('xn,xn->n', penx, wx)

            if self.num_dims == 1:
                rpen = penx
            else:
                wy = w.sum(dim=(0,1,3)) # sum over x and t
        
                if self.same_dims:
                    peny = torch.einsum('yn,yw->wn', wy, self.localx_pen)
                    peny = torch.einsum('yn,yn->n', peny, wy)
                else:    
                    peny = torch.einsum('yn,yw->wn', wy, self.localy_pen)
                    peny = torch.einsum('yn,yn->n', peny, wy)

                rpen = penx + peny

        elif self.reg_type == 'glocalt':
            
            # glocal on time
            #w = weights.reshape(self.input_dims + [-1])**2
            wt = w.sum(dim=(0,1,2))

            rpen = torch.einsum('tn,tw->wn', wt, self.localt_pen)
            rpen = torch.einsum('tn,tn->n', rpen, wt)
        
        elif self.reg_type == 'localx':
            
            # glocal
            #w = weights.reshape(self.input_dims + [-1])**2 #[C, NX, NY, num_lags, num_filters]

            penx = torch.einsum('cxytn,xw->wn', w, self.localx_pen)
            penx = torch.einsum('xn,cxytn->n', penx, w)

            if self.num_dims == 1:
                rpen = penx
            else:
                if self.same_dims:
                    peny = torch.einsum('cxytn,yw->wn', w, self.localx_pen)
                    peny = torch.einsum('yn,cxytn->n', peny, w)
                else:    
                    peny = torch.einsum('cxytn,yw->wn', w, self.localy_pen)
                    peny = torch.einsum('yn,cxytn->n', peny, w)

                rpen = penx + peny
        
        elif self.reg_type == 'localt':
            
            # glocal on time
            #w = weights.reshape(self.input_dims + [-1])**2

            rpen = torch.einsum('cxytn,tw->wn', w, self.localt_pen)
            rpen = torch.einsum('tn,cxytn->n', rpen, w)
        
        elif self.reg_type == 'trd':
            
            # penalty on time
            #w = weights.reshape(self.input_dims + [-1])**2

            rpen = torch.einsum('cxytn,tw->wn', w, self.trd_pen)
            rpen = torch.einsum('tn,cxytn->n', rpen, w)
        
        return rpen
    # END LocalityReg.compute_reg_penalty

    def build_reg_mats(self):
        
        if self.reg_type == 'glocalx' or self.reg_type == 'localx':
            self.register_buffer('localx_pen',((torch.arange(self.input_dims[1])-torch.arange(self.input_dims[1])[:,None])**2).float()/self.input_dims[1]**2)
            if self.input_dims[1]==self.input_dims[2]:
                self.same_dims = True
            else:
                self.same_dims = False
                self.register_buffer('localy_pen',((torch.arange(self.input_dims[2])-torch.arange(self.input_dims[2])[:,None])**2).float()/self.input_dims[2]**2)

        elif self.reg_type == 'glocalt' or self.reg_type == 'localt':
            self.register_buffer('localt_pen',((torch.arange(self.input_dims[3])-torch.arange(self.input_dims[3])[:,None])**2).float()/self.input_dims[3]**2)
        
        elif self.reg_type == 'trd':
            self.register_buffer('trd_pen', (torch.sqrt(torch.arange(self.input_dims[3])**2+torch.arange(self.input_dims[3])[:,None]**2)).float()/self.input_dims[3]**2)

class DiagonalReg(RegModule):
    """ Regularization module for diagonal penalties"""

    def __init__(self, reg_type=None, reg_val=None, input_dims=None, **kwargs):

        assert reg_type in ['center', 'edge_t', 'edge_t0', 'edge_x'], "{} is not a valid Diagonal Regularization type".format(reg_type)
        super().__init__(reg_type=reg_type, reg_val=reg_val, input_dims=input_dims, **kwargs)

        # the "input_dims" are ordered differently for matrix implementations,
        # so this is a temporary fix to be able to use old functions
        self.input_dims = input_dims

        reg_mat = self.build_reg_mats(reg_type)
    
    def compute_reg_penalty(self, weights):
        """Compute regularization loss"""

        if self.pos_constraint:
            w = weights.reshape(self.input_dims + [-1]) #[C, NX, NY, num_lags, num_filters]
        else:
            w = weights.reshape(self.input_dims + [-1])**2 #[C, NX, NY, num_lags, num_filters]

        # Collapse weights across non-spatial dimensions
        if self.reg_type == 'center':
            wcollapse = w.mean(dim=(0,3,4)).reshape([-1])
            rpen = torch.mean( torch.matmul( wcollapse, self.regcost) )

        elif self.reg_type in ['edge_t', 'edge_t0']:
            wcollapse = w.mean(dim=(0,1,2,4)).reshape([-1])
            rpen = torch.mean( torch.matmul( wcollapse, self.regcost) )

        elif self.reg_type == 'edge_x':
            wcollapse = w.mean(dim=(0,2,3,4)).reshape([-1])
            rpen = torch.mean( torch.matmul( wcollapse, self.regcost_x) )
            if self.input_dims[2] > 1:
                wcollapse = w.mean(dim=(0,1,3,4)).reshape([-1])
                rpen += torch.mean( torch.matmul( wcollapse, self.regcost_y) )

        return rpen
    
    def build_reg_mats(self, reg_type):
    
        NX, NY = self.input_dims[1:3]

        if reg_type == 'center':

            center_x = (NX - 1) / 2
            px = (np.arange(NX)-center_x)/center_x

            if NY == 1:  # then 1-d space
                rmat = px**2
            else:
                center_y = (NY - 1) / 2
                py = (np.arange(NY)-center_y)/center_y

                # Make a matrix and then reshape
                rmat = np.reshape( px[:,None]**2 @ np.ones([1,NY]) + np.ones([NX,1]) @ py[None,:]**2, [-1])

            self.register_buffer( 'regcost', torch.Tensor(rmat))

        elif reg_type == 'edge_t':
            rmat = np.zeros(self.input_dims[3])
            rmat[0] = 1
            rmat[-1] = 1
            self.register_buffer( 'regcost', torch.Tensor(rmat))

        elif reg_type == 'edge_t0':
            rmat = np.zeros(self.input_dims[3])
            rmat[0] = 1
            self.register_buffer( 'regcost', torch.Tensor(rmat))

        elif reg_type == 'edge_x':
            rmat_x = np.zeros(self.input_dims[1])
            rmat_x[0] = 1
            rmat_x[-1] = 1
            self.register_buffer( 'regcost_x', torch.Tensor(rmat_x))
            rmat_y = np.zeros(self.input_dims[2])
            rmat_y[0] = 1
            rmat_y[-1] = 1
            self.register_buffer( 'regcost_y', torch.Tensor(rmat_y))


class InlineReg(RegModule):
    """ Regularization module for inline penalties"""
    def __init__(self, reg_type=None, reg_val=None, input_dims=None, **kwargs):
        assert reg_type in ['l1', 'l2', 'norm2', 'norm', 'pos', 'neg', 'orth', 'bi_t'], "{} is not a valid Inline Regularization type".format(reg_type)
        super().__init__(reg_type=reg_type, reg_val=reg_val, input_dims=input_dims, **kwargs)

    def compute_reg_penalty(self, weights):
        
        """Calculate regularization penalty for various reg types"""
        if self.reg_type == 'l1':
            reg_pen = torch.mean(torch.abs(weights), dim=0)

        elif self.reg_type == 'l2':
            reg_pen = torch.mean(torch.square(weights), dim=0)
        
        elif self.reg_type == 'norm2':  # [custom] convex (I think) soft-normalization regularization
            reg_pen = torch.square( torch.mean(torch.square(weights), dim=0)-1 )
        
        elif self.reg_type == 'norm':  # [custom] convex (I think) soft-normalization regularization
            reg_pen = (weights.norm(dim=0)-1).pow(2).mean()
        elif self.reg_type == 'pos':  # [custom] soft positive regularization
            reg_pen = F.relu(-weights).mean()

        elif self.reg_type == 'neg':  # [custom] soft negative regularization
            reg_pen = F.relu(weights).mean()

        elif self.reg_type == 'orth':  # [custom] orthogonal regularization
            w = (weights.T @ weights).abs()
            reg_pen = w.sum() - w.trace()
        elif self.reg_type == 'bi_t':
            w = weights.reshape(self.input_dims + [-1])
            n = w.shape[-1]
            w = w.sum(dim=3)**2
            w = w.reshape(-1, n)
            reg_pen = torch.mean(w, dim=0)
        else:
            reg_pen = 0.0

        return reg_pen

class ConvReg(RegModule):
    """ Regularization module for convolutional penalties"""
    def __init__(self, reg_type=None, reg_val=None, input_dims=None, bc_val=1, **kwargs):
        """Constructor for Reg_module class"""
        _valid_reg_types = ['d2x', 'd2t', 'd2xt']
        assert reg_type in _valid_reg_types, '{} is not a valid ConvReg type'.format(reg_type)
        super().__init__(reg_type=reg_type, reg_val=reg_val, input_dims=input_dims, **kwargs)

        reg_mat = self._make_laplacian(reg_type)
        self.register_buffer( 'rmat', torch.Tensor(reg_mat))
        self.BC = bc_val
    
    def compute_reg_penalty(self, weights):
        """I'm separating the code for more complicated regularization penalties in the simplest possible way here, but
        this can be done more (or less) elaborately in the future. The challenge here is that the dimension of the weight
        vector (1-D, 2-D, 3-D) determines what sort of Laplacian matrix, and convolution, to do
        Note that all convolutions are implicitly 'valid' so no boundary conditions"""
        num_filters = weights.shape[1]
        weight_dims = self.input_dims + [num_filters]
        if self.folded_lags:
            w = weights.reshape((weight_dims[0], weight_dims[3], weight_dims[1], weight_dims[2]))            
        else:
            w = weights.reshape(weight_dims)
        # puts in [C, W, H, T, num_filters]: to reorder depending on reg type
        # default reg_dims
        if self.reg_type == 'd2t':
            w = w.permute(4,0,1,2,3) # needs temporal dimension last so only convolved
            #reg_dims = [weight_dims[3]]
        elif self.reg_type == 'd2xt':
            if self.folded_lags:
                print('FL')
                w = w.permute(4,0,2,3,1) # weights correspding to lag are up front
            else:
                w = w.permute(4,0,1,2,3) 
            # Reg-dims will depend on whether space is one- or two-dimensional
            #if weight_dims[2] > 1:
            #    reg_dims = [weight_dims[1], weight_dims[2], weight_dims[3]]
            #else:
            #    reg_dims = [weight_dims[1], weight_dims[3]]
        else:  # then d2x
            w = w.permute(4,0,3,1,2)  # rotate temporal dimensions next to filter dims
            #reg_dims = [weight_dims[1], weight_dims[2]]

        # Apply boundary conditions dependent on default values (that can be set by hand):
        if self.num_dims == 1:
            # prepare for 1-d convolve
            rpen = F.conv1d( 
                w.reshape( [-1, 1] + self.reg_dims[:1] ),  # [batch_dim, all non-conv dims, conv_dim]
                self.rmat, padding=self.BC ).pow(2) #/ weight_dims[-1]

        elif self.num_dims == 2:
            rpen = F.conv2d( 
                w.reshape( [-1, 1] + self.reg_dims[:2] ), 
                self.rmat, padding=self.BC ).pow(2) #/ weight_dims[-1]
            

        elif self.num_dims == 3:
            rpen = F.conv3d( 
                w.reshape( [-1,1] + self.reg_dims),
                self.rmat, padding=self.BC ).pow(2) #/ weight_dims[-1]
        
        # average over all dimensions except the filter dimension
        rpen = rpen.reshape(num_filters, -1)
        rpen = rpen.mean(dim=1)

        return rpen
    
    def _make_laplacian( self, reg_type ):
        """This will make the Laplacian of the right dimensionality depending on d2xt, d2t, d2x"""

        import numpy as np

        # Determine relevant reg_dims for Laplacian
        if self.reg_type == 'd2t':
            self.reg_dims = [self.input_dims[3]]
        elif self.reg_type == 'd2x':
            self.reg_dims = [self.input_dims[1], self.input_dims[2]]
        else:  # d2xt
            if self.input_dims[2] == 1:
                self.reg_dims = [self.input_dims[1], self.input_dims[3]]
            else:
                self.reg_dims = self.input_dims[1:]

        # Determine number of dimensions for laplacian matrix (and convolution)
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
            #rmat = np.array([[[[0,-1,0],[-1, 4, -1], [0,-1,0]]]])
            # Isotropic form of discrete Laplacian operator (https://en.wikipedia.org/wiki/Discrete_Laplace_operator)
            rmat = np.array([[[[0.25,0.5,0.25],[0.5, -3, 0.5], [0.25,0.5,0.25]]]])
        elif self.num_dims == 3:
            #rmat = np.array(
            #    [[[[[0, 0, 0],[0, -1, 0], [0, 0, 0]],
            #    [[0, -1, 0],[-1, 6, -1], [0, -1, 0]],
            #    [[0, 0, 0],[0, -1, 0], [0, 0, 0]]]]])
            # Isotropic form:
            rmat = 1/26*np.array(
                [[[[[2, 3, 2],[3, 6, 3], [2, 3, 2]],
                [[3, 6, 3],[6, -88, 6], [3, 6, 3]],
                [[2, 3, 2],[3, 6, 3], [2, 3, 2]]]]])
        else:
            rmat = np.array([1])
            print( "Warning: %s regularization does not have the necessary filter dimensions."%self.reg_type )
        return rmat
        

class Tikhanov(RegModule):
    """Regularization module for Tikhanov regularization"""
    def __init__(self, reg_type=None, reg_val=None, input_dims=None, bc_val=0, **kwargs):
        """Constructor for Tikhanov class"""

        super().__init__(reg_type=reg_type, reg_val=reg_val, input_dims=input_dims, bc_val=bc_val, **kwargs)
        
        _valid_reg_types = ['local', 'glocal', 'max', 'max_filt', 'max_space']
        assert reg_type in _valid_reg_types, "{} is not a valid Tikhanov regularization type".format(reg_type)
        
        # the "input_dims" are ordered differently for matrix implementations,
        # so this is a temporary fix to be able to use old functions
        self.input_dims = [
            input_dims[0]*input_dims[3], # non-spatial
            input_dims[1], input_dims[2]]  # spatial 

        # Make appropriate reg_matrix as buffer (non-fit parameter)
        reg_tensor = self._build_reg_mats( reg_type )
        self.register_buffer( 'rmat', reg_tensor)
    
    def compute_reg_penalty(self, weights):
        """Calculate regularization penalty for Tikhanov reg types"""

        w2 = torch.square(weights)
        reg_pen = torch.matmul( w2.T, torch.matmul(self.rmat, w2) )
        reg_pen = torch.diagonal(reg_pen, 0) # sum later (replaced call to torch.trace)

        return reg_pen

    def _build_reg_mats(self, reg_type):
        """Build regularization matrices in default tf Graph

        Args:
            reg_type (str): see `_allowed_reg_types` for options
        """
        import NDNT.utils.create_reg_matrices as get_rmats

        if (reg_type == 'max') or (reg_type == 'max_filt') or (reg_type == 'max_space'):
            reg_mat = get_rmats.create_maxpenalty_matrix(self.input_dims, reg_type)
        elif reg_type == 'local':
            reg_mat = get_rmats.create_localpenalty_matrix(
                self.input_dims, separable=False)
        elif reg_type == 'glocal':
            reg_mat = get_rmats.create_localpenalty_matrix(
                self.input_dims, separable=False, spatial_global=True)
        else:
            reg_mat = None

        if reg_mat is None:
            return None
        else:
            return torch.Tensor(reg_mat)
    # END Tikhanov.build_reg_mats


class TikhanovC(RegModule):
    """Regularization module for Tikhanov-collapse regularization -- where all but one
    dimension is marginalized over first"""

    def __init__(self, reg_type=None, reg_val=None, input_dims=None, bc_val=0, **kwargs):
        """Constructor for Tikhanov class"""

        super().__init__(reg_type=reg_type, reg_val=reg_val, input_dims=input_dims, bc_val=bc_val, **kwargs)
        
        _valid_reg_types = ['gmax_t', 'gmax_space', 'gmax_filter']
        assert reg_type in _valid_reg_types, "{} is not a valid Tikhanov-C regularization type".format(reg_type)
        
        # the "input_dims" are ordered differently for matrix implementations,
        # so this is a temporary fix to be able to use old functions
        self.input_dims = [input_dims[0], input_dims[1]*input_dims[2], input_dims[3]]

        self.collapse_dims = [2, 1, 0]
        if reg_type == 'gmax_t':
            self.reg_dim = 2
        elif reg_type == 'gmax_filter':
            self.reg_dim = 0
        else:
            self.reg_dim = 1
        self.collapse_dims.remove(self.reg_dim)

        # Make appropriate reg_matrix as buffer (non-fit parameter)
        reg_tensor = self._build_reg_mats( reg_type )
        self.register_buffer( 'rmat', reg_tensor)
    
    def compute_reg_penalty(self, weights):
        """Calculate regularization penalty for Tikhanov reg types"""

        #w2 = torch.square(weights)  # assuming positive constraints here -- no squaring needed
        w = torch.sum( torch.sum(
                weights.reshape(self.input_dims + [-1]), # [C, W*H, T, NF]
                axis=self.collapse_dims[0]), 
                axis=self.collapse_dims[1])

        reg_pen = torch.matmul( w.T, torch.matmul(self.rmat, w) )
        reg_pen = torch.diagonal(reg_pen, 0) # sum later (replaced call to torch.trace)

        return reg_pen

    def _build_reg_mats(self, reg_type):
        """Build regularization matrices in default tf Graph

        Args:
            reg_type (str): see `_allowed_reg_types` for options
        """
        #import NDNT.utils.create_reg_matrices as get_rmats
        L = self.input_dims[self.reg_dim]
        reg_mat = np.ones([L,L]) - np.eye(L)
        return torch.Tensor(reg_mat)
    # END TikhanovC.build_reg_mats


class ActivityReg(RegModule):
    """ Regularization to penalize activity separably for each dimension
    Note that the penalty needs to be computed elsewhere and just stored here"""
    
    def __init__(self, reg_type=None, reg_val=None, input_dims=None, num_dims=0, **kwargs):
        """Constructor for LocalityReg class"""

        _valid_reg_types = ['activity']
        assert reg_type in _valid_reg_types, '{} is not a valid Locality Reg type'.format(reg_type)

        super().__init__(reg_type, reg_val, input_dims, num_dims, **kwargs)

        #self.register_buffer( 'activity_penalty', torch.zeros(1) )
        self.activity_penalty = 0.0
    # END ActivityReg.__init__

    def compute_activity_penalty(self, acts ):
        """Computes activity penalty from activations -- called in layer forward"""
        # Can put conditionals if there is more than one reg module
        self.activity_penalty = torch.mean(torch.sum(acts**2, axis=1), axis=0)
        #test = torch.mean(torch.sum(acts**2, axis=1), axis=0)  

    def compute_reg_penalty(self, weights):
        """Compute regularization penalty for locality"""
        return self.activity_penalty  # this is precomputed
    # END ActivityReg.__init__
