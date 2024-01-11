from numpy.lib.arraysetops import isin
import torch
from torch.nn import functional as F
import torch.nn as nn

from .ndnlayer import NDNLayer
from .convlayers import ConvLayer, TconvLayer
import numpy as np
import torchvision
import torchvision.transforms.functional as TF

class OriLayer(NDNLayer):
    """
    Orientation layer.
    """

    def __init__(
            self, input_dims=None, num_filters=None,
            filter_dims=None, angles=None, **kwargs): 
        """
        Initialize orientation layer.
        :param input_dims: input dimensions
        :param num_filters: number of filters
        :param filter_dims: filter dimensions
        :param angles: angles for rotation (in degrees)
        """

        assert input_dims is not None, "OriLayer: Must specify input dimensions"
        assert num_filters is not None, "OriLayer: Must specify number of filters"
        assert angles is not None, "OriLayer: Must specify angles for rotation"

        super().__init__(input_dims=input_dims, num_filters=num_filters, 

        filter_dims=filter_dims, **kwargs)
        self.angles=angles

        rotation_matrices=self.rotation_matrix_tensor(self.filter_dims, self.angles)
        self.register_buffer('rotation_matrices', rotation_matrices)

        new_output_dims=[self.num_filters, 1, 1, len(self.angles)+1]  
        self.output_dims=new_output_dims 

    def rotation_matrix_tensor(self, filter_dims, theta_list):
        assert self.filter_dims[2] != 1, "OriLayer: Stimulus must be 2-D"

        w = filter_dims[1]
        if w%2 == 1:             
            x_pos = np.repeat(np.linspace(-np.floor(w/2), np.floor(w/2), w), w)
            y_pos = np.tile(np.linspace(-np.floor(w/2), np.floor(w/2), w), w)
        else:
            x_pos = np.repeat((np.linspace(-(w-1), w-1, w)), w)
            y_pos = np.tile((np.linspace(-(w-1), w-1, w)), w)

        indices=[]
        for k in range(len(theta_list)):
            assert theta_list[k]==int(theta_list[k]), "OriLayer: All angles must be in degrees!"

            theta=theta_list[k]*np.pi/180
            rotation_matrix=rotation_matrix=np.array(
                [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)]).reshape(2, 2)
            
            for i in range(len(x_pos)):
                vector = np.array([x_pos[i], y_pos[i]]).reshape(2, 1)
                rotated_vector = np.matmul(rotation_matrix, vector)     
                integer_center = np.round(rotated_vector)

                if w%2 == 0: 
                    frac_part,_ =np.modf(integer_center)
                    for j in range(2):
                        if integer_center[j]%2==0:
                            if abs(frac_part[j])<0.5:
                                integer_center[j]+=np.sign(integer_center[j])
                            else:
                                integer_center[j]-=np.sign(integer_center[j])

                x_loc = np.argwhere(x_pos == integer_center[0])
                if len(x_loc) == w:
                    x_search = x_loc.reshape(w)
                    y_loc = x_search[np.argwhere(y_pos[x_search] == integer_center[1])]

                    if len(y_loc) == 0:
                        continue
                    else:
                        indices.append([k, i, int(y_loc[0, 0])])
                else:
                    continue
        
        rotation_matrix_tensor=torch.sparse_coo_tensor(
            torch.tensor(indices).t(), torch.ones(len(indices)), size=(len(theta_list), w**2, w**2))
        
        return rotation_matrix_tensor 
    
    #Minor issue: Einstein summation does not seem to work with sparse tensors?
    #This is actually a major problem as the tensors are 3600 by 3600 by other stuff.

    def forward(self, x):
        w = self.preprocess_weights() #What shape is this? (whatever self.shape is)
        #You still need the original! w is NC*NXY*NT by NF 
        x_0 = torch.matmul(x, w) #linearity 
        if self.norm_type == 2:
            x_0 = x_0 / self.weight_scale #normalization 

        x_0 = x_0 + self.bias #bias 

        #if self.output_norm is not None:
        #    x = self.output_norm(x)

        # Nonlinearity
        if self.NL is not None:
            x_0 = self.NL(x_0)

        # Constrain output to be signed
        if self._ei_mask is not None:
            x_0 = x_0 * self._ei_mask
        
        x_hats = torch.zeros(tuple([x_0.shape[0], x_0.shape[1], len(self.angles)+1])) 
        x_hats[:,:,0] = x_0 # Wrangle shapes now. 
        w_slicing = w.reshape(self.filter_dims[0], self.filter_dims[1], self.filter_dims[2], self.filter_dims[3], self.num_filters)
        w_flattened = w_slicing.reshape(self.filter_dims[1]*self.filter_dims[2], self.filter_dims[0]*self.filter_dims[3]*self.num_filters) 

        for i in range(len(self.angles)):
            w_theta = torch.sparse.mm(self.rotation_matrices[i], w_flattened)
            w_reshaped = w_theta.reshape(self.shape) #NC*NXY*NT by NF 
            x_theta = torch.sparse.mm(x, w_reshaped) #Dimensionality! (x is B by NC*NXY*NT)
            if self.norm_type == 2:
                x_theta = x_theta / self.weight_scale
            x_theta = x_theta + self.bias
            if self.NL is not None:
                x_theta = self.NL(x_theta)
            if self._ei_mask is not None:
                x_theta = x_theta * self._ei_mask
            #self.activity_regularization = self.activity_reg.regularize(x_theta) #Do we need to change that? 
            x_hats[:,:,i+1] = x_theta

        # store activity regularization to add to loss later
        if hasattr(self.reg, 'activity_regmodule'):  # to put buffer in case old model
            self.reg.compute_activity_regularization(x_hats)

        return x_hats.reshape([x_0.shape[0], -1])
    #END OriLayer.forward

    @classmethod
    def layer_dict(cls, angles=None, **kwargs):
        Ldict = super().layer_dict(**kwargs)
        Ldict["layer_type"]="ori"
        Ldict["angles"]=angles
        return Ldict


class OriConvLayer(ConvLayer):
    """
    Orientation layer.
    """

    def __init__(self, input_dims=None, num_filters=None, res_layer=False,
                 filter_dims=None, padding="valid", output_norm=None, angles=None, **kwargs): 
        """
        Initialize orientation layer.
        :param input_dims: input dimensions
        :param num_filters: number of filters
        :param filter_dims: filter dimensions
        :param angles: angles for rotation (in degrees)
        """

        # input validation
        assert input_dims is not None, "OriConvLayer: Must specify input dimensions"
        assert len(input_dims) == 4, "OriConvLayer: Stimulus must be 2-D"
        assert input_dims[3] == 1, "OriConvLayer: Stimulus must be 2-D"
        assert num_filters is not None, "OriConvLayer: Must specify number of filters"
        assert angles is not None, "OriConvLayer: Must specify angles for rotation"
        assert not res_layer, "OriConvLayer: res_layer not yet supported"

        super().__init__(
            input_dims=input_dims, num_filters=num_filters,
            filter_dims=filter_dims, padding=padding, res_layer=False,
            output_norm=None, **kwargs)

        if 'bias' in kwargs.keys():
            assert kwargs['bias'] == False, "OriConvLayer: bias is partially implemented, but not debugged"

        #self.is1D = (self.input_dims[2] == 1)
        assert not self.is1D, "OriConvLayer: Stimulus must be 2-D"

        self.angles = angles

        # Fix output norm to be 3d
        if output_norm in ['batch', 'batchX']:
            if output_norm == 'batchX':
                affine = False
            else:
                affine = True
            self.output_norm = nn.BatchNorm2d(self.num_filters*len(angles), affine=affine)

        # make the rotation matrices and store them as a buffer
        rotation_matrices = self.rotation_matrix_tensor(self.filter_dims, self.angles)
        self.register_buffer('rotation_matrices', rotation_matrices)

        # make the ei mask and store it as a buffer,
        # repeat it for each orientation (plus one for the original orientation)
        if self._ei_mask is not None: 
            self.register_buffer('_ei_mask',
                                torch.cat(
                                    (torch.ones(self.num_filters-self._num_inh), 
                                    -torch.ones(self._num_inh))
                                ).repeat(len(self.angles)))
            #print('ei_mask', self._ei_mask.shape)

        # Make additional window function to preserve rotations  
        L = self.filter_dims[1]
        xs = np.arange(L)+0.5-L/2
        rs = np.sqrt(np.repeat(xs[:,None]**2, L, axis=1) + np.repeat(xs[None,:]**2, L, axis=0))
        win_circle = np.ones([L,L], dtype=np.float32)
        win_circle[rs > L/2] = 0.0
        if self.window:
            self.window_function *= torch.tensor(win_circle, dtype=torch.float32)
        else:
            self.register_buffer('window_function', torch.tensor(win_circle, dtype=torch.float32))

        # folded_dims is num_filter * num_angles * num_incoming_filters
        self.folded_dims = self.filter_dims[0] # input filters * lags
        # we need to set the entire output_dims so that num_outputs gets updated in the setter
        self.output_dims = [self.output_dims[0], self.output_dims[1], self.output_dims[2], len(self.angles)]

    def rotation_matrix_tensor(self, filter_dims, theta_list):
        w = filter_dims[1] # width
        # if the width is odd, then the center is at the floor of the center
        if w%2 == 1:
            x_pos = np.repeat(np.linspace(-np.floor(w/2), np.floor(w/2), w), w)
            y_pos = np.tile(np.linspace(-np.floor(w/2), np.floor(w/2), w), w)
        # if the width is even, then the center is at the center
        if w%2 == 0:
            x_pos = np.repeat((np.linspace(-(w-1), w-1, w)), w)
            y_pos = np.tile((np.linspace(-(w-1), w-1, w)), w)
        
        indices = []
        for k in range(len(theta_list)):
            assert theta_list[k] == int(theta_list[k]), "OriLayer: All angles must be in degrees!"
            theta = theta_list[k]*np.pi/180
            rotation_matrix = np.array([np.cos(theta), 
                                        -np.sin(theta), 
                                        np.sin(theta), 
                                        np.cos(theta)]).reshape(2, 2)
            
            for i in range(len(x_pos)):
                vector = np.array([x_pos[i], y_pos[i]]).reshape(2, 1)
                rotated_vector = np.matmul(rotation_matrix, vector)     
                integer_center = np.round(rotated_vector)
                if w%2 == 0:
                    frac_part, _ = np.modf(integer_center)
                    for j in range(2):
                        if integer_center[j]%2 == 0:
                            if abs(frac_part[j]) < 0.5:
                                integer_center[j] += np.sign(integer_center[j])
                            else:
                                integer_center[j] -= np.sign(integer_center[j])
                x_loc = np.argwhere(x_pos == integer_center[0])
                if len(x_loc) == w:
                    x_search = x_loc.reshape(w)
                    y_loc = x_search[np.argwhere(y_pos[x_search] == integer_center[1])] 
                    if len(y_loc) == 0:
                        continue
                    else:
                        indices.append([k, i, int(y_loc[0,0])])
                else:
                    continue 

        rotation_matrix_tensor = torch.sparse_coo_tensor(
            torch.tensor(indices).t(), torch.ones(len(indices)), size=(len(theta_list), w**2, w**2))

        return rotation_matrix_tensor 

    def get_rot_mat(self, theta, device):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]]).to(device)

    def rotate_tensor(self, x, theta, dtype):
        # repeat along the batch dimension
        rot_mat = self.get_rot_mat(theta, x.device)[None, ...].type(dtype).repeat(x.shape[0],1,1)

        grid = F.affine_grid(rot_mat, x.size()).type(dtype)
        x = F.grid_sample(x, grid)
        # Remove the added dimensions
        x = x.squeeze(0).squeeze(0)
        return x

    def forward(self, x):
        if self._padding == 'circular':
            pad_type = 'circular'
        else:
            pad_type = 'constant'
        
        w = self.preprocess_weights()
        # permute num_filters, input_dims[0], width, height, (no lag)
        w = w.reshape(self.filter_dims[:3]+[self.num_filters]).permute(3, 0, 1, 2)
        # combine num_filters*input_dims[0] and width*height into one dimension
        w_flattened = w.reshape(self.num_filters*self.filter_dims[0], self.filter_dims[1]*self.filter_dims[2]) 
        s = x.reshape(-1, self.input_dims[0], self.input_dims[1], self.input_dims[2])

        # repeat weights for each angle along a new dimension
        # 1, 1 specifies to keep the filters and width*height dimensions the same
        rotated_ws = w_flattened.repeat(len(self.angles), 1, 1).permute(1, 2, 0)
        
        # use torch.sparse.mm to multiply the rotation matrices by the weights
        for i in range(len(self.angles)):
            # get the weights for the given angle
            #w_theta = rotated_ws[:, :, i]

            # sparse matmul method, but causes artifacting at non 90 degree angles
            # rotate the weight matrix for the given angle
            #w_theta = torch.sparse.mm(w_flattened, self.rotation_matrices[i])

            # rotate using torchvision transform
            w_theta = TF.rotate(img=w_flattened.reshape(-1, self.filter_dims[1], self.filter_dims[2]),
                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                angle=self.angles[i]).reshape(-1, self.filter_dims[1]*self.filter_dims[2])

            # put w_theta back into the full weight matrix
            rotated_ws[:, :, i] = w_theta
        
        # reshape the weights so we can put the angles in the second dimension
        rotated_ws_reshaped = rotated_ws.reshape((self.num_filters,
                                                  self.filter_dims[0], # in filters
                                                  self.filter_dims[1], # width
                                                  self.filter_dims[2], # height
                                                  len(self.angles)))

        # move the angles to the second dimension
        rotated_ws_reshaped = rotated_ws_reshaped.permute(0, 4, 1, 2, 3)
        # and combine the filters and angles into the folded_dims dimension
        # since we are convolving in the folded_dims dimension to do all filters at once
        rotated_ws_reshaped = rotated_ws_reshaped.reshape((self.num_filters*len(self.angles),
                                                           self.filter_dims[0],
                                                           self.filter_dims[1],
                                                           self.filter_dims[2]))

        if self._fullpadding:
            s_padded = F.pad(s, self.npads, pad_type, 0)
            y = F.conv2d(s_padded,
                         rotated_ws_reshaped, 
                         bias=self.bias.repeat(len(self.angles)),
                         stride=self.stride,
                         dilation=self.dilation)
        else:
            if self.padding == 'circular':
                s_padded = F.pad(s, self._npads, pad_type, 0)
                y = F.conv2d(s_padded, rotated_ws_reshaped,
                             bias=self.bias.repeat(len(self.angles)),
                             stride=self.stride,
                             dilation=self.dilation)
            else: # this is faster if not circular
                y = F.conv2d(s, rotated_ws_reshaped,
                             padding=(self._npads[2], self._npads[0]),
                             bias=self.bias.repeat(len(self.angles)),
                             stride=self.stride,
                             dilation=self.dilation)

        if not self.res_layer:
            if self.output_norm is not None:
                y = self.output_norm(y)
        
        if self.NL is not None:
            y = self.NL(y)
        if self._ei_mask is not None:
            y = y*self._ei_mask[None, :, None, None]

        # if self.res_layer:
        #     y = y+torch.reshape(s, (-1, self.folded_dims, self.input_dims[1], self.input_dims[2]) )
        #     if self.output_norm is not None:
        #         y = self.output_norm(y)
        
        # # store activity regularization to add to loss later
        # if hasattr(self.reg, 'activity_regmodule'):  # to put buffer in case old model
        #     self.reg.compute_activity_regularization(y)

        # pull the filters and angles apart again
        y = y.reshape(-1, # the batch dimension
                      self.num_filters,
                      len(self.angles),
                      self.output_dims[1], 
                      self.output_dims[2])
        # reshape y to have the orientiations in the last column
        y = y.permute(0, 1, 3, 4, 2)
        # flatten the last dimensions
        return y.reshape(-1, # the batch dimension
                         self.num_outputs)
    # OriConvLayer.forward

    @classmethod
    def layer_dict(cls, angles=None, **kwargs):
        Ldict = super().layer_dict(**kwargs)
        Ldict["layer_type"]="oriconv"
        Ldict["angles"]=angles
        return Ldict



class ConvLayer3D(ConvLayer):
    def __init__(self,
        input_dims:list=None, # [C, W, H, T]
        filter_width:int=None,
        num_filters:int=None,
        output_norm:int=None,
        **kwargs):
        
        assert input_dims is not None, "ConvLayer3D: input_dims must be specified"
        assert num_filters is not None, "ConvLayer3D: num_filters must be specified"
        assert filter_width is not None, "ConvLayer3D: filter_width must be specified"

        full_filter_dims = [input_dims[0], filter_width, filter_width, 1]
        input_dims_2D = [input_dims[0], input_dims[1], input_dims[2], 1]
        
        super().__init__(input_dims_2D, num_filters, filter_dims=full_filter_dims, output_norm=None, **kwargs)

        # output_norm will be the wrong dimensionality, so define here
        if output_norm in ['batch', 'batchX']:
            if output_norm == 'batchX':
                affine = False
            else:
                affine = True
            self.output_norm = nn.BatchNorm3d(self.num_filters, affine=affine)

        self.input_dims = input_dims
        self.output_dims = [num_filters, input_dims[1], input_dims[2], input_dims[3]]

        if self.res_layer:
            assert False, 'res_layer not implemented for ConvLayer3D'

    def forward(self, x):
        s = x.reshape([-1]+self.input_dims)

        w = self.preprocess_weights().reshape(self.filter_dims+[self.num_filters])

        if self._fullpadding:
            s = F.pad(s, self._npads, "constant", 0)
            y = F.conv3d(
                s, # we do our own padding
                w.permute(4,0,1,2,3), # num_filters is first
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation)
        else:
            # functional pads since padding is simple
            y = F.conv3d(
                s, 
                w.permute(4,0,1,2,3), # num_filters is first,
                padding=(self._npads[2], self._npads[0], 0),
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation)
        
        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)

        # EI mask
        if self._ei_mask is not None:
            # we just want to multiply the second dimension here
            y = y * self._ei_mask[None, :, None, None, None]

        # TODO: this is not tested yet
        # if self.res_layer:
        #     # s is with dimensions: B, C, T, X, Y 
        #     y = y + s                 

        # output norm (e.g. batch norm)
        if self.output_norm is not None:
            y = self.output_norm(y)

        # flatten the output
        y = torch.reshape(y, (-1, self.num_outputs))

        # TODO: this is not tested yet
        # # store activity regularization to add to loss later
        # #self.activity_regularization = self.activity_reg.regularize(y)
        # if hasattr(self.reg, 'activity_regmodule'):  # to put buffer in case old model
        #     self.reg.compute_activity_regularization(y)
        
        return y
    # END ConvLayer3D.forward

    def plot_filters( self, cmaps='gray', num_cols=8, row_height=2, time_reverse=False):
        # Overload plot_filters to automatically time_reverse
        super().plot_filters( 
            cmaps=cmaps, num_cols=num_cols, row_height=row_height, 
            time_reverse=time_reverse)

    @classmethod
    def layer_dict(cls, filter_width=None, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        del Ldict['filter_dims'] # remove this since we are manually setting it
        # Added arguments
        Ldict['layer_type'] = 'conv3d'
        Ldict['filter_width'] = filter_width
        return Ldict

