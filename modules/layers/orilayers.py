from numpy.lib.arraysetops import isin
import torch
from torch.nn import functional as F
import torch.nn as nn

from .ndnlayer import NDNLayer
from .convlayers import ConvLayer, TconvLayer
import numpy as np
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
        #assert units is not None, "OriLayer: Must state if angles are in radians or degrees"

        super().__init__(input_dims=input_dims, num_filters=num_filters, 

        filter_dims=filter_dims, **kwargs) #Self gets made here. 
        self.angles=angles
        #self.units=units

        rotation_matrices=self.rotation_matrix_tensor(self.filter_dims, self.angles)
        self.register_buffer('rotation_matrices', rotation_matrices) #Defines the thing!

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

    def __init__(self, input_dims=None, num_filters=None,
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

        super().__init__(
            input_dims=input_dims, num_filters=num_filters,
            filter_dims=filter_dims, padding=padding,
            output_norm=output_norm, **kwargs)
        
        if 'bias' in kwargs.keys():
            assert kwargs['bias'] == False, "OriConvLayer: bias is partially implemented, but not debugged"

        #print('init w', self.weight.shape)

        self.is1D = (self.input_dims[2] == 1)
        assert not self.is1D, "OriConvLayer: Stimulus must be 2-D"

        #print('input_dims', self.input_dims)

        self.angles = angles

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
        # repeate along the batch dimension
        rot_mat = self.get_rot_mat(theta, x.device)[None, ...].type(dtype).repeat(x.shape[0],1,1)

        grid = F.affine_grid(rot_mat, x.size()).type(dtype)
        x = F.grid_sample(x, grid)
        # Remove the added dimensions
        x = x.squeeze(0).squeeze(0)
        return x

    def forward(self, x):
        #print()
        #print('==== FORWARD ====')
        #print('input dims', self.input_dims)
        #print('x', x.shape)
        #print('w0', self.weight.shape)
        w = self.preprocess_weights()
        #w = self.weight
        #print('w', w.shape)
        # permute num_filters, input_dims[0], width, height, (no lag)
        w = w.reshape(self.filter_dims[:3]+[self.num_filters]).permute(3, 0, 1, 2)
        #print('w', w.shape)
        # combine num_filters*input_dims[0] and width*height into one dimension
        w_flattened = w.reshape(self.num_filters*self.filter_dims[0], self.filter_dims[1]*self.filter_dims[2]) 
        #print('wf', w_flattened.shape)
        s = x.reshape(-1, self.input_dims[0], self.input_dims[1], self.input_dims[2])
        #print('s', s.shape)

        # repeat weights for each angle along a new dimension
        # 1, 1 specifies to keep the filters and width*height dimensions the same
        rotated_ws = w_flattened.repeat(len(self.angles), 1, 1).permute(1, 2, 0)
        #print('wr', rotated_ws.shape)
        
        # use torch.sparse.mm to multiply the rotation matrices by the weights
        for i in range(len(self.angles)):
            # get the weights for the given angle
            #w_theta = rotated_ws[:, :, i]
            ##print('w_theta', w_theta.shape)

            # sparse matmul method, but causes artifacting at non 90 degree angles
            # rotate the weight matrix for the given angle
            #print('rotation_matrix', self.rotation_matrices[i].shape)
            #w_theta = torch.sparse.mm(w_flattened, self.rotation_matrices[i])
            #print('w_theta', w_theta.shape)

            # rotate using torchvision transform
            w_theta = TF.rotate(w_flattened.reshape(-1, self.filter_dims[1], self.filter_dims[2]), self.angles[i]).reshape(-1, self.filter_dims[1]*self.filter_dims[2])

            # put w_theta back into the full weight matrix
            rotated_ws[:, :, i] = w_theta
            #print('rotated_ws\'', rotated_ws.shape)
        
        # reshape the weights so we can put the angles in the second dimension
        rotated_ws_reshaped = rotated_ws.reshape((self.num_filters,
                                                  self.filter_dims[0], # in filters
                                                  self.filter_dims[1], # width
                                                  self.filter_dims[2], # height
                                                  len(self.angles)))

        # import matplotlib.pyplot as plt
        # for angle in range(len(self.angles)):
        #     plt.subplot(1, len(self.angles), angle+1)
        #     plt.imshow(rotated_ws_reshaped[0, 0, :, :, angle].detach().numpy())
        # plt.show()

        #print('wr\'', rotated_ws_reshaped.shape)
        # move the angles to the second dimension
        rotated_ws_reshaped = rotated_ws_reshaped.permute(0, 4, 1, 2, 3)
        #print('wr\'', rotated_ws_reshaped.shape)
        # and combine the filters and angles into the folded_dims dimension
        # put a 1 in the second dimension to match the input
        # since we are convolving in the folded_dims dimension to do all filters at once
        rotated_ws_reshaped = rotated_ws_reshaped.reshape((self.num_filters*len(self.angles),
                                                           self.filter_dims[0],
                                                           self.filter_dims[1],
                                                           self.filter_dims[2]))
        #print('wr\'', rotated_ws_reshaped.shape)

        if self._fullpadding:
            s_padded = F.pad(s, self.npads, "constant", 0)
            y = F.conv2d(s_padded, rotated_ws_reshaped, 
                        bias=self.bias.repeat(len(self.angles)), stride=self.stride, dilation=self.dilation)
        else:
            y = F.conv2d(s, rotated_ws_reshaped,
                        padding=(self._npads[2], self._npads[0]), 
                        bias=self.bias.repeat(len(self.angles)),
                        stride=self.stride, dilation=self.dilation)

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
        #print('y', y.shape)
        y = y.reshape(-1, # the batch dimension
                      self.num_filters,
                      len(self.angles),
                      self.output_dims[1], 
                      self.output_dims[2])
        # reshape y to have the orientiations in the last column
        y = y.permute(0, 1, 3, 4, 2)
        #print('y\'', y.shape)
        #print('=================')
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

        print('input_dims', input_dims)
        print('num_filters', num_filters)
        
        assert input_dims is not None, "ConvLayer3D: input_dims must be specified"
        assert num_filters is not None, "ConvLayer3D: num_filters must be specified"
        assert filter_width is not None, "ConvLayer3D: filter_width must be specified"

        full_filter_dims = [input_dims[0], filter_width, filter_width, 1]
        input_dims_2D = [input_dims[0], input_dims[1], input_dims[2], 1]
        super().__init__(input_dims_2D, num_filters, filter_dims=full_filter_dims, output_norm=output_norm, **kwargs)
        self.input_dims = input_dims
        self.output_dims = [num_filters, input_dims[1], input_dims[2], input_dims[3]]

        if self.res_layer:
            assert False, 'res_layer not implemented for ConvLayer3D'

    def forward(self, x):
        s = x.reshape([-1]+self.input_dims)
        #print('s', s.shape)

        w = self.preprocess_weights().reshape(self.filter_dims+[self.num_filters])
        #print('w', w.shape)

        if self._fullpadding:
            s = F.pad(s, self._npads, "constant", 0)
            #print('s\'', s.shape)
            y = F.conv3d(
                s, # we do our own padding
                w.permute(4,0,1,2,3), # num_filters is first
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation)
        else:
            # functional pads since padding is simple
            #print('_npads', self._npads)
            y = F.conv3d(
                s, 
                w.permute(4,0,1,2,3), # num_filters is first,
                padding=(self._npads[2], self._npads[0], 0),
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation)
        
        #print('y0', y.shape)
        # Nonlinearity
        if self.NL is not None:
            y = self.NL(y)

        # EI mask
        if self._ei_mask is not None:
            #print('y', y.shape)
            #print('ei_mask', self._ei_mask.shape)
            #print('output_dims', self.output_dims)
            # move the orientiation dimension after the filter dim
            #y = y.permute(0, 1, 4, 2, 3)
            # combine the filter and orientation dimensions
            #y = y.reshape(-1, self.num_filters, self.output_dims[1], self.output_dims[2])
            #print('y\'', y.shape)
            # we just want to multiply the second dimension here
            y = y * self._ei_mask[None, :, None, None, None]
            # split the filter and orientation dimensions again
            #y = y.reshape(-1, self.num_filters, self.output_dims[3], self.output_dims[1], self.output_dims[2])
            #y = y.permute(0, 1, 3, 4, 2)
            #print('y', y.shape)

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
    def layer_dict(cls, filter_width, **kwargs):
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

