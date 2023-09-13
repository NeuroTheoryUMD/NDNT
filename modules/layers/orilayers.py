from numpy.lib.arraysetops import isin
import torch
from torch.nn import functional as F
import torch.nn as nn

from .ndnlayer import NDNLayer
from .convlayers import ConvLayer
import numpy as np

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

        self.is1D = (self.input_dims[2] == 1)
        assert not self.is1D, "OriConvLayer: Stimulus must be 2-D"

        print('input_dims', self.input_dims)

        self.angles = angles

        # repeat the bias for each orientation
        self.bias = nn.Parameter(self.bias.repeat(len(self.angles)))

        # make the rotation matrices and store them as a buffer
        rotation_matrices = self.rotation_matrix_tensor(self.filter_dims, self.angles)
        self.register_buffer('rotation_matrices', rotation_matrices)

        # make the ei mask and store it as a buffer,
        # repeat it for each orientation (plus one for the original orientation)
        self.register_buffer('_ei_mask',
                             torch.cat(
                                 (torch.ones(self.num_filters-self._num_inh), 
                                  -torch.ones(self._num_inh))
                             ).repeat(len(self.angles)))
        
        print('ei_mask', self._ei_mask.shape)

        # folded_dims is num_channels * num_angles
        self.folded_dims = self.num_filters*(len(self.angles))
        self.output_dims[3] = len(self.angles)

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
    # END OriConvLayer.rotation_matrix_tensor

    def forward(self, x):
        print()
        print('==== FORWARD ====')
        print('input dims', self.input_dims)
        print('x', x.shape)
        w = self.preprocess_weights().reshape(self.filter_dims+[self.num_filters]).permute(4, 0, 3, 1, 2)
        print('w', w.shape)
        w_flattened = w.reshape(self.num_filters, self.filter_dims[1]*self.filter_dims[2]) 
        print('w\'', w_flattened.shape)
        s = x.reshape(-1, self.input_dims[0], self.input_dims[1], self.input_dims[2])
        print('s', s.shape)

        rotated_ws = w_flattened.repeat(1, 1, len(self.angles)).reshape(
            self.folded_dims,    # num_filters * num_angles
            self.filter_dims[1]*self.filter_dims[2]) # width*height
        print('wr', rotated_ws.shape)
        
        # use torch.sparse.mm to multiply the rotation matrices by the weights
        for i in range(len(self.angles)):
            # get a view of the weights for the given angle
            w_theta = rotated_ws[i*self.num_filters:(i+1)*self.num_filters, :]
            print('w_theta', w_theta.shape)

            # rotate the weight matrix for the given angle
            print('rotation_matrix', self.rotation_matrices[i].shape)
            # we need to transpose w_theta to multiple with the rotation matrix,
            # then transpose it back
            w_theta = torch.sparse.mm(self.rotation_matrices[i], w_theta.T).T
            print('w_theta\'', w_theta.shape)

            # put w_theta back into the full weight matrix
            rotated_ws[i*self.num_filters:(i+1)*self.num_filters, :] = w_theta
            print('rotated_ws\'', rotated_ws.shape)
        
        # TODO: update to use the right batch size
        # TODO: we hack to make put folded_dims in the batch dimension
        #       this lets us convolve all the filters at once
        #       but we need to reshape the output to have the folded_dims in the channel dimension again
        rotated_ws_reshaped = rotated_ws.reshape((self.folded_dims, -1, self.filter_dims[1], self.filter_dims[2]))
        print('wr\'', rotated_ws_reshaped.shape)

        if self._fullpadding:
            s_padded = F.pad(s, self.npads, "constant", 0)
            y = F.conv2d(s_padded, rotated_ws_reshaped, 
                        bias=self.bias, stride=self.stride, dilation=self.dilation)
        else:
            y = F.conv2d(s, rotated_ws_reshaped,
                        padding=(self._npads[2], self._npads[0]), 
                        bias=self.bias,
                        stride=self.stride, dilation=self.dilation)
        print('y', y.shape)

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

        # reshape y to have the orientiations in the last column
        print('y', y.shape)
        y = y.reshape(-1, len(self.angles), self.num_filters,
                             self.output_dims[1], self.output_dims[2]).permute(0, 2, 3, 4, 1)
        print('y\'', y.shape)
        print('=================')
        return y.reshape(-1, self.num_filters*(len(self.angles))*self.output_dims[1]*self.output_dims[2])
    # OriConvLayer.forward

    @classmethod
    def layer_dict(cls, angles=None, **kwargs):
        Ldict = super().layer_dict(**kwargs)
        Ldict["layer_type"]="oriconv"
        Ldict["angles"]=angles
        return Ldict
