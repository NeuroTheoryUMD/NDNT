import torch
import torch.nn as nn
from .ndnlayer import NDNLayer
import numpy as np
from torch.nn import functional as F
from torch.nn import Parameter
from .specialtylayers import Tlayer

class NonConvlayer(Tlayer):
    def __init__(self, input_dims=None,
        num_filters=None,
        filter_dims=None,
        num_bank_filters = None,
        filter_bank_width = None,
        NLtype:str='lin',
        pos_constraint=0,
        num_inh:int=0,
        bias:bool=False,
        weights_initializer:str='xavier_uniform',
        #output_norm=None,
        initialize_center=False,
        bias_initializer:str='zeros',
        reg_vals:dict=None,
        frac = 1,
        crt = False,
        tent_spacing = 1,
        num_lags = 1,
        sw_pos = False,
        transform = False,
        **kwargs,
        ):
            super().__init__(input_dims=input_dims,num_filters=num_filters, reg_vals=reg_vals,
                              initialize_center=initialize_center, weights_initializer=weights_initializer, num_inh = num_inh,
                              NLtype=NLtype, pos_constraint=pos_constraint, bias=bias, num_lags=num_lags,
                              bias_initializer=bias_initializer, tent_spacing = tent_spacing, **kwargs)
            self.frac = frac #upsample
            self.crt = crt #delta function stimulus or not
            # self.locs = Parameter(torch.zeros((num_filters, 2))) #x,y (or possibly y,x) locations for each output filter
            if input_dims[2] == 1: #1d space
                self.sdim = 1
            else: #2d space
                self.sdim = 2
            
            self.locs = Parameter(torch.zeros((num_bank_filters, num_filters, self.sdim))) #x,y (or possibly y,x) locations for each output filter
            self.set_parameters(val=False, name='locs')
            self.sw_pos = sw_pos
            self.transform = transform

            self.num_lags = num_lags

            #creating grid for grid sample in preprocess weights
            x = torch.linspace(-1,1,input_dims[1]) #-1 to 1 spanning the final filter size
            d = torch.linspace(-1,1, num_filters) #this dimension is going to be the different output filters of the layer
            #this is hopefully to make the processing faster, but the tradeoff is more memory storing weights temporarily, can experiment with this later
            
            if self.sdim == 2:
                self.filter_bank_dims = [1, filter_bank_width, filter_bank_width, self.num_lags] #individual filter dimensions in the filter bank

                gridx, gridy, gridz = torch.meshgrid(d,x,x, indexing='ij') #make a grid of previous linspace values
                grid = torch.stack([gridx,gridy,gridz], dim=-1).repeat(num_bank_filters,1,1,1,1) #make a grid for each filter bank filter
            else:
                self.filter_bank_dims = [1, filter_bank_width, 1, self.num_lags] #individual filter dimensions in the filter bank

                gridx, gridz = torch.meshgrid(d,x, indexing='ij') #make a grid of previous linspace values
                grid = torch.stack([gridx,gridz], dim=-1).repeat(num_bank_filters,1,1,1) #make a grid for each filter bank filter

            self.grid = grid
            self.transform_matrix = Parameter(torch.eye(self.sdim).unsqueeze(0).repeat(num_filters,1,1))
            self.filter_bank_shape = tuple([np.prod(self.filter_bank_dims), num_bank_filters]) #shape the weights are stored in

            self.filter_bank_weight_scale = np.sqrt(self.filter_bank_shape[0]) / 100
            self.weight = Parameter(torch.Tensor(size=self.filter_bank_shape))

            self.super_weight = Parameter(torch.zeros((num_bank_filters, num_filters)))

            
            if not self.transform:
                self.set_parameters(name='transform_matrix', val = False)

            #Weights/locs initialization
            self.reset_parameters( weights_initializer, bias_initializer )
            if initialize_center:
                self.initialize_gaussian_envelope() # this doesnt work for now




    @classmethod
    def layer_dict(cls, frac = 1, num_bank_filters = 5, filter_bank_width = 11, crt = False, sw_pos = False, transform = False, **kwargs):
        """
        This outputs a dictionary of parameters that need to input into the layer to completely specify.
        Output is a dictionary with these keywords. 
        -- All layer-specific inputs are included in the returned dict
        -- Values that must be set are set to empty lists
        -- Other values will be given their defaults
        """

        Ldict = super().layer_dict(**kwargs)
        # Added arguments
        Ldict['layer_type'] = 'NonConv_layer'
        Ldict['frac'] = frac
        Ldict['crt'] = crt
        Ldict['num_bank_filters'] = num_bank_filters
        Ldict['filter_bank_width'] = filter_bank_width
        Ldict['sw_pos'] = sw_pos
        Ldict['transform'] = transform



        return Ldict


    def preprocess_weights(self):
        """
        Preprocess the weights before using them in the forward pass.

        Returns:
            w: torch.Tensor, preprocessed weights
        """
        # Apply positive constraints
        if self.pos_constraint > 0:
            w = torch.square(self.weight) # to promote continuous gradients around 0
            #w = self.weight.clamp(min=0)
        elif self.pos_constraint < 0:
            w = -torch.square(self.weight)  # to promote continuous gradients around 0
            #w = self.weight.clamp(max=0)
            # note this is instead of w = self.weight.clamp(min=0)
        else:
            w = self.weight


        if self.norm_type == 1: # so far just standard filter-specific normalization
            w = F.normalize( w, dim=0 ) / self.filter_bank_weight_scale

        w = w.reshape(self.filter_bank_dims[1:] + [self.filter_bank_shape[-1]]).permute(3,2,0,1) # width, height, lags, fb_num --> fb_num, lags, width, height
        if self.sdim == 2:
            w = w.unsqueeze(-1).repeat(1,1,1,1,self.num_filters) #make copy of weights for each output filter fb_num, lags, width, height, f_num
        else:
            w = w.repeat(1,1,1,self.num_filters) #make copy of weights for each output filter fb_num, lags, width, f_num


        locs_scale = 2/(self.input_dims[1]-1) #conversion from units of pixels in the final filter to affine grid units
        # locs = torch.einsum('ijk,ik->ij',self.transform_matrix,self.locs*locs_scale)
        locs = torch.einsum('ijk,lik->lij',self.transform_matrix,self.locs*locs_scale)
        # locs = locs.flip(-1)

        grid = (self.grid).to(self.weight.device)

        if self.sdim == 2:
            T = F.pad(self.transform_matrix,(0,1),mode='constant')

            # T[:,0,2] = T[:,0,0]*locs[:,0] #scale loc numbers so that they don't depend on the scaling of the filter
            # T[:,1,2] = T[:,1,1]*locs[:,1]

            a_grid = torch.nn.functional.affine_grid(T, size = [self.num_filters, self.num_lags, self.input_dims[1],self.input_dims[1]], align_corners=True)
            grid[...,1:] = a_grid.unsqueeze(0).repeat(self.filter_bank_shape[-1],1,1,1,1) - locs[:,:,None,None,:].repeat(1,1,self.filter_dims[1],self.filter_dims[1],1)
        else:
            # print(grid[...,1].shape)
            # print(self.transform_matrix.shape)
            # print(torch.einsum('ijk,lim -> lim',self.transform_matrix,grid[...,1]).shape)
            # print((locs[:,:,None,:].repeat(1,1,self.filter_dims[1],1)).shape)
            grid_temp = torch.einsum('ijk,lim -> lim',self.transform_matrix,grid[...,1]) - locs[:,:,:].repeat(1,1,self.filter_dims[1])
            grid = torch.stack([grid[...,0], grid_temp], dim = -1)
        
        grid[...,1:] = (self.input_dims[1]//2)/(self.filter_bank_dims[1]//2)*grid[...,1:]

        sw = self.super_weight
        if self.sw_pos == True:
            sw = torch.square(sw) #enforce positive constraints on this one to make sub filters more interpretable
        w = torch.einsum('im,i...m->i...m', (sw, w)) #scale each filter according to its weight

        w = torch.sum(torch.nn.functional.grid_sample(w, grid, align_corners=True, mode='bilinear'), 0) #placing all the filters and summing them together

        if self.sdim == 2:
            w = w.permute(2,3,0,1).reshape(self.shape) #reshape into expected shape
        else:
            w = w.permute(1,0,2).reshape(self.shape) #reshape into expected shape

        return w

    def forward(self,x):
        y = super().forward(x)

        if self.frac > 1:
            # if self.crt:
            #     z = torch.zeros_like(y)
            #     z = z.repeat_interleave(self.frac, dim=0)
            #     z[::self.frac] = y
            #     y = z
            # else:
            #     z = y.repeat_interleave(self.frac, dim=0)
            #     y = z
            y = y.unsqueeze(0).permute(0,2,1) # interpolate ignores first two dimensions
            y = F.interpolate(y,scale_factor = self.frac, mode='linear',align_corners=True)
            y = y.permute(0,2,1).squeeze(0)

        return y
    
