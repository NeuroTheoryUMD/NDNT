
#%%
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import torch

from NDNT.modules.layers.convlayers import NDNLayer

#%%
input_dims = [1, 5, 5, 4] # C, H, W, T
batch_size = 1000

reg_types = ['d2x', 'd2t', 'l1', 'l2', 'local', 'orth', 'norm2', 'glocalx', 'glocalt', 'center']

for reg_type in reg_types:
    print(reg_type)

    layer = NDNLayer(input_dims=input_dims, num_filters=22,
        NLtype='relu', norm_type=2,
        reg_vals={reg_type:0.1})

    x = torch.randn([batch_size] + input_dims).flatten(start_dim=1)
    layer.reg.build_reg_modules()

    y = layer(x)

    layer.reg.set_reg_val(reg_type, .001)
    layer.reg.build_reg_modules()
    rpen = layer.compute_reg_loss()

    print(rpen)
# %%
