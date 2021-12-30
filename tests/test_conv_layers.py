
#%%
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import torch
import numpy as np

from NDNT.modules.layers.convlayers import ConvLayer, TconvLayer, STconvLayer



# %%
num_lags = 10
input_dims = [1, 30, 30, num_lags] # C, H, W, T
batch_size = 1000
conv_dims = [15,15,num_lags]
num_filters = 10

conv = ConvLayer(input_dims=input_dims,
                    num_filters=num_filters,
                    padding='same',
                    conv_dims=conv_dims)


x = torch.randn([batch_size] + input_dims)

y = conv(x)
out_shape = y.shape
print("'same' padding -> output shape {}".format(out_shape))

y = y.reshape([-1] + conv.output_dims)
print("'same' padding -> output true shape {}".format(y.shape))

# test that when padding is "same" the output shape is the same as the input shape with no lags
output_dims = [batch_size, num_filters, input_dims[1], input_dims[2], 1]
assert list(y.shape)==output_dims, "output shape {} does not match expected {}".format(y.shape, output_dims)
print("'same' padding test passed")


# Valid padding
conv = ConvLayer(input_dims=input_dims,
                    num_filters=num_filters,
                    padding='valid',
                    conv_dims=conv_dims)

y = conv(x)
out_shape = y.shape
print("'valid' padding -> output shape {}".format(out_shape))

y = y.reshape([-1] + conv.output_dims)
print("'valid' padding -> output true shape {}".format(y.shape))

# test that when padding is "same" the output shape is the same as the input shape with no lags
output_dims = [batch_size, num_filters, input_dims[1]-conv_dims[0]+1, input_dims[2]-conv_dims[1]+1, 1]
assert list(y.shape)==output_dims, "output shape {} does not match expected {}".format(y.shape, output_dims)
print("'valid' padding test passed")
# %%
num_lags = 10
conv_dims = [15,15,num_lags]
conv = ConvLayer(input_dims=input_dims,
                    num_filters=num_filters,
                    padding='valid',
                    temporal_tent_spacing=2,
                    conv_dims=conv_dims)

w = conv.preprocess_weights()
w.reshape(conv.filter_dims + [conv.num_filters]).shape

conv.weight.shape

w = conv.get_weights()
w.shape

y = conv(x)
out_shape = y.shape
print("tent_spacing=2 -> output shape {}".format(out_shape))

y = y.reshape([-1] + conv.output_dims)
print("tent_spacing=2 -> output true shape {}".format(y.shape))

# test that when padding is "same" the output shape is the same as the input shape with no lags
output_dims = [batch_size, num_filters, input_dims[1]-conv_dims[0]+1, input_dims[2]-conv_dims[1]+1, 1]
assert list(y.shape)==output_dims, "output shape {} does not match expected {}".format(y.shape, output_dims)
print("tent_spacing=2 test passed")
# %%
padding = 'valid'
conv_dims = [15,15,9]
conv = TconvLayer(input_dims=input_dims,
                    num_filters=num_filters,
                    padding=padding,
                    temporal_tent_spacing=2,
                    conv_dims=conv_dims)

y = conv(x.flatten(start_dim=1))

out_shape = y.shape
print("Tconv -> output shape {}".format(out_shape))

y = y.reshape([-1] + conv.output_dims)
print("Tconv=2 -> output true shape {}".format(y.shape))

# test that when padding is "same" the output shape is the same as the input shape with no lags
if padding == 'same':
    output_dims = [batch_size, num_filters, input_dims[1], input_dims[2], input_dims[3]]
else:
    output_dims = [batch_size, num_filters, input_dims[1]-conv.filter_dims[1]+1, input_dims[2]-conv.filter_dims[2]+1, input_dims[3]-conv.filter_dims[3]+1]

assert list(y.shape)==output_dims, "output shape {} does not match expected {}".format(y.shape, output_dims)
print("Tconv test passed")

# %%
