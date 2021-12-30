
#%%
import sys
sys.path.insert(0, '/home/jake/Data/Repos/')

import torch
import numpy as np

from NDNT.modules.layers.convlayers import ConvLayer, TconvLayer, STconvLayer
import torch.utils.benchmark as benchmark

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%

print(" Conv Layer Test") 
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

conv.to(device)
x = x.to(device)

y = conv(x)
out_shape = y.shape
print("'same' padding -> output shape {}".format(out_shape))

y = y.reshape([-1] + conv.output_dims)
print("'same' padding -> output true shape {}".format(y.shape))

# test that when padding is "same" the output shape is the same as the input shape with no lags
output_dims = [batch_size, num_filters, input_dims[1], input_dims[2], 1]
assert list(y.shape)==output_dims, "output shape {} does not match expected {}".format(y.shape, output_dims)
print("'same' padding test passed")

t0 = benchmark.Timer(
    stmt='f(x)',
    globals={'x': x, 'f': conv})

print(t0.timeit(10))

# Valid padding
conv = ConvLayer(input_dims=input_dims,
                    num_filters=num_filters,
                    padding='valid',
                    conv_dims=conv_dims)

conv.to(device)

y = conv(x)
out_shape = y.shape
print("'valid' padding -> output shape {}".format(out_shape))

y = y.reshape([-1] + conv.output_dims)
print("'valid' padding -> output true shape {}".format(y.shape))

# test that when padding is "same" the output shape is the same as the input shape with no lags
output_dims = [batch_size, num_filters, input_dims[1]-conv_dims[0]+1, input_dims[2]-conv_dims[1]+1, 1]
assert list(y.shape)==output_dims, "output shape {} does not match expected {}".format(y.shape, output_dims)
print("'valid' padding test passed")

t0 = benchmark.Timer(
    stmt='f(x)',
    globals={'x': x, 'f': conv})

print(t0.timeit(10))

# %%
print("Conv Layer with Temporal Tent Spacing")
num_lags = 10
conv_dims = [15,15,num_lags]
conv = ConvLayer(input_dims=input_dims,
                    num_filters=num_filters,
                    padding='valid',
                    temporal_tent_spacing=2,
                    conv_dims=conv_dims)

conv.to(device)

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

t0 = benchmark.Timer(
    stmt='f(x)',
    globals={'x': x, 'f': conv})

print(t0.timeit(10))

# %% Tconv
print("Tconv Layer Test (tent spacing on)")
padding = 'valid'
print("padding: {}".format(padding))

conv_dims = [15,15,9]
conv = TconvLayer(input_dims=input_dims,
                    num_filters=num_filters,
                    padding=padding,
                    temporal_tent_spacing=2,
                    conv_dims=conv_dims)
conv.to(device)

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

t0 = benchmark.Timer(
    stmt='f(x)',
    globals={'x': x, 'f': conv})

t0.timeit(10)

# %% STconv

input_dims = [1, 30, 30, 10]
x = torch.randn([batch_size, input_dims[1], input_dims[2]])
x = x.to(device)

padding = 'valid'
conv_dims = [15,15,input_dims[3]]
conv = STconvLayer(input_dims=input_dims,
                    num_filters=num_filters,
                    padding=padding,
                    temporal_tent_spacing=2,
                    conv_dims=conv_dims)
conv.to(device)

y = conv(x.flatten(start_dim=1))

out_shape = y.shape
print("STconv -> output shape {}".format(out_shape))

y = y.reshape([-1] + conv.output_dims)
print("STconv -> output true shape {}".format(y.shape))

# test that when padding is "same" the output shape is the same as the input shape with no lags
if padding == 'same':
    output_dims = [batch_size, num_filters, input_dims[1], input_dims[2], 1]
else:
    output_dims = [batch_size, num_filters, input_dims[1]-conv.filter_dims[1]+1, input_dims[2]-conv.filter_dims[2]+1, 1]

assert list(y.shape)==output_dims, "output shape {} does not match expected {}".format(y.shape, output_dims)
print("STconv test passed")

t0 = benchmark.Timer(
    stmt='f(x)',
    globals={'x': x, 'f': conv})

print(t0.timeit(10))

# %%
