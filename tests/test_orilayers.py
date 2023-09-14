import sys
sys.path.append('../../') # to have access to NDNT

import torch
import NDNT.utils as utils # some other utilities
import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *

stim_width = 4
stim_height = 4
stim_dims = [1,stim_width,stim_height,1]
NCv = 7
NA = 44 # drift terms
# some good starting parameters
Treg = 0.01
Xreg = None
Mreg = 0.0001
Creg = None
Dreg = 0.5


def construct_test_stim(batch_size=1, all_ones=False, stim_dims=stim_dims):
    stim_width = stim_dims[1]
    stim_height = stim_dims[2]
    if all_ones:
        stim = torch.ones(batch_size, stim_width*stim_height)
    else:
        stim = torch.FloatTensor(batch_size, stim_width*stim_height).uniform_(-1,1)
    return stim

def construct_test_Xdrift(batch_size=1):
    Xdrift = torch.FloatTensor(batch_size, 44).uniform_(0,1)
    return Xdrift

def construct_test_data(batch_size=1, stim_dims=stim_dims):
    stim = construct_test_stim(batch_size=batch_size, stim_dims=stim_dims)
    Xdrift = construct_test_Xdrift(batch_size=batch_size)
    data = {'stim': stim, 'Xdrift': Xdrift}
    return data


def test_layer_forward_2_angle_1_filter_1_batch():
    # get the test data
    stim = construct_test_stim(batch_size=1, all_ones=True)

    # define a basic layer and confirm that data passes through it
    oriconv_layer = OriConvLayer.layer_dict(
        input_dims = stim_dims,
        num_filters=1,
        bias=False,
        norm_type=0,
        filter_dims=3,
        NLtype='relu',
        initialize_center=True,
        angles=[0, 90])

    cnn = NDN.NDN(layer_list=[oriconv_layer],
                    loss_type='poisson')
    cnn.block_sample = True

    # set the weights to be a linspace
    print('weight', cnn.networks[0].layers[0].weight.shape)
    weight = cnn.networks[0].layers[0].weight
    ascending_weights = torch.arange(0,9, dtype=torch.float32)
    print('ascending_weights', ascending_weights)
    cnn.networks[0].layers[0].weight.data = ascending_weights.reshape(1,9).T
    print('weight', cnn.networks[0].layers[0].weight)

    # run the data through the layer
    output = cnn.networks[0](stim)
    print('num outputs', cnn.networks[0].layers[0].num_outputs, np.prod(cnn.networks[0].layers[0].output_dims))
    assert output.shape == (1, 2*stim_width*stim_height) # 1 filter x 2 angles x 9 stim_dims
    output = output.reshape(1,1,stim_width,stim_height,2).permute(4,0,1,2,3)
    print(output)
    assert torch.allclose(output, torch.tensor(
                          [[[[[24., 33., 33., 20.],
                            [27., 36., 36., 21.],
                            [27., 36., 36., 21.],
                            [12., 15., 15.,  8.]]]],
                            [[[[20., 21., 21.,  8.],
                            [33., 36., 36., 15.],
                            [33., 36., 36., 15.],
                            [24., 27., 27., 12.]]]]]))


def test_layer_forward_4_angle_1_filter_1_batch():
    # get the test data
    stim = construct_test_stim(batch_size=1, all_ones=True)

    # define a basic layer and confirm that data passes through it
    oriconv_layer = OriConvLayer.layer_dict(
        input_dims = stim_dims,
        num_filters=1,
        bias=False,
        norm_type=0,
        filter_dims=3,
        NLtype='relu',
        initialize_center=True,
        angles=[0, 90, 180, 270])

    cnn = NDN.NDN(layer_list=[oriconv_layer],
                    loss_type='poisson')
    cnn.block_sample = True

    # set the weights to be a linspace
    print('weight', cnn.networks[0].layers[0].weight.shape)
    weight = cnn.networks[0].layers[0].weight
    ascending_weights = torch.arange(0,9, dtype=torch.float32)
    print('ascending_weights', ascending_weights)
    cnn.networks[0].layers[0].weight.data = ascending_weights.reshape(1,9).T
    print('weight', cnn.networks[0].layers[0].weight)

    # run the data through the layer
    output = cnn.networks[0](stim)
    assert output.shape == (1, 4*stim_width*stim_height) # 1 filter x 2 angles x 9 stim_dims
    output = output.reshape(1,1,stim_width,stim_height,4).permute(4,0,1,2,3)
    print(output)
    expected = torch.tensor(
        [[[[[24., 33., 33., 20.],
           [27., 36., 36., 21.],
           [27., 36., 36., 21.],
           [12., 15., 15.,  8.]]]],
        [[[[20., 21., 21.,  8.],
           [33., 36., 36., 15.],
           [33., 36., 36., 15.],
           [24., 27., 27., 12.]]]],
        [[[[ 8., 15., 15., 12.],
           [21., 36., 36., 27.],
           [21., 36., 36., 27.],
           [20., 33., 33., 24.]]]],
        [[[[12., 27., 27., 24.],
           [15., 36., 36., 33.],
           [15., 36., 36., 33.],
           [ 8., 21., 21., 20.]]]]])
    print(output.shape, expected.shape)
    assert torch.allclose(output, expected)


def test_layer_forward_2_angle_2_filter_1_batch():
    # get the test data
    stim = construct_test_stim(batch_size=1, all_ones=True)

    # define a basic layer and confirm that data passes through it
    oriconv_layer = OriConvLayer.layer_dict(
        input_dims = stim_dims,
        num_filters=2,
        bias=False,
        norm_type=0,
        filter_dims=3,
        NLtype='relu',
        initialize_center=True,
        angles=[0, 90])

    cnn = NDN.NDN(layer_list=[oriconv_layer],
                    loss_type='poisson')
    cnn.block_sample = True

    # set the weights to be a linspace
    print('weight', cnn.networks[0].layers[0].weight.shape)
    ascending_weights = torch.arange(0,9, dtype=torch.float32)
    # repeat the weights for each filter
    ascending_weights = ascending_weights.repeat(2)
    print('ascending_weights', ascending_weights)
    cnn.networks[0].layers[0].weight.data = ascending_weights.reshape(2,9).T
    print('weight', cnn.networks[0].layers[0].weight)

    # run the data through the layer
    output = cnn.networks[0](stim)
    assert output.shape == (1, 2*2*stim_width*stim_height) # 2 filters x 2 angles x 9 stim_dims
    # reshape and put the filters first and angles second (batch is size 1)
    output = output.reshape(1,2,stim_width,stim_height,2).permute(1,4,0,2,3)
    print(output)
    assert torch.allclose(output, torch.tensor(
                          [[[[[24., 33., 33., 20.],
           [27., 36., 36., 21.],
           [27., 36., 36., 21.],
           [12., 15., 15.,  8.]]],
         [[[20., 21., 21.,  8.],
           [33., 36., 36., 15.],
           [33., 36., 36., 15.],
           [24., 27., 27., 12.]]]],
        [[[[24., 33., 33., 20.],
           [27., 36., 36., 21.],
           [27., 36., 36., 21.],
           [12., 15., 15.,  8.]]],
         [[[20., 21., 21.,  8.],
           [33., 36., 36., 15.],
           [33., 36., 36., 15.],
           [24., 27., 27., 12.]]]]]))


def test_layer_forward_2_angle_1_filter_5_batch():
    # get the test data
    stim = construct_test_stim(batch_size=5, all_ones=True)

    # define a basic layer and confirm that data passes through it
    oriconv_layer = OriConvLayer.layer_dict(
        input_dims = stim_dims,
        num_filters=1,
        bias=False,
        norm_type=0,
        filter_dims=3,
        NLtype='relu',
        initialize_center=True,
        angles=[0, 90])

    cnn = NDN.NDN(layer_list=[oriconv_layer],
                    loss_type='poisson')
    cnn.block_sample = True

    # set the weights to be a linspace
    print('weight', cnn.networks[0].layers[0].weight.shape)
    weight = cnn.networks[0].layers[0].weight
    ascending_weights = torch.arange(0,9, dtype=torch.float32)
    print('ascending_weights', ascending_weights)
    cnn.networks[0].layers[0].weight.data = ascending_weights.reshape(1,9).T
    print('weight', cnn.networks[0].layers[0].weight)

    # run the data through the layer
    output = cnn.networks[0](stim)
    assert output.shape == (5, 2*stim_width*stim_height) # 1 filter x 2 angles x 9 stim_dims
    output = output.reshape(5,1,stim_width,stim_height,2).permute(4,0,1,2,3)
    print(output)
    assert torch.allclose(output, torch.tensor(
                          [[[[[24., 33., 33., 20.],
           [27., 36., 36., 21.],
           [27., 36., 36., 21.],
           [12., 15., 15.,  8.]]],
         [[[24., 33., 33., 20.],
           [27., 36., 36., 21.],
           [27., 36., 36., 21.],
           [12., 15., 15.,  8.]]],
         [[[24., 33., 33., 20.],
           [27., 36., 36., 21.],
           [27., 36., 36., 21.],
           [12., 15., 15.,  8.]]],
         [[[24., 33., 33., 20.],
           [27., 36., 36., 21.],
           [27., 36., 36., 21.],
           [12., 15., 15.,  8.]]],
         [[[24., 33., 33., 20.],
           [27., 36., 36., 21.],
           [27., 36., 36., 21.],
           [12., 15., 15.,  8.]]]],
        [[[[20., 21., 21.,  8.],
           [33., 36., 36., 15.],
           [33., 36., 36., 15.],
           [24., 27., 27., 12.]]],
         [[[20., 21., 21.,  8.],
           [33., 36., 36., 15.],
           [33., 36., 36., 15.],
           [24., 27., 27., 12.]]],
         [[[20., 21., 21.,  8.],
           [33., 36., 36., 15.],
           [33., 36., 36., 15.],
           [24., 27., 27., 12.]]],
         [[[20., 21., 21.,  8.],
           [33., 36., 36., 15.],
           [33., 36., 36., 15.],
           [24., 27., 27., 12.]]],
         [[[20., 21., 21.,  8.],
           [33., 36., 36., 15.],
           [33., 36., 36., 15.],
           [24., 27., 27., 12.]]]]]))


def test_layer_forward_3_angle_4_filter_2_batch():
    # get the test data
    stim = construct_test_stim(batch_size=2, all_ones=True)

    # define a basic layer and confirm that data passes through it
    oriconv_layer = OriConvLayer.layer_dict(
        input_dims = stim_dims,
        num_filters=4,
        bias=False,
        norm_type=0,
        filter_dims=3,
        NLtype='relu',
        initialize_center=True,
        angles=[0, 90, 180])

    cnn = NDN.NDN(layer_list=[oriconv_layer],
                    loss_type='poisson')
    cnn.block_sample = True

    # set the weights to be a linspace
    print('weight', cnn.networks[0].layers[0].weight.shape)
    ascending_weights = torch.arange(0,9, dtype=torch.float32)
    # repeat the weights for each filter
    ascending_weights = ascending_weights.repeat(4)
    print('ascending_weights', ascending_weights)
    cnn.networks[0].layers[0].weight.data = ascending_weights.reshape(4,9).T
    print('weight', cnn.networks[0].layers[0].weight)

    # run the data through the layer
    output = cnn.networks[0](stim)
    assert output.shape == (2, 4*3*stim_width*stim_height) # 5 filters x 3 angles x 9 stim_dims
    # reshape and put the filters first after the batch and angles second (batch is size 2)
    output = output.reshape(2,4,stim_width,stim_height,3).permute(0,1,4,2,3)
    print(output)
    assert torch.allclose(output, torch.tensor(
                          [[[[[24., 33., 33., 20.],
                              [27., 36., 36., 21.],
                              [27., 36., 36., 21.],
                              [12., 15., 15.,  8.]],
                             [[20., 21., 21.,  8.],
                              [33., 36., 36., 15.],
                              [33., 36., 36., 15.],
                              [24., 27., 27., 12.]],
                             [[ 8., 15., 15., 12.], 
                              [21., 36., 36., 27.],
                              [21., 36., 36., 27.],
                              [20., 33., 33., 24.]]],
                            [[[24., 33., 33., 20.],
                              [27., 36., 36., 21.],
                              [27., 36., 36., 21.],
                              [12., 15., 15.,  8.]],
                             [[20., 21., 21.,  8.],
                              [33., 36., 36., 15.],
                              [33., 36., 36., 15.],
                              [24., 27., 27., 12.]],
                             [[ 8., 15., 15., 12.],
                              [21., 36., 36., 27.],
                              [21., 36., 36., 27.],
                              [20., 33., 33., 24.]]],
                            [[[24., 33., 33., 20.],
                              [27., 36., 36., 21.],
                              [27., 36., 36., 21.],
                              [12., 15., 15.,  8.]],
                             [[20., 21., 21.,  8.],
                              [33., 36., 36., 15.],
                              [33., 36., 36., 15.],
                              [24., 27., 27., 12.]],
                             [[ 8., 15., 15., 12.],
                              [21., 36., 36., 27.],
                              [21., 36., 36., 27.],
                              [20., 33., 33., 24.]]],
                            [[[24., 33., 33., 20.],
                              [27., 36., 36., 21.],
                              [27., 36., 36., 21.],
                              [12., 15., 15.,  8.]],
                             [[20., 21., 21.,  8.],
                              [33., 36., 36., 15.],
                              [33., 36., 36., 15.],
                              [24., 27., 27., 12.]],
                             [[ 8., 15., 15., 12.],
                              [21., 36., 36., 27.],
                              [21., 36., 36., 27.],
                              [20., 33., 33., 24.]]]],
                           [[[[24., 33., 33., 20.],
                              [27., 36., 36., 21.],
                              [27., 36., 36., 21.],
                              [12., 15., 15.,  8.]],
                             [[20., 21., 21.,  8.],
                              [33., 36., 36., 15.],
                              [33., 36., 36., 15.],
                              [24., 27., 27., 12.]],
                             [[ 8., 15., 15., 12.],
                              [21., 36., 36., 27.],
                              [21., 36., 36., 27.],
                              [20., 33., 33., 24.]]],
                            [[[24., 33., 33., 20.],
                              [27., 36., 36., 21.],
                              [27., 36., 36., 21.],
                              [12., 15., 15.,  8.]],
                             [[20., 21., 21.,  8.],
                              [33., 36., 36., 15.],
                              [33., 36., 36., 15.],
                              [24., 27., 27., 12.]],
                             [[ 8., 15., 15., 12.],
                              [21., 36., 36., 27.],
                              [21., 36., 36., 27.],
                              [20., 33., 33., 24.]]],
                            [[[24., 33., 33., 20.],
                              [27., 36., 36., 21.],
                              [27., 36., 36., 21.],
                              [12., 15., 15.,  8.]],
                             [[20., 21., 21.,  8.],
                              [33., 36., 36., 15.],
                              [33., 36., 36., 15.],
                              [24., 27., 27., 12.]],
                             [[ 8., 15., 15., 12.],
                              [21., 36., 36., 27.],
                              [21., 36., 36., 27.],
                              [20., 33., 33., 24.]]],
                            [[[24., 33., 33., 20.],
                              [27., 36., 36., 21.],
                              [27., 36., 36., 21.],
                              [12., 15., 15.,  8.]],
                             [[20., 21., 21.,  8.],
                              [33., 36., 36., 15.],
                              [33., 36., 36., 15.],
                              [24., 27., 27., 12.]],
                             [[ 8., 15., 15., 12.],
                              [21., 36., 36., 27.],
                              [21., 36., 36., 27.],
                              [20., 33., 33., 24.]]]]]))


def test_network_forward_2_angle_5_filter_7_batch():
    # get the test data
    stim = construct_test_stim(batch_size=7, all_ones=True)

    lgn_layer = STconvLayer.layer_dict(
        input_dims = stim_dims,
        num_filters=4,
        num_inh=2,
        bias=False,
        norm_type=1,
        filter_dims=[1,  # channels
                     3,  # width
                     3,  # height
                     2], # lags
        NLtype='relu',
        initialize_center=True)
    lgn_layer['output_norm']='batch'
    lgn_layer['window']='hamming'
    lgn_layer['reg_vals'] = {'d2x': Xreg,
                            'd2t': Treg,
                            'center': Creg, # None
                            'edge_t': 100} # just pushes the edge to be sharper

    # define a basic layer and confirm that data passes through it
    oriconv_layer = OriConvLayer.layer_dict(
        num_filters=5,
        bias=False,
        norm_type=0,
        filter_dims=3,
        NLtype='relu',
        initialize_center=True,
        angles=[0, 90])

    cnn = NDN.NDN(layer_list=[lgn_layer, oriconv_layer],
                    loss_type='poisson')
    cnn.block_sample = True

    # set the weights to be a linspace
    print('weight', cnn.networks[0].layers[1].weight.shape)
    weight = cnn.networks[0].layers[1].weight
    ascending_weights = torch.arange(0,9, dtype=torch.float32)
    # repeat the weights for each filter
    ascending_weights = ascending_weights.repeat(5,4)
    print('ascending_weights', ascending_weights)
    cnn.networks[0].layers[1].weight.data = ascending_weights.reshape(5,36).T
    print('weight', cnn.networks[0].layers[1].weight)

    # run the data through the layer
    output = cnn.networks[0](stim)
    assert output.shape == (7, 5*2*stim_width*stim_height) # 5 filter x 2 angles x 9 stim_dims


def test_big_cnn_forward_smoke():
    stim_dims = [1, 60, 60, 1]
    data = construct_test_data(10, stim_dims=stim_dims)
    
    lgn_layer = STconvLayer.layer_dict(
        input_dims = stim_dims,
        num_filters=4,
        num_inh=2,
        bias=False,
        norm_type=1,
        filter_dims=[1,  # channels
                     9,  # width
                     9,  # height
                     14], # lags
        NLtype='relu',
        initialize_center=True)
    lgn_layer['output_norm']='batch'
    lgn_layer['window']='hamming'
    lgn_layer['reg_vals'] = {'d2x': Xreg,
                            'd2t': Treg,
                            'center': Creg, # None
                            'edge_t': 100} # just pushes the edge to be sharper

    proj_layer = OriConvLayer.layer_dict(
        num_filters=32,
        bias=False,
        norm_type=1,
        num_inh=32//2,
        filter_dims=17,
        NLtype='relu',
        initialize_center=True,
        angles=[0, 90, 180, 270])
    proj_layer['window']='hamming'

    # proj_layer = ConvLayer.layer_dict(
    #     num_filters=32,
    #     bias=False,
    #     norm_type=1,
    #     num_inh=32//2,
    #     filter_dims=17,
    #     NLtype='relu',
    #     initialize_center=True)
    # proj_layer['output_norm']='batch'
    # proj_layer['window']='hamming'
    # proj_layer['reg_vals'] = {'center': 0.0001}

    scaffold_net = FFnetwork.ffnet_dict(
        ffnet_type='scaffold',
        xstim_n='stim',
        layer_list=[lgn_layer, proj_layer],
        scaffold_levels=[1],
        num_lags_out=None)

    ## 1: READOUT
    # reads out from a specific location in the scaffold network
    # this location is specified by the mus
    readout_pars = ReadoutLayer.layer_dict(
        num_filters=NCv,
        NLtype='lin',
        bias=False,
        pos_constraint=True)
    # for defining how to sample from the mu (location) of the receptive field
    readout_pars['gauss_type'] = 'isotropic'
    readout_pars['reg_vals'] = {'max': Mreg}

    readout_net = FFnetwork.ffnet_dict(
        xstim_n = None,
        ffnet_n=[0],
        layer_list = [readout_pars],
        ffnet_type='readout')

    ## 2: DRIFT
    drift_pars = NDNLayer.layer_dict(
        input_dims=[1,1,1,NA],
        num_filters=NCv,
        bias=False,
        norm_type=0,
        NLtype='lin')
    drift_pars['reg_vals'] = {'d2t': Dreg}

    drift_net = FFnetwork.ffnet_dict(xstim_n = 'Xdrift', layer_list = [drift_pars])

    ## 3: COMB 
    comb_layer = ChannelLayer.layer_dict(
        num_filters=NCv,
        NLtype='softplus',
        bias=True)
    comb_layer['weights_initializer'] = 'ones'

    comb_net = FFnetwork.ffnet_dict(
        xstim_n = None,
        ffnet_n=[1,2],
        layer_list=[comb_layer],
        ffnet_type='add')

    cnn = NDN.NDN(ffnet_list = [scaffold_net, readout_net, drift_net, comb_net],
                    loss_type='poisson')
    cnn.block_sample = True

    ## Network 1: readout: fixed mus / sigmas
    cnn.networks[1].layers[0].sample = False
    # mus and sigmas are the centers and "widths" of the receptive field center to start at
    cnn.networks[1].set_parameters(val=False, name='mu')
    cnn.networks[1].set_parameters(val=False, name='sigma')

    ## Network 2: drift: not fit
    cnn.networks[2].set_parameters(val=False)

    ## Network 3: Comb
    cnn.networks[-1].set_parameters(val=False, name='weight')

    # test the model
    out = cnn(data)
    assert out.shape == (10, 7)