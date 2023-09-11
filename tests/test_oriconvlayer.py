import sys
sys.path.append('../../') # to have access to NDNT

import torch
import NDNT.utils as utils # some other utilities
import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *

stim_dims = [1,60,60,1]
NCv = 10
NA = 44 # drift terms
# some good starting parameters
Treg = 0.01
Xreg = None
Mreg = 0.0001
Creg = None
Dreg = 0.5


def construct_test_stim(batch_size=1):
    stim = torch.FloatTensor(batch_size, 3600).uniform_(-1,1)
    return stim

def construct_test_Xdrift(batch_size=1):
    Xdrift = torch.FloatTensor(batch_size, 44).uniform_(0,1)
    return Xdrift

def construct_test_data(batch_size=1):
    stim = construct_test_stim(batch_size=batch_size)
    Xdrift = construct_test_Xdrift(batch_size=batch_size)
    data = {'stim': stim, 'Xdrift': Xdrift}
    return data


def test_layer_happypath():
    # TODO: test that the output weights are transformed correctly
    
    # get the test data
    stim = construct_test_stim()

    # define a basic layer and confirm that data passes through it
    oriconv_layer = OriConvLayer.layer_dict(
        input_dims = stim_dims,
        num_filters=32,
        bias=True,
        norm_type=1,
        num_inh=32//2,
        filter_dims=17,
        NLtype='relu',
        initialize_center=True,
        angles=[0, 90, 180, 270])
    oriconv_layer['window']='hamming'

    cnn = NDN.NDN(layer_list=[oriconv_layer],
                    loss_type='poisson')
    cnn.block_sample = True

    # run the data through the layer
    output = cnn.networks[0](stim)
    assert output.shape == (1, 576000) # 32 filters x 5 angles x 3600 stim_dims


def test_combined_network():
    data = construct_test_data()
    
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
    assert out.shape == (1, 10)