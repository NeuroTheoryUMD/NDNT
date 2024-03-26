import sys
sys.path.append('../../') # for importing from NDNT

import os
import torch

# NDN tools
import NDNT.utils as utils # some other utilities
import NDNT.NDN as NDN
from NDNT.modules.layers import *
from NDNT.networks import *


def make_complex_network():
    stim_dims = [1, 60, 60, 1]
    NCv = 10
    NT = 1
    NA = 44

    # some good starting parameters
    Treg = 0.01
    Xreg = None
    Mreg = 0.0001
    Creg = None
    Dreg = 0.5

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
    lgn_layer['reg_vals'] = {'d2x':Xreg,
                            'd2t':Treg,
                            'center': Creg, # None
                            'edge_t':100} # just pushes the edge to be sharper

    proj_layer = ConvLayer.layer_dict(
        num_filters=32,
        bias=False,
        norm_type=1,
        num_inh=32//2,
        filter_dims=17,
        NLtype='relu',
        initialize_center=True)
    #proj_layer['output_norm']='batch'
    proj_layer['window']='hamming'
    #proj_layer['reg_vals'] = {'center': 0.0001}

    conv_layer0 = ConvLayer.layer_dict(
        num_filters=32,
        num_inh=32//2,
        bias=False,
        norm_type=1,
        filter_dims=17,
        NLtype='relu',
        initialize_center=False,
        angles=[90, 180, 270])
    conv_layer0['output_norm'] = 'batch'

    scaffold_net = FFnetwork.ffnet_dict(
        ffnet_type='scaffold',
        xstim_n='stim',
        layer_list=[lgn_layer, proj_layer, conv_layer0],
        scaffold_levels=[1,2],
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

    ffnet_list = [scaffold_net, readout_net, drift_net, comb_net]

    cnn = NDN.NDN(ffnet_list=ffnet_list)
    cnn.block_sample = True

    # need to call this to set up the regularization
    # this is what 'fit()' normally does
    cnn.prepare_regularization()

    return cnn, ffnet_list


def test_save_model_zip():
    cnn, _ = make_complex_network()
    cnn.save_model_zip('test.zip')
    cnn_loaded = NDN.NDN.load_model_zip('test.zip')

    # compare weights between the two models within some tolerance
    for i in range(len(cnn.networks)):
        for j in range(len(cnn.networks[i].layers)):
            if hasattr(cnn.networks[i].layers[j], 'weight'):
                assert torch.allclose(cnn.networks[i].layers[j].weight, cnn_loaded.networks[i].layers[j].weight, atol=1e-8)

    # clean up
    os.remove('test.zip')

def test_save_model_zip_with_params():
    cnn, ffnet_list = make_complex_network()
    cnn.save_model_zip('test.zip', ffnet_list=ffnet_list)
    cnn_loaded = NDN.NDN.load_model_zip('test.zip')

    # compare weights between the two models within some tolerance
    for i in range(len(cnn.networks)):
        for j in range(len(cnn.networks[i].layers)):
            if hasattr(cnn.networks[i].layers[j], 'weight'):
                assert torch.allclose(cnn.networks[i].layers[j].weight, cnn_loaded.networks[i].layers[j].weight, atol=1e-8)

    # clean up
    os.remove('test.zip')
