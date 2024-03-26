# '''
# This script will test standalone NDN without lightning.
# '''
# #%% Import Libraries
# import sys
# import os

# # setup paths
# iteration = 1 # which version of this tutorial to run (in case want results in different dirs)
# NBname = 'pytorch_testing{}'.format(iteration)

# myhost = os.uname()[1] # get name of machine
# print("Running on Computer: [%s]" %myhost)

# if myhost=='mt': # this is sigur
#     sys.path.insert(0, '/home/jake/Repos/')
#     datadir = '/home/dbutts/V1/Monocular/Mdata/'
#     dirname = os.path.join('.', 'checkpoints', NBname)
# elif myhost=='bancanus': # this is jake's workstation
#     sys.path.insert(0, '/home/jake/Data/Repos/')
#     datadir = '/home/jake/Data/Datasets/Cumming/Monocular/'
#     dirname = os.path.join('.', 'checkpoints', NBname)
# else:
#     sys.path.insert(0, '/home/dbutts/Code/') # you need this Repo, NDN3, and V1FreeViewingCode
#     datadir = './'  # the datadir is part of the repository in this tutorial, but can be somewhere else
#     # Working directory -- this determines where models and checkpoints are saved
#     dirname = '/home/dbutts/V1/Monocular/'




# # loading data
# import scipy.io as sio
# from copy import deepcopy

# # pytorch
# import numpy as np

# # plotting
# import matplotlib.pyplot as plt

# # Import torch
# import torch
# import torch.nn.functional as F
# from torch import nn

# # pytorch data management
# from torch.utils.data import DataLoader

# # NDN tools
# import datasets_dab as dataset  # some utilities
# import NDNutils as NDNutils # some other utilities
# import NDNtorch as NDN

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dtype = torch.float32

# # Where saved models and checkpoints go -- this is to be automated
# print( 'Data_dir =', datadir)
# print( 'Save_dir =', dirname)  # this will be set by default

# #%% Data loading functions

# # Define data-helpers
# def time_in_blocks(block_inds):
#     num_blocks = block_inds.shape[0]
#     #print( "%d number of blocks." %num_blocks)
#     NT = 0
#     for nn in range(num_blocks):
#         NT += block_inds[nn,1]-block_inds[nn,0]+1
#     return NT


# def make_block_inds( block_lims, gap=20, separate = False):
#     block_inds = []
#     for nn in range(block_lims.shape[0]):
#         if separate:
#             block_inds.append(np.arange(block_lims[nn,0]-1+gap, block_lims[nn,1]), dtype='int')
#         else:
#             block_inds = np.concatenate( 
#                 (block_inds, np.arange(block_lims[nn,0]-1+gap, block_lims[nn,1], dtype='int')), axis=0)
#     return block_inds


# def monocular_data_import( datadir, exptn, num_lags=20 ):
#     """Usage: stim, Robs, DFs, used_inds, Eadd_info = binocular_data_import( datadir, expt_num )
#             Note: expt num is starting with 1
#             block_output determines whether to fit using block info, or used_inds info (derived from blocks)
#         """
    
#     time_shift = 1
#     l_files_to_use = np.add(list(range(16)), 1)  # good laminar-probe experiments
#     u_files_to_use = [1, 2, 5, 6, 12]  # good utah-array experiments
#     assert exptn <= len(l_files_to_use)+len(u_files_to_use), 'Expt number too high.'
#     if exptn <= len(l_files_to_use):
#         filename = 'expt'
#         ee = l_files_to_use[exptn-1]
#         is_utah = False
#     else:
#         filename = 'Uexpt'
#         #utah_array[nn] = True
#         ee = u_files_to_use[exptn-1 - len(l_files_to_use)]
#         is_utah = True
#     if ee < 10:
#         filename += '0'
#     filename += str(ee) + '.mat'         
#     matdata = sio.loadmat( datadir+filename )

#     sus = matdata['goodSUs'][:,0] - 1  # switch from matlab indexing
#     print('SUs:', sus)
#     NC = len(sus)
#     layers = matdata['layers'][0,:]
#     block_list = matdata['block_inds'] # note matlab indexing
#     stim_all = NDNutils.shift_mat_zpad(matdata['stimulus'], time_shift, 0)
#     NTtot, NX = stim_all.shape
#     DFs_all = deepcopy(matdata['data_filters'][:,sus])
#     Robs_all = deepcopy(matdata['binned_SU'][:,sus])
    
#     # Break up into train and test blocks
#     # Assemble train and test indices based on BIlist
#     NBL = block_list.shape[0]
#     Xb = np.arange(2, NBL, 5)  # Every fifth trial is cross-validation
#     Ub = np.array(list(set(list(range(NBL)))-set(Xb)), dtype='int')
    
#     used_inds = make_block_inds( block_list, gap=num_lags )
#     Ui, Xi = NDNutils.generate_xv_folds( len(used_inds) )
#     TRinds, TEinds = used_inds[Ui].astype(int), used_inds[Xi].astype(int)

#     Eadd_info = {
#         'cortical_layer':layers, 'used_inds': used_inds, 
#         'TRinds':TRinds, 'TEinds': TEinds, #'TRinds': Ui, 'TEinds': Xi, 
#         'block_list': block_list, 'TRblocks': Ub, 'TEblocks': Xb}
#     return stim_all, Robs_all, DFs_all, Eadd_info

# # %% Load data
# ee = 4
# num_lags = 16

# stim, Robs, DFs, Eadd_info = monocular_data_import( datadir, ee, num_lags=num_lags )
# NX = stim.shape[1]
# Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, 1])
# NT, NC = Robs.shape
# # For index parsing
# used_inds = Eadd_info['used_inds']
# Ui, Xi = Eadd_info['TRinds'], Eadd_info['TEinds']
# print( "%d SUs, %d / %d used time points"%(NC, len(used_inds), NT) )


# # # Jake torch models expect [samples x Lags x Xspace x Yspace]
# # # if there is no Yspace, leave it off, but it's no longer a 2D model and we'll have to implement 1D convolutions.
# # # For now, let's fit a model with no convolutions
# # x2 = np.reshape(Xstim, (NT, NX, num_lags))
# # x2 = np.transpose(x2, (0, 2, 1))
# # x2 = np.expand_dims(x2, axis=3) # Y-spatial dim is 1


# # Now make pytorch dataset
# train_ds = dataset.generic_recording(Xstim[Ui,:], Robs[Ui,:], DFs[Ui,:], device=None)
# test_ds = dataset.generic_recording(Xstim[Xi,:], Robs[Xi,:], DFs[Xi,:], device=None)

# sample = train_ds[:10]
# print(sample['stim'].shape, sample['robs'].shape, sample['dfs'].shape)


# #%% test load
# model_name = 'glm0'
# idtag = NBname + '/' + model_name # what to call these data
# print(idtag)

# # glm0 = NDN.NDN.load_model('./checkpoints', idtag)
# #%% Test GLM
# glm_ffnet = NDNutils.ffnet_dict_NIM(
#     input_dims = [1, NX, 1, num_lags], layer_sizes = [NC], act_funcs = ['softplus'],
#     reg_list={'d2xt':[0.001], 'l1':[1e-4]})

# opt_pars = NDNutils.create_optimizer_params(
#     learning_rate=.01, early_stopping_patience=10,
#     weight_decay = 0.01) # high initial learning rate because we decay on plateau)

# glm0 = NDN.NDN( ffnet_list= [glm_ffnet], model_name=idtag, optimizer_params=opt_pars)

# #%% Does it run?
# sample = train_ds[:10]
# out = glm0(sample['stim'])
# print(out.shape)

# #%% Train
# ver = None # none will auto number versions
# glm0.fit( dataset=train_ds, version=ver)


# #%% Plot filters
# ws = glm0.networks[0].layers[0].weights.detach().cpu().numpy()
# fig = plt.figure()
# fig.set_size_inches(16, 4)
# for nn in range(10):
#     plt.subplot(2,5,nn+1)
#     plt.imshow(ws[:,nn].reshape([NX,num_lags]).T, cmap='gray')
# plt.tight_layout()
# plt.show()

# #%% Evaluate model
# LLs0 = glm0.eval_models(sample=test_ds[:], null_adjusted=True)
# print(LLs0)
# # %%
# glm1 = NDN.NDN.load_model('./checkpoints', idtag)
# # %%
# LLs0 = glm1.eval_models(sample=test_ds[:], null_adjusted=True)
# print(LLs0)

# # %%
# ws = glm1.networks[0].layers[0].weights.detach().cpu().numpy()
# fig = plt.figure()
# fig.set_size_inches(16, 4)
# for nn in range(10):
#     plt.subplot(2,5,nn+1)
#     plt.imshow(ws[:,nn].reshape([NX,num_lags]).T, cmap='gray')
# plt.tight_layout()
# plt.show()

# # %%
