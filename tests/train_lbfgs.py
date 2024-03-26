# #%% Import Libraries

# import sys
# import os

# # setup paths
# iteration = 1 # which version of this tutorial to run (in case want results in different dirs)
# NBname = 'example_Monocular{}'.format(iteration)

# myhost = os.uname()[1] # get name of machine
# print("Running on Computer: [%s]" %myhost)

# if myhost=='mt': # this is sigur
#     sys.path.insert(0, '/home/jake/Repos/')
#     dirname = os.path.join('.', 'checkpoints')
# elif myhost=='bancanus': # this is jake's workstation
#     sys.path.insert(0, '/home/jake/Data/Repos/')
#     datadir = '/home/jake/Data/Datasets/Cumming/Monocular/'
#     dirname = os.path.join('.', 'checkpoints')
# else:
#     sys.path.insert(0, '/home/dbutts/Code/') # you need this Repo, NDN3, and V1FreeViewingCode
#     datadir = './'  # the datadir is part of the repository in this tutorial, but can be somewhere else
#     # Working directory -- this determines where models and checkpoints are saved
#     dirname = '/home/dbutts/V1/Monocular/'

# # pytorch
# import numpy as np

# # plotting
# import matplotlib.pyplot as plt

# # Import torch
# import torch
# import torch.nn.functional as F
# from torch import nn

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# dtype = torch.float32

# # Where saved models and checkpoints go -- this is to be automated
# print( 'Save_dir =', dirname)

# %load_ext autoreload
# %autoreload 2
# %aimport
# # %%
# # 1D Dataset from Cumming lab
# from datasets.cumming.monocular import MonocularDataset

# data_dir = '/home/jake/Datasets/Cumming/Monocular'

# sessname = 'expt04'
# train_ds = MonocularDataset(sessname=sessname, dirname=data_dir, stimset='Train')
# test_ds = MonocularDataset(sessname=sessname, dirname=data_dir, stimset='Test')

# #%% test load
# model_name = 'glm1'
# idtag = NBname + '/' + model_name # what to call these data
# print(idtag)

# #%% Test GLM
# from NDNT import NDN
# from NDNT import utils

# #%%
# dims = [train_ds.NF, train_ds.NX, train_ds.NY, train_ds.num_lags]
# NC = train_ds.NC

# glm_ffnet = utils.dicts.ffnet_dict_NIM(
#     input_dims = dims, layer_sizes = [NC], act_funcs = ['softplus'],
#     reg_list={'d2xt':[0.001], 'l1':[1e-4]})

# opt_pars = utils.create_optimizer_params(
#     learning_rate=.01, early_stopping_patience=4,
#     weight_decay = 0.01) # high initial learning rate because we decay on plateau)

# glm0 = NDN( ffnet_list= [glm_ffnet],
#     model_name=idtag, optimizer_params=opt_pars,
#     data_dir=dirname)

# #%% Train Full Batch LBFGS
# ver = 1 # none will auto number versions

# opt_pars['optimizer'] = 'LBFGS'
# opt_pars['max_iter'] = 10 # iterations per epoch
# opt_pars['batch_size'] = 1000
# opt_pars['max_epochs'] = 100
# opt_pars['early_stopping_patience'] = 1

# glm0.fit( dataset=train_ds,
#     version=ver,
#     opt_params=opt_pars,
#     device=device,
#     accumulated_grad_batches=1,
#     full_batch=True,
#     log_activations=False)

# LLs0 = glm0.eval_models(test_ds[:], data_inds=None, bits=False)
# print(LLs0)

# #%% Train again with Stochastic LBFGS
# glm1 = NDN( ffnet_list= [glm_ffnet],
#     model_name=idtag, optimizer_params=opt_pars,
#     data_dir=dirname)

# ver += 1

# glm1.fit( dataset=train_ds,
#     version=ver,
#     opt_params=opt_pars,
#     device=device,
#     accumulated_grad_batches=20,
#     full_batch=False,
#     log_activations=False)

# LLs1= glm1.eval_models(test_ds[:], data_inds=None, bits=False)
# print(LLs0)
# print(LLs1)

# #%% Compare
# plt.figure()
# plt.plot(LLs0, LLs1, '.')
# plt.plot(plt.xlim(), plt.xlim(), 'k')
# plt.title('GLM (convex)')
# plt.xlabel("Full Batch (deterministic)")
# plt.ylabel("Stochastic LBFGS")
# plt.show()

# #%% Develoment of LBFGS training

# # reload model
# glm0 = NDN( ffnet_list= [glm_ffnet],
#     model_name=idtag, optimizer_params=opt_pars,
#     data_dir=dirname)

# # get dataloaders
# train_dl, valid_dl = glm0.get_dataloaders(train_ds, batch_size=100, num_workers=8)

# from torch.optim import LBFGS
# optimizer = LBFGS(glm0.parameters(), max_iter=10, history_size=7)

# # prepare NDN model for fitting
# glm0.prepare_regularization()
# avRs = glm0.compute_average_responses(train_ds)
# glm0.loss_module.set_unit_normalization(avRs) 

# #%% Version 1: minibatch LBFGS

# for epoch in range(10):
    
#     for data in train_dl:

#         def closure():
        
#             optimizer.zero_grad()

#             out = glm0.training_step(data)
                    
#             loss = out['loss']
#             loss.backward()       

#             return loss

#     loss = optimizer.step(closure)

#     print("Epoch {}: loss={}".format(epoch, loss.detach().item()))

# #%% Version 2: full batch LBFGS
# for epoch in range(2):
#     optimizer.zero_grad()

#     def closure():
        
#         optimizer.zero_grad()

#         loss = torch.zeros(1)

#         for data in train_dl:
#             out = glm0.training_step(data)
                
#             loss += out['loss']

#         loss/=len(train_dl)

#         loss.backward()       

#         return loss

#     loss = optimizer.step(closure)

#     print("Epoch {}: loss={}".format(epoch, loss.detach().item()))

# #%% Version 3: accumulated gradient LBFGS (in between the two)

# accumulated_grad_batches = 20
# n_batches = len(train_dl)
# for epoch in range(20):
#     optimizer.zero_grad()

#     def closure():
        
#         optimizer.zero_grad()

#         loss = torch.zeros(1)

#         for batch_idx, data in enumerate(train_dl):

#             if batch_idx < accumulated_grad_batches:
#                 out = glm0.training_step(data)
#                 loss += out['loss']
#             else:
#                 break

#         loss/=accumulated_grad_batches

#         loss.backward()

#         return loss

#     loss = optimizer.step(closure)

#     print("Epoch {}: loss={}".format(epoch, loss.detach().item()))


    

# #%% Plot filters
# ws = glm0.get_weights(ffnet_target=0, layer_target=0, to_reshape=True)
# fig = plt.figure()
# fig.set_size_inches(16, 4)
# for nn in range(10):
#     plt.subplot(2,5,nn+1)
#     plt.imshow(ws[:,:,nn].T, cmap='gray')
# plt.tight_layout()
# plt.show()

# # %% Load model: test it produces same filters and likelihoods
# '''
# Note: this might not match if the best model saved with a different
# version number than what was just trained
# '''
# glm1 = NDN.load_model(dirname, idtag)

# # likelihoods
# LLs1 = glm1.eval_models(test_ds[:], null_adjusted=True)

# assert np.allclose(LLs0, LLs1), "Likelihoods from loaded model does not match"

# # filters
# ws = glm1.get_weights(to_reshape=True)
# fig = plt.figure()
# fig.set_size_inches(16, 4)
# for nn in range(10):
#     plt.subplot(2,5,nn+1)
#     plt.imshow(ws[:,:,nn].T, cmap='gray')
# plt.tight_layout()
# plt.show()

# # %% 2. NIM
# '''
# ===============================================================================
# Model 2: NIM

# '''

# num_subunits = 12
# snim_ffnet = utils.dicts.ffnet_dict_NIM(
#     input_dims = dims, layer_sizes = [num_subunits, NC],
#     act_funcs = ['relu', 'softplus'],
#     reg_list={'d2xt':[0.001], 'l1':[1e-5, 1e-6]})

# model_name = 'snim0'
# idtag = NBname + '/' + model_name # what to call these data
# print(idtag)

# opt_pars = utils.create_optimizer_params(
#     learning_rate=.01, early_stopping_patience=4,
#     weight_decay = 0.01)

# nim0 = NDN( ffnet_list= [snim_ffnet],
#     model_name=idtag,
#     optimizer_params=opt_pars,
#     data_dir=dirname)

# #%% Train
# ver = None
# nim0.fit( dataset=train_ds, version=ver, name=idtag )
# LLs1 = nim0.eval_models(test_ds[:], null_adjusted=True)
# print(np.mean(LLs1)) 

# #%% plot Likelihoods
# plt.figure(figsize=(8,4))
# plt.plot(np.ones(NC), LLs1-LLs0, 'o', label='NIM')
# plt.xlim((0,3))
# plt.axhline(0, color='k')
# plt.xlabel('Model')
# plt.ylabel('LL difference')
# plt.show()
# #%% plot filters
# ws = nim0.get_weights(to_reshape=True)
# fig = plt.figure()
# fig.set_size_inches(16, 4)
# for nn in range(12):
#     plt.subplot(2,6,nn+1)
#     plt.imshow(ws[:,:,nn].T, cmap='gray')
# plt.tight_layout()
# plt.show()

# plt.figure()
# ws = nim0.get_weights(ffnet_target=0, layer_target=1, to_reshape=True)
# plt.imshow(ws)
# plt.title("Layer 2")
# plt.xlabel("Neuron #")
# plt.ylabel("Subunit #")
# plt.show()

# #%% CONV NIM
# '''
# ===============================================================================
# Model 3: Convolutional NIM

# '''
# num_subunits = 12
# fw = 15 # filter width
# cnim_ffnet = NDNT.utils.dicts.ffnet_dict_NIM(
#     input_dims = dims, layer_sizes = [num_subunits, NC], 
#     layer_types=['conv', 'normal'], conv_widths=[fw,None],
#     act_funcs = ['relu', 'softplus'],
#     reg_list={'d2xt':[0.00001, 0.1], 'l1':[1e-4, 1e-3], 'center':[10]})
# #glm_ffnet

# model_name = 'cnim0'
# idtag = NBname + '/' + model_name # what to call these data
# print(idtag)

# opt_pars = NDNT.utils.create_optimizer_params(
#     learning_rate=.01, early_stopping_patience=4,
#     weight_decay = 0.1) # high initial learning rate because we decay on plateau)

# cnim0 = NDNT.NDN( ffnet_list= [cnim_ffnet],
#     model_name=idtag,
#     optimizer_params=opt_pars,
#     data_dir=dirname)

# #%% Train
# ver = None
# cnim0.fit( dataset=train_ds, version=ver, name=idtag )
# LLs1c = cnim0.eval_models(sample=test_ds[:], null_adjusted=True)
# print(np.mean(LLs1c)) 

# #%% plot filters
# fw = 15
# ws = cnim0.get_weights(to_reshape=True)
# fig = plt.figure()
# fig.set_size_inches(16, 4)
# for nn in range(12):
#     plt.subplot(2,6,nn+1)
#     plt.imshow(ws[:,:,nn].T, cmap='gray')
# plt.tight_layout()
# plt.show()


# #%% Point Readout
# '''
# ===============================================================================
# Model 3: Convolutional NIM

# '''
# num_subunits = 12
# fw = 15
# core_par = Dicts.ffnet_dict_NIM(
#     input_dims = dims, layer_sizes = [num_subunits], 
#     layer_types=['conv'], conv_widths=[fw], act_funcs = ['relu'],
#     reg_list={'d2xt':[0.00001], 'l1':[1e-4, 1e-3], 'center':[0.1]})
# readout_par = Dicts.ffnet_dict_readout(ffnet_n=0, num_cells=NC)
# #glm_ffnet

# model_name = 'cr0'
# idtag = NBname + '/' + model_name # what to call these data
# print(idtag)

# opt_pars = NDNutils.create_optimizer_params(
#     learning_rate=.01, early_stopping_patience=4,
#     weight_decay = 0.1) # high initial learning rate because we decay on plateau)


# #cr0 = NDN.NDN( ffnet_list= [core_par, readout_par], model_name=idtag, optimizer_params=opt_pars, ffnet_out=-1)
# cr0 = NDN.NDN( ffnet_list= [core_par, readout_par],
#     model_name=idtag,
#     optimizer_params=opt_pars,
#     data_dir=dirname)

# #%% Train
# ver = None
# cr0.fit( dataset=train_ds, version=ver, name=idtag )
# LLs1d = cr0.eval_models(sample=test_ds[:], null_adjusted=True)
# print(np.mean(LLs1d)) 

# #%% plot filters
# ws = cr0.get_weights(ffnet_target=0, layer_target=0)
# fig = plt.figure()
# fig.set_size_inches(16, 4)
# for nn in range(12):
#     plt.subplot(2,6,nn+1)
#     plt.imshow(ws[:,:,nn].T, cmap='gray')
#     #plt.imshow(ws[:,nn].reshape([fw,num_lags]).T, cmap='gray')
# plt.tight_layout()
# plt.show()
 
# #%% plot all model comparisons
# plt.figure(figsize=(8,4))
# plt.plot(np.ones(NC), LLs1-LLs0, 'o', label='NIM')
# plt.plot(2*np.ones(NC), LLs1c-LLs0, 'o', label='Conv NIM')
# plt.plot(3*np.ones(NC), LLs1d-LLs0, 'o', label='Point Readout')
# plt.xlim((0,5))
# plt.axhline(0, color='k')
# plt.xlabel('Model')
# plt.ylabel('LL difference')
# plt.legend()
