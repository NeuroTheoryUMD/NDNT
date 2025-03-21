### NDNTorchUtils.py ####
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import os
import shutil
from collections import OrderedDict
# for unpickler
import pickle
import json
import io


########### LBFGS TRAINER (just a function!) #################
def fit_lbfgs(mod, data, #val_data=None,
              parameters=None,
              optimizer=None,
              verbose=True,
              max_iter=1000,
              lr=1,
              line_search='strong_wolfe',
              history_size=100,
              tolerance_change=1e-7,
              tolerance_grad=1e-7):
    '''
    Runs fullbatch LBFGS on a Pytorch model and data dictionary
    Inputs:
        Model: Pytorch model
        data: Dictionary to used with Model.training_step(data)
    '''

    assert isinstance(data, dict), "data must be a dictionary"
    mod.prepare_regularization()
    mod.train()
    if parameters is None:
        parameters = mod.parameters()
    if optimizer is None:
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=lr, max_iter=max_iter,
            history_size=history_size,
            tolerance_change=tolerance_change,
            line_search_fn=line_search,
            tolerance_grad=tolerance_grad)
        
    def closure():
        optimizer.zero_grad()
        out = mod.training_step(data)
        loss = out['loss']
        if np.isnan(loss.item()):
            return loss
        if loss.requires_grad:
            loss.backward()
        if verbose > 0:
            print('Iteration: {} | Loss: {}'.format(optimizer.state_dict()['state'][0]['n_iter'], loss.cpu().item()))
        return loss
    
    #loss = optimizer.step(closure)
    try:
        loss = optimizer.step(closure)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    except TypeError:
        print("stopped early")

    optimizer.zero_grad()
    torch.cuda.empty_cache()
    #return loss
# END fit_lbfgs

def fit_lbfgs_batch(
        model, dataset,
        batch_size=1000,
        num_chunks=None,
        train_inds=None,
        device=None,
        verbose=True, 
        #val_dataset=None,
        #parameters=None, optimizer=None,
        max_iter=1000, lr=1,
        history_size=100,
        tolerance_change=1e-7,
        tolerance_grad=1e-7,
        line_search='strong_wolfe'):
    '''
    Runs LBFGS on a Pytorch model with batching a dataset into chunks. Both model and 
    data should be on the cpu, and it copies the model and individual batches to the GPU,
    accumulating full-dataset gradients for each epoch

    Inputs:
        Model: Pytorch model
        dataset: Dataset that can be sampled
        batch_size: size of chunk to sample; superceded by num_chunks
        num_chunks: divide dataset into number of chunks -> sets batch_size (def: None)
        train_inds: since passing in dataset, need to give indices to train over
        device: GPU device to copy everything to
        verbose: output to screen

    '''
    #from tqdm import tqdm
    from .DanUtils import chunker

    if train_inds is None:
        train_inds = np.arange(ds_size)
    ds_size = len(train_inds)

    if num_chunks is not None:
        batch_size = int(np.ceil(ds_size)/num_chunks)
        if verbose:
            print('Setting batch size to', batch_size)

    model.prepare_regularization()
    model.train()
    if device is not None:
        d0 = model.device
        model = model.to(device)
    #if parameters is None:
    #    parameters = model.parameters()
    #if optimizer is None:
    optimizer = torch.optim.LBFGS(
        model.parameters(), 
        lr=lr, history_size=history_size,
        tolerance_change=tolerance_change, tolerance_grad=tolerance_grad,
        max_iter=max_iter, line_search_fn=line_search)

    losses = [] #, vlosses = [], []
    #optimizer = torch.optim.LBFGS(model.model.net[1].parameters(), lr=1, max_iter=max_iter)
    patience = 200

    #best_model = model.state_dict()
    # Need to make iterable for tqqm
    def closure():
        #global patience
        #if patience < 0:
        #    print("patience exhausted")
        #    return
        optimizer.zero_grad()
        #for b in tqdm( chunker(range(ds_size), batch_size), position=0, leave=True ):
        loss = 0
        for b in chunker(range(ds_size), batch_size):
            data_chunk = dataset[train_inds[b]]
            # copy data to device
            if device is not None:
                for dsub in data_chunk:
                    data_chunk[dsub] = data_chunk[dsub].to(device)

            loss += model.training_step(data_chunk)['loss']*len(b)  # weights by number of time steps
        loss = loss/ds_size
        loss.backward()
            #del b
        losses.append(loss.item())
        #with torch.no_grad():
        #    model.eval()
            #vlosses.append(-eval_model_fast(model, val_dl).mean())
            #if val_dataset is not None:
            #    vlosses.append(model.eval_models(val_dataset))
            #else:
            #    vlosses.append(0)
        #    model.train()
        #    if min(vlosses) == vlosses[-1]:
        #        global best_model
        #        best_model = model.state_dict()
            #    patience = 200
            #else:
            #    patience -= 1
        if verbose:
            #print("loss:", losses[-1], "vloss:", vlosses[-1], "step:", len(losses))
            print("%5d:%12.7f"%(len(losses), losses[-1]))
        #pbar.update(1)
        return loss

    #with tqdm(total=max_iter) as pbar:
    try:
        optimizer.step(closure)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    except TypeError:
        print("stopped early")
    if device is not None:
        model = model.to(d0)
# END fit_lbfgs_batch

#################### CREATE OTHER PARAMETER-DICTS ####################
def create_optimizer_params(
        optimizer_type='AdamW',
        batch_size=1000,
        max_iter=None,
        max_epochs=None,
        num_workers=0,
        num_gpus=1,
        optimize_graph = True,
        log_activations = False,
        progress_bar_refresh=20, # num of batches 
        # AdamW specific
        early_stopping=True,
        early_stopping_patience=4,
        early_stopping_delta=0.0,
        weight_decay=0.01,
        learning_rate=1e-3,
        betas=[0.9, 0.999],
        auto_lr=False,
        # L-BFGS specific
        full_batch=True,
        tolerance_change=1e-8,
        tolerance_grad=1e-10,
        history_size=10,
        accumulated_grad_batches=1,
        device=None,
        line_search_fn=None):

    # Make optimizer params based on Adam or LBFGS, filling in chosen defaults
    if optimizer_type in ['LBFGS', 'lbfgs']:

        # L-BFGS specific defaults
        if max_iter is None:
            max_iter = 300
        if max_epochs is None:
            max_epochs = 1

        optpar = {
            'optimizer_type': 'LBFGS',
            'full_batch': full_batch,
            'batch_size': batch_size,
            'max_iter': max_iter,
            'max_epochs': max_epochs,
            #'max_eval': max_iter*2, # automatically defaults to 1.25xmax_iter, so easier to not set
            'progress_bar_refresh': progress_bar_refresh,
            'log_activations': log_activations, 
            'optimize_graph': optimize_graph,
            'num_workers': num_workers,
            'num_gpus': num_gpus,
            'history_size': history_size,
            'line_search_fn': line_search_fn,
            'tolerance_change': tolerance_change,
            'tolerance_grad': tolerance_grad,
            'accumulated_grad_batches': accumulated_grad_batches,
            'device': device,
            'early_stopping': False}

    else: # Assume some sort of adam / sgd with similar params/defaults
        if max_iter is None:
            max_iter = 10000

        if max_epochs is None:
            if early_stopping:
                max_epochs = 100
            else:
                max_epochs = 30
        

        optpar = {
            'optimizer_type': optimizer_type,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'early_stopping': early_stopping,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_delta': early_stopping_delta,
            'max_iter': max_iter,
            'max_epochs': max_epochs,
            'learning_rate': learning_rate,
            'betas': betas, 
            'amsgrad': False,
            'auto_lr': auto_lr,
            'num_workers': num_workers,
            'num_gpus': num_gpus,
            'progress_bar_refresh': progress_bar_refresh,
            'accumulated_grad_batches': accumulated_grad_batches,
            'log_activations': log_activations, 
            'device': device,
            'optimize_graph': optimize_graph} 
        if optimizer_type in ['sgd', 'SGD']:
            optpar['momentum'] = 0.9  # extra term for SGD -- although doesnt have the others

    return optpar
# END create_optimizer_params

######## PICKLE FIXER -- load to CPU automatically ######
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


#################### DIRECTORY ORGANIZATION ####################
def default_save_dir():
    savedir = './checkpoints/'


#################### GENERAL EMBEDDED UTILS ####################
def create_time_embedding(stim, num_lags, up_fac=1, tent_spacing=1):
    """
    Takes a Txd stimulus matrix and creates a time-embedded matrix of size 
    Tx(d*L), where L is the desired number of time lags, included as last (folded) dimension. 
    If stim is a 3d array, its dimensions are folded into the 2nd dimension. 
    
    Assumes zero-padding.
     
    Optional up-sampling of stimulus and tent-basis representation for filter 
    estimation.
    
    Note that xmatrix is formatted so that adjacent time lags are adjacent 
    within a time-slice of the xmatrix, thus x(t, 1:nLags) gives all time lags 
    of the first spatial pixel at time t.
    
    Args:
        stim (type): simulus matrix (time must be in the first dim).
        num_lags
        up_fac (type): description
        tent_spacing (type): description
        
    Returns:
        numpy array: time-embedded stim matrix
        
    """
    from scipy.linalg import toeplitz
    # Note for myself: pdims[0] is nLags and the rest is spatial dimension

    sz = list(np.shape(stim))

    # If there are two spatial dims, fold them into one
    if len(sz) > 2:
        stim = np.reshape(stim, (sz[0], np.prod(sz[1:])))
        print('Flattening stimulus to produce design matrix.')
    elif len(sz) == 1:
        stim = stim[:, None]
    sz = list(np.shape(stim))

    # No support for more than two spatial dimensions
    #if len(sz) > 3:
    #    print('More than two spatial dimensions not supported, but creating xmatrix anyways...')

    modstim = deepcopy(stim)
    # Up-sample stimulus if required
    if up_fac > 1:
        # Repeats the stimulus along the time dimension
        modstim = np.repeat(modstim, up_fac, 0)
        # Since we have a new value for time dimension
        sz = list(np.shape(modstim))

    # If using tent-basis representation
    if tent_spacing > 1:
        # Create a tent-basis (triangle) filter
        tent_filter = np.append(
            np.arange(1, tent_spacing) / tent_spacing,
            1-np.arange(tent_spacing)/tent_spacing) / tent_spacing
        # Apply to the stimulus
        filtered_stim = np.zeros(sz)
        for ii in range(len(tent_filter)):
            filtered_stim = filtered_stim + shift_mat_zpad(modstim, ii-tent_spacing+1, 0) * tent_filter[ii]
        modstim = filtered_stim

    sz = list(np.shape(modstim))
    lag_spacing = tent_spacing

    # If tent_spacing is not given in input then manually put lag_spacing = 1
    # For temporal-only stimuli (this method can be faster if you're not using
    # tent-basis rep)
    # For myself, add: & tent_spacing is empty (= & isempty...).
    # Since isempty(tent_spa...) is equivalent to its value being 1 I added
    # this condition to the if below temporarily:
    if sz[1] == 1 and tent_spacing == 1:
        xmat = toeplitz(np.reshape(modstim, (1, sz[0])),
                        np.concatenate((modstim[0], np.zeros(num_lags - 1)),
                                       axis=0))
    else:  # Otherwise loop over lags and manually shift the stim matrix
        xmat = np.zeros([sz[0], sz[1], num_lags])
        for lag in range(num_lags):
            for xx in range(sz[1]):
                xmat[:, xx, lag] = shift_mat_zpad(modstim[:, xx], lag_spacing * lag, 0)
        xmat = xmat.reshape([sz[0], -1])

    return xmat
# END create_time_embedding


def create_time_embedding_NIM(stim, pdims, up_fac=1, tent_spacing=1):
    """All the arguments starting with a p are part of params structure which I 
    will fix later.
    
    Takes a Txd stimulus matrix and creates a time-embedded matrix of size 
    Tx(d*L), where L is the desired number of time lags. If stim is a 3d array, 
    the spatial dimensions are folded into the 2nd dimension. 
    
    Assumes zero-padding.
     
    Optional up-sampling of stimulus and tent-basis representation for filter 
    estimation.
    
    Note that xmatrix is formatted so that adjacent time lags are adjacent 
    within a time-slice of the xmatrix, thus x(t, 1:nLags) gives all time lags 
    of the first spatial pixel at time t.
    
    Args:
        stim (type): simulus matrix (time must be in the first dim).
        pdims (list/array): length(3) list of stimulus dimensions
        up_fac (type): description
        tent_spacing (type): description
        
    Returns:
        numpy array: time-embedded stim matrix
        
    """
    from scipy.linalg import toeplitz
    # Note for myself: pdims[0] is nLags and the rest is spatial dimension

    sz = list(np.shape(stim))

    # If there are two spatial dims, fold them into one
    if len(sz) > 2:
        stim = np.reshape(stim, (sz[0], np.prod(sz[1:])))
        print('Flattening stimulus to produce design matrix.')
    elif len(sz) == 1:
        stim = np.expand_dims(stim, axis=1)
    sz = list(np.shape(stim))

    # No support for more than two spatial dimensions
    if len(sz) > 3:
        print('More than two spatial dimensions not supported, but creating xmatrix anyways...')

    # Check that the size of stim matches with the specified stim_params
    # structure
    if np.prod(pdims[1:]) != sz[1]:
        print('Stimulus dimension mismatch')
        raise ValueError

    modstim = deepcopy(stim)
    # Up-sample stimulus if required
    if up_fac > 1:
        # Repeats the stimulus along the time dimension
        modstim = np.repeat(modstim, up_fac, 0)
        # Since we have a new value for time dimension
        sz = list(np.shape(modstim))

    # If using tent-basis representation
    if tent_spacing > 1:
        # Create a tent-basis (triangle) filter
        tent_filter = np.append(
            np.arange(1, tent_spacing) / tent_spacing,
            1-np.arange(tent_spacing)/tent_spacing) / tent_spacing
        # Apply to the stimulus
        filtered_stim = np.zeros(sz)
        for ii in range(len(tent_filter)):
            filtered_stim = filtered_stim + \
                            shift_mat_zpad(modstim,
                                           ii-tent_spacing+1,
                                           0) * tent_filter[ii]
        modstim = filtered_stim

    sz = list(np.shape(modstim))
    lag_spacing = tent_spacing

    # If tent_spacing is not given in input then manually put lag_spacing = 1
    # For temporal-only stimuli (this method can be faster if you're not using
    # tent-basis rep)
    # For myself, add: & tent_spacing is empty (= & isempty...).
    # Since isempty(tent_spa...) is equivalent to its value being 1 I added
    # this condition to the if below temporarily:
    if sz[1] == 1 and tent_spacing == 1:
        xmat = toeplitz(np.reshape(modstim, (1, sz[0])),
                        np.concatenate((modstim[0], np.zeros(pdims[0] - 1)),
                                       axis=0))
    else:  # Otherwise loop over lags and manually shift the stim matrix
        xmat = np.zeros((sz[0], np.prod(pdims)))
        for lag in range(pdims[0]):
            for xx in range(0, sz[1]):
                xmat[:, xx*pdims[0]+lag] = shift_mat_zpad(
                    modstim[:, xx], lag_spacing * lag, 0)

    return xmat
# END create_time_embedding


def is_int( val ):
    """returns Boolean as to whether val is one of many types of integers"""
    if isinstance(val, int) or \
        isinstance(val, np.int32) or isinstance(val, np.int64) or \
        (isinstance(val, np.ndarray) and (len(val.shape) == 0)):
        return True
    else:
        return False


def design_matrix_tent_basis( s, anchors, zero_left=False, zero_right=False):
    """Produce a design matrix based on continuous data (s) and anchor points for a tent_basis.
    Here s is a continuous variable (e.g., a stimulus) that is function of time -- single dimension --
    and this will generate apply a tent basis set to s with a basis variable for each anchor point. 
    The end anchor points will be one-sided, but these can be dropped by changing "zero_left" and/or
    "zero_right" into "True".

    Inputs: 
        s: continuous one-dimensional variable with NT time points
        anchors: list or array of anchor points for tent-basis set
        zero_left, zero_right: boolean whether to drop the edge bases (default for both is False)
    Outputs:
        X: design matrix that will be NT x the number of anchors left after zeroing out left and right
    """

    if len(s.shape) > 1:
        assert s.shape[1] == 1, 'Can only work on 1-d variables currently'
        s = np.squeeze(s)

    NT = len(s)
    NA = len(anchors)
    X = np.zeros([NT, NA])
    for nn in range(NA):
        if nn == 0:
            #X[np.where(s < anchors[0])[0], 0] = 1
            X[:, 0] = 1
        else:
            dx = anchors[nn]-anchors[nn-1]
            X[:, nn] = np.minimum(np.maximum(np.divide( deepcopy(s)-anchors[nn-1], dx ), 0), 1)
        if nn < NA-1:
            dx = anchors[nn+1]-anchors[nn]
            X[:, nn] *= np.maximum(np.minimum(np.divide(np.add(-deepcopy(s), anchors[nn+1]), dx), 1), 0)
    if zero_left:
        X = X[:,1:]
    if zero_right:
        X = X[:,:-1]
    return X


def tent_basis_generate( xs=None, num_params=None, doubling_time=None, init_spacing=1, first_lag=0 ):
    """Computes tent-bases over the range of 'xs', with center points at each value of 'xs'.
    Alternatively (if xs=None), will generate a list with init_space and doubling_time up to
    the total number of parameters. Must specify xs OR num_params. 
    Note this assumes discrete (binned) variables to be acted on.
    
    Defaults:
        doubling_time = num_params
        init_space = 1"""

    # Determine anchor-points
    if xs is not None:
        tbx = np.array(xs,dtype='int32')
        if num_params is not None: 
            print( 'Warning: will only use xs input -- num_params is ignored.' )
    else:
        assert num_params is not None, 'Need to specify either xs or num_params'
        if doubling_time is None:
            doubling_time = num_params+1  # never doubles
        tbx = np.zeros( num_params, dtype='int32' )
        cur_loc, cur_spacing, sp_count = first_lag, init_spacing, 0
        for nn in range(num_params):
            tbx[nn] = cur_loc
            cur_loc += cur_spacing
            sp_count += 1
            if sp_count == doubling_time:
                sp_count = 0
                cur_spacing *= 2

    # Generate tent-basis given anchor points
    NB = len(tbx)
    NX = (np.max(tbx)+1).astype(int)
    tent_basis = np.zeros([NX,NB], dtype='float32')
    for nn in range(NB):
        if nn > 0:
            dx = tbx[nn]-tbx[nn-1]
            tent_basis[range(tbx[nn-1], tbx[nn]+1), nn] = np.array(list(range(dx+1)))/dx
        elif tbx[0] > 0:  # option to have function go to zero at beginning
            dx = tbx[0]
            tent_basis[range(tbx[nn]+1), nn] = np.array(list(range(dx+1)))/dx
        if nn < NB-1:
            dx = tbx[nn+1]-tbx[nn]
            tent_basis[range(tbx[nn], tbx[nn+1]+1), nn] = 1-np.array(list(range(dx+1)))/dx

    return tent_basis


def shift_mat_zpad(x, shift, dim=0):
    """Takes a vector or matrix and shifts it along dimension dim by amount 
    shift using zero-padding. Positive shifts move the matrix right or down.
    
    Args:
        x (type): description
        shift (type): description
        dim (type): description
        
    Returns:
        type: description
            
    Raises:
            
    """

    assert x.ndim < 3, 'only works in 2 dims or less at the moment.'
    if x.ndim == 1:
        oneDarray = True
        xcopy = np.zeros([len(x), 1])
        xcopy[:, 0] = x
    else:
        xcopy = deepcopy(x)
        oneDarray = False
    sz = list(np.shape(xcopy))

    if sz[0] == 1:
        dim = 1

    if dim == 0:
        if shift >= 0:
            a = np.zeros((shift, sz[1]))
            b = xcopy[0:sz[0]-shift, :]
            xshifted = np.concatenate((a, b), axis=dim)
        else:
            a = np.zeros((-shift, sz[1]))
            b = xcopy[-shift:, :]
            xshifted = np.concatenate((b, a), axis=dim)
    elif dim == 1:
        if shift >= 0:
            a = np.zeros((sz[0], shift))
            b = xcopy[:, 0:sz[1]-shift]
            xshifted = np.concatenate((a, b), axis=dim)
        else:
            a = np.zeros((sz[0], -shift))
            b = xcopy[:, -shift:]
            xshifted = np.concatenate((b, a), axis=dim)

    # If the shift in one direction is bigger than the size of the stimulus in
    # that direction return a zero matrix
    if (dim == 0 and abs(shift) > sz[0]) or (dim == 1 and abs(shift) > sz[1]):
        xshifted = np.zeros(sz)

    # Make into single-dimension if it started that way
    if oneDarray:
        xshifted = xshifted[:,0]

    return xshifted
# END shift_mat_zpad

def generate_xv_folds(nt, num_folds=5, num_blocks=3, which_fold=None):
    """Will generate unique and cross-validation indices, but subsample in each block
        NT = number of time steps
        num_folds = fraction of data (1/fold) to set aside for cross-validation
        which_fold = which fraction of data to set aside for cross-validation (default: middle of each block)
        num_blocks = how many blocks to sample fold validation from"""

    test_inds = []
    NTblock = np.floor(nt/num_blocks).astype(int)
    block_sizes = np.zeros(num_blocks, dtype='int32')
    block_sizes[range(num_blocks-1)] = NTblock
    block_sizes[num_blocks-1] = nt-(num_blocks-1)*NTblock

    if which_fold is None:
        which_fold = num_folds//2
    else:
        assert which_fold < num_folds, 'Must choose XV fold within num_folds =' + str(num_folds)

    # Pick XV indices for each block
    cnt = 0
    for bb in range(num_blocks):
        tstart = np.floor(block_sizes[bb] * (which_fold / num_folds))
        if which_fold < num_folds-1:
            tstop = np.floor(block_sizes[bb] * ((which_fold+1) / num_folds))
        else: 
            tstop = block_sizes[bb]

        test_inds = test_inds + list(range(int(cnt+tstart), int(cnt+tstop)))
        cnt = cnt + block_sizes[bb]

    test_inds = np.array(test_inds, dtype='int')
    train_inds = np.setdiff1d(np.arange(0, nt, 1), test_inds)

    return train_inds, test_inds


def initialize_gaussian_envelope( ws, w_shape):
    """
    This assumes a set of filters is passed in, and windows by Gaussian along each non-singleton dimension
    ws is all filters (ndims x nfilters)
    wshape is individual filter shape
    """
    ndims, nfilt = ws.shape
    assert np.prod(w_shape) == ndims
    wx = np.reshape(deepcopy(ws), w_shape + [nfilt])
    for dd in range(1,len(w_shape)):
        if w_shape[dd] > 1:
            L = w_shape[dd]
            genv = np.exp(-(np.arange(L)-L/2)**2/(2*(L/6)**2))
            if dd == 0:
                wx = np.einsum('abcde, a->abcde', wx, genv)
            elif dd == 1:
                wx = np.einsum('abcde, b->abcde', wx, genv)
            elif dd == 2:
                wx = np.einsum('abcde, c->abcde', wx, genv)
            else:
                wx = np.einsum('abcde, d->abcde', wx, genv)
    return np.reshape(wx, [-1, nfilt])

## CONVERSION BETWEEN grid and pixel coordinates for READOUT LAYERS
# Specifically grid_sample (align_corners=False) and other interpolation
def pixel2mu( p, L=60, flip_axes=True ):
    """
    Converts from pixel coordinates to mu values, used by grid_sample. The default will flip
    horizontal and vertical axes automatically, which is what grid_sample requires
    Pixels are starting with number 0 up to L-1, converted to range of -1 to 1
    
    Args:
        p: list of pixel locations, presumably num_locations x 2 (horizontal and vertical coords)
        L: size of grid (number of pixels), assuming square (default 60)
        flip_axes: whether to swap horizontal and vertical axes as grid_sample needs (default: True)
    
    Returns:
        x: mu values
    """
    x = (2*p+1)/L - 1
    if flip_axes:
        if len(x.shape) < 2:
            print("Warning: cannot flip axes, since only passed in one dimension")
        else:
            assert x.shape[1] == 2, "need 2 axes to flip them"
            x = np.roll(x, 1, axis=1)
    return x
# END pixel2mu()


def mu2pixel( x, L=60, force_int=True, enforce_bounds=True, flip_axes=True ):
    """
    Converts from mu values back into pixel coordinates, where mus are coordinates used by grid_sample.
    Can be continuous (fractional pixels), but default is rounding to nearest int

    Args: 
        x: mu values, assuming num_locations x 2 (although can be 1-dimensional)
        L: size of grid (number of pixels), assuming square (default 60)
        force_int: force to be integer (versus continuous-valued)
        enforce bounds: go to edges if answer ends up bigger than L-1 or smaller than 0
        flip_axes: whether to swap horizontal and vertical axes as grid_sample needs (default: True)

    Returns:
        p: pixel values
    """
    if flip_axes:
        if len(x.shape) < 2:
            print("Warning: cannot flip axes, since only passed in one dimension")
        else:
            assert x.shape[1] == 2, "need 2 axes to flip them"
            x = np.roll(x, 1, axis=1)

    p = (L*(x+1)-1)/2
    if enforce_bounds:
        a = np.where(p > L-1)[0]
        if len(a) > 0:   
            p[a] = L-1
        a = np.where(p < 0)[0]
        if len(a) > 0:
            p[a] = 0    
    if force_int:
        if isinstance(x, np.ndarray):
            return np.round(p).astype(np.int64)
        else:
            return int(np.round(p))
    else:
        return p
# END mu2pixel()


def pixel2grid( p, L=60 ):
    """
    Pixels are starting with number 0 up to L-1, converted to range of -1 to 1. This is the
    old function, to be replaced by pixel2mu
    """
    x = (2*p+1)/L - 1
    return x


def grid2pixel( x, L=60, force_int=True, enforce_bounds=False ):
    """Older function: replaced by mu2pixel"""
    p = (L*(x+1)-1)/2
    if enforce_bounds:
        a = np.where(p > L-1)[0]
        if len(a) > 0:   
            p[a] = L-1
        a = np.where(p < 0)[0]
        if len(a) > 0:
            p[a] = 0    
    if force_int:
        if isinstance(x, np.ndarray):
            return np.round(p).astype(np.int64)
        else:
            return int(np.round(p))
    else:
        return p


def set_scaffold_level_reg( ndn, reg_val=None, level_exponent=1.0, core_net=0, readout_net=1 ):
    """
    Sets up regularization for scaffold_level, which requires pulling scaffold structure from 
    core network and passing it into the readout layer at beginning of readout_net
    
    Args:
        ndn: model that has scaffold and readout ffnetworks
        reg_val: scaffold_level reg value to set. None (default) resets
        level_exponent: how much to weight each level, default 1
        core_net: which ffnetwork is the core (default 0)
        readout_net: which ffnetwork is the readout (default 1)

    Returns:
        None
    """

    # Pull core information 
    #num_levels = len(ndn.networks[core_net].layers)
    level_parse = []
    for ii in ndn.networks[core_net].scaffold_levels:
        level_parse.append( ndn.networks[core_net].layers[ii].output_dims[0] )

    ndn.networks[readout_net].layers[0]._set_scaffold_reg(reg_val=reg_val, scaffold_level_parse=level_parse, level_exponent=level_exponent)
# END set_scaffold_level_reg


def save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = None):
    """Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network (use model.state_dict() to get it)
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
    
    credit: pulled from https://github.com/IgorSusmelj/pytorch-styleguide
    """

    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint.txt')

    save_base = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_base + '\n'] + ckpt_list
    else:
        ckpt_list = [save_base + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.

    credit: pulled from https://github.com/IgorSusmelj/pytorch-styleguide
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def ensure_dir(dir_name: str):
    """Creates folder if not exists.

    credit: pulled from https://github.com/IgorSusmelj/pytorch-styleguide
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def ModelSummary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0][0].size())
            #summary[m_key]["input_shape"] = list(input[0].size())   # CHANGED to above: inputs are now list 
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)

def get_fit_versions(data_dir, model_name):
    '''
        Find versions of the fit model
        Arguments:
            data_dir: directory where the checkpoints are stored
            model_name: name of the model
    '''

    import re
    from tensorboard.backend.event_processing import event_accumulator

    dirlist = [x for x in os.listdir(os.path.join(data_dir, model_name)) if os.path.isdir(os.path.join(data_dir, model_name, x))]
        
    versionlist = [re.findall(r'(?!version)\d+', x) for x in dirlist]
    versionlist = [int(x[0]) for x in versionlist if not not x]

    outdict = {'version_num': [],
        'events_file': [],
        'model_file': [],
        'val_loss': [],
        'val_loss_steps': []}

    for v in versionlist:
        # events_file = os.path.join(data_dir, model_name, 'version_%d' %v, 'events.out.tfevents.%d' %v)
        vpath = os.path.join(data_dir, model_name, 'version%d' %v)
        vplist = os.listdir(vpath)

        tfeventsfiles = [x for x in vplist if 'events.out.tfevents' in x]
        modelfiles = [x for x in vplist if 'model.pt' in x]

        if len(tfeventsfiles) == 1 and len(modelfiles) == 1:
            evfile = os.path.join(vpath, tfeventsfiles[0])
            # read from tensorboard backend
            ea = event_accumulator.EventAccumulator(evfile)
            ea.Reload()
            try:
                val = np.asarray([x.value for x in ea.scalars.Items("Loss/Validation (Epoch)")])
                bestval = np.min(val)

                outdict['version_num'].append(v)
                outdict['events_file'].append(evfile)
                outdict['model_file'].append(os.path.join(vpath, modelfiles[0]))
                outdict['val_loss_steps'].append(val)
                outdict['val_loss'].append(bestval)
            except:
                continue

    return outdict


def ensure_dir(dir_name: str):
    """Creates folder if not exists.

    credit: pulled from https://github.com/IgorSusmelj/pytorch-styleguide
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def ModelSummary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0][0].size())
            #summary[m_key]["input_shape"] = list(input[0].size())   # CHANGED to above: inputs are now list
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)
    
    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)

def load_model(checkpoint_path, model_name='', version=None, verbose=True):
    out = get_fit_versions(checkpoint_path, model_name)
    if version is None:
        version = out['version_num'][np.nanargmin(np.asarray(out['val_loss']))]
        if verbose:
            print("No version requested. Using (best) version (v=%d)" %version)

    assert version in out['version_num'], "Version %d not found in %s. Must be: %s" %(version, checkpoint_path, str(out['version_num']))
    ver_ix = np.where(version==np.asarray(out['version_num']))[0][0]
    # Load the model
    try:
        model = torch.load(out['model_file'][ver_ix])
        dirpath = os.path.dirname(out['model_file'][ver_ix])
        if os.path.exists(os.path.join(dirpath, 'best_model.ckpt')):
            state_dict = torch.load(os.path.join(dirpath, 'best_model.ckpt'))
            model.load_state_dict(state_dict['net'])
    except AttributeError:
        print("load_model: could not load model. AttributeError. This likely means that the file [%s] was not pickled correctly because you were changing the class too much while training" %out['model_file'][ver_ix])
        print("Loading the state dict from the last checkpoint instead")
        filename = out['model_file'][ver_ix]
        modelname = os.path.basename(filename)
        model = torch.load(filename.replace(modelname, 'model_checkpoint.ckpt'))
        return model

    return model


## FROM JAKE
def get_max_samples(dataset, device,
    history_size=10,
    nquad=0,
    num_cells=None,
    buffer=1.2):
    """
    get the maximum number of samples that fit in memory
    Inputs:
        dataset: the dataset to get the samples from
        device: the device to put the samples on
    Optional:
        history_size: the history size parameter for LBFGS (scales memory usage)
        nquad: the number of quadratic kernels for a gqm (adds # parameters for every new quad filter)
        num_cells: the number of cells in model (n cells * n parameters)
        buffer: extra memory to keep free (in GB)
    """
    if num_cells is None:
        num_cells = dataset.NC
    
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    free = t - (a+r)

    data = dataset[0]
    # mempersample = data['stim'].element_size() * data['stim'].nelement() + 2*data['robs'].element_size() * data['robs'].nelement()
    mempersample = 0
    for cov in list(data.keys()):
        mempersample += data[cov].element_size() * data[cov].nelement()

    mempercell = mempersample * (nquad+1) * (history_size + 1)
    buffer_bytes = buffer*1024**3

    maxsamples = int(free - mempercell*num_cells - buffer_bytes) // mempersample
    print("# samples that can fit on device: {} K".format(maxsamples/1000))
    return maxsamples


# for JSON serialization
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
