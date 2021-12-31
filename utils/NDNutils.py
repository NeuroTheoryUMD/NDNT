### NDNTorchUtils.py ####
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import os
import shutil
from collections import OrderedDict


#################### CREATE OTHER PARAMETER-DICTS ####################
def create_optimizer_params(
        optimizer='AdamW',
        batch_size=1000,
        max_iter=None,
        max_epochs=None,
        num_workers=1,
        num_gpus=1,
        optimize_graph = True,
        log_activations = False,
        progress_bar_refresh=20, # num of batches 
        # AdamW specific
        early_stopping=True,
        early_stopping_patience=4,
        weight_decay=0.01,
        learning_rate=1e-3,
        betas=[0.9, 0.999],
        # L-BFGS specific
        full_batch=True,
        tolerance_change=1e-5,
        tolerance_grad=1e-9,
        history_size=10,
        accumulated_grad_batches=1,
        line_search_fn=None):

    # Make optimizer params based on Adam or LBFGS, filling in chosen defaults
    if optimizer in ['LBFGS', 'lbfgs']:

        # L-BFGS specific defaults
        if max_iter is None:
            max_iter = 300
        if max_epochs is None:
            max_epochs = 5

        optpar = {
            'optimizer': 'LBFGS',
            'full_batch': full_batch,
            'batch_size': batch_size,
            'max_iter': max_iter,
            'max_epochs': max_epochs,
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
            'early_stopping': False,
            'device': None}

    else: # Assume some sort of adam / sgd with similar params/defaults
        if max_iter is None:
            max_iter = 10000

        if max_epochs is None:
            if early_stopping:
                max_epochs = 100
            else:
                max_epochs = 30

        optpar = {
            'optimizer': optimizer,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'early_stopping': early_stopping,
            'early_stopping_patience': early_stopping_patience,
            'max_iter': max_iter,
            'max_epochs': max_epochs,
            'learning_rate': learning_rate,
            'betas': betas, 
            'amsgrad': False,
            'auto_lr': False,
            'num_workers': num_workers,
            'num_gpus': num_gpus,
            'progress_bar_refresh': progress_bar_refresh,
            'log_activations': log_activations, 
            'optimize_graph': optimize_graph,
            'device': None} 

    return optpar
# END create_optimizer_params




#################### DIRECTORY ORGANIZATION ####################
def default_save_dir():
    savedir = './checkpoints/'


#################### GENERAL EMBEDDED UTILS ####################
def create_time_embedding(stim, pdims, up_fac=1, tent_spacing=1):
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
        isinstance(val, np.int) or isinstance(val, np.int32) or isinstance(val, np.int64) or \
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

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

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
        
    versionlist = [re.findall('(?!version)\d+', x) for x in dirlist]
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
        version = out['version_num'][np.argmin(np.asarray(out['val_loss']))]
        if verbose:
            print("No version requested. Using (best) version (v=%d)" %version)

    assert version in out['version_num'], "Version %d not found in %s. Must be: %s" %(version, checkpoint_path, str(out['version_num']))
    ver_ix = np.where(version==np.asarray(out['version_num']))[0][0]
    # Load the model
    try:
        model = torch.load(out['model_file'][ver_ix])
    except AttributeError:
        print("load_model: could not load model. AttributeError. This likely means that the file [%s] was not pickled correctly because you were changing the class too much while training" %out['model_file'][ver_ix])
        print("Loading the state dict from the last checkpoint instead")
        filename = out['model_file'][ver_ix]
        modelname = os.path.basename(filename)
        model = torch.load(filename.replace(modelname, 'model_checkpoint.ckpt'))
        return model

    return model
