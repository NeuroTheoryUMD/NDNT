### NDNTorchUtils.py ####
import numpy as np
from copy import deepcopy

#################### CREATE NETWORK PARAMETER-DICTS ####################
def layer_dict(
    input_dims=None, num_filters=1, NLtype='relu', 
    norm_type=0, pos_constraint=False, num_inh=0,
    conv=False, conv_width=None):

    """input dims are [num_filters, space1, space2, num_lags]"""

    # Add any other nonlinaerities here (and pass to functionals below)
    val_nls = ['lin', 'relu', 'quad', 'softplus', 'tanh', 'sigmoid']
    assert NLtype in val_nls, 'NLtype not valid.'

    if input_dims is None:
        input_dims = [1,1,1,1]
    output_dims = [num_filters, 1, 1, 1]
    if conv:
        output_dims[1:3] = input_dims[1:3]
        if conv_width is None:
            TypeError( 'Need to define conv filter-width.')
        filter_dims = [input_dims[0], conv_width, conv_width, input_dims[3]]
        if input_dims[2] == 1:  # then 1-d spatial
            filter_dims[2] = 1
    else:
        filter_dims = deepcopy(input_dims)

    if num_inh > num_filters:
        print("Warning: num_inh is too large. Adjusted to ", num_filters)
        num_inh = num_filters
        
    params_dict = {
        'input_dims': input_dims,
        'output_dims': output_dims,
        'num_filters': num_filters,
        'filter_dims': filter_dims, 
        'NLtype': NLtype, 
        'norm_type': norm_type, 
        'pos_constraint': pos_constraint,
        'conv': conv,
        'num_inh': num_inh,
        'initializer': 'uniform',
        'bias': True}

    return params_dict
# END layer_dict

def ffnet_params_default(xstim_n=None, ffnet_n=None):
    """This creates a ffnetwork_params object that specifies details of ffnetwork
    ffnetwork dicts have these fields:
        ffnet_type: string specifying network type (default = 'normal')
        layer_list: defaults to None, to be set in internal function
        input_dims: defaults to None, but will be set when network made, if not sooner
        xstim_n: external input (or None). Note can only be from one source
        ffnet_n: list of internal network inputs (has to be a list, or None)
        conv: [boolean] whether ffnetwork is convolutional or not, defaults to False but to be set
        
    -- Note xstim_n and ffnet_n are created/formatted here. 
    -- If xstim_n is specified, it must specify input dimensions
    -- This should set all required fields as needed (even if none)"""

    if ffnet_n is None:
        if xstim_n is None:
            xstim_n = 0  # default to first external input if unspecified
    else:
        # currently does not concatenate internal and external inputs (should it?)
        assert xstim_n is None, "Currently cannot have external and internal inputs."
        if type(ffnet_n) is not list:
            ffnet_n = [ffnet_n]

    ffnet_params = {
        'ffnet_type': 'normal',
        'layer_list': None,
        'input_dims': None,
        'xstim_n': xstim_n,
        'ffnet_n': ffnet_n,
        'conv': False,
        'reg_list': None
        }
    return ffnet_params
# END ffnet_params_default


def ffnet_dict_NIM(
    input_dims=None, 
    layer_sizes=None, 
    act_funcs=None,
    ei_layers=None,
    norm_list=None,
    reg_list=None):

    """This creates will make a list of layer dicts corresponding to a non-convolutional NIM].
    Note that input_dims can be set to none"""

    ffnet_params = ffnet_params_default(xstim_n=0, ffnet_n=None)
    ffnet_params['input_dims'] = input_dims
    ffnet_params['reg_list'] = reg_list

    num_layers = len(layer_sizes)
    assert len(act_funcs) == num_layers, "act_funcs is wrong length."

    layer_list = []   
    indims = input_dims  # starts with network input dims

    if ei_layers is None:
        ei_layers = [None]*num_layers
    if norm_list is None:
        norm_list = [None]*num_layers

    for ll in range(num_layers):
        pos_con = False
        if ll > 0:
            if ei_layers[ll-1] is not None:
                pos_con = True
        if ei_layers[ll] is None:
            num_inh = 0
        else:
            num_inh = ei_layers[ll]

        layer_list.append(
            layer_dict(
                input_dims = indims, 
                num_filters = layer_sizes[ll],
                NLtype = act_funcs[ll],
                norm_type = norm_list[ll],
                pos_constraint = pos_con,
                num_inh = num_inh))

        indims = layer_list[-1]['output_dims']
    ffnet_params['layer_list'] = layer_list

    return ffnet_params
# END ffnet_dict_NIM


#################### CREATE OTHER PARAMETER-DICTS ####################
def create_optimizer_params(
        batch_size=1000,
        weight_decay=None,
        early_stopping=True,
        early_stopping_patience=4,
        max_iter=10000,
        learning_rate=1e-3,
        betas=[0.9, 0.999],
        num_gpus=1,
        progress_bar_refresh=20, # num of batches 
        num_workers=4):

    if early_stopping:
        max_epochs = 1000
    else:
        max_epochs = 300

    optpar = {
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
        'progress_bar_refresh': progress_bar_refresh,
        'num_workers': num_workers,
        'num_gpus': num_gpus,
        'optimizer': 'AdamW'}

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