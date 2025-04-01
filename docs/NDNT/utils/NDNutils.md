Module NDNT.utils.NDNutils
==========================

Functions
---------

`ModelSummary(model, input_size, batch_size=-1, device=device(type='cuda', index=0), dtypes=None)`
:   

`create_optimizer_params(optimizer_type='AdamW', batch_size=1000, accumulate_grad_batches=1, max_iter=None, max_epochs=None, num_workers=0, num_gpus=1, optimize_graph=True, log_activations=False, progress_bar_refresh=20, early_stopping=True, early_stopping_patience=4, early_stopping_delta=0.0, weight_decay=0.01, learning_rate=0.001, betas=[0.9, 0.999], auto_lr=False, full_batch=True, tolerance_change=1e-08, tolerance_grad=1e-10, history_size=10, device=None, line_search_fn=None)`
:   

`create_time_embedding(stim, num_lags, up_fac=1, tent_spacing=1)`
:   Takes a Txd stimulus matrix and creates a time-embedded matrix of size 
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

`create_time_embedding_NIM(stim, pdims, up_fac=1, tent_spacing=1)`
:   All the arguments starting with a p are part of params structure which I 
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

`default_save_dir()`
:   

`design_matrix_tent_basis(s, anchors, zero_left=False, zero_right=False)`
:   Produce a design matrix based on continuous data (s) and anchor points for a tent_basis.
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

`ensure_dir(dir_name: str)`
:   Creates folder if not exists.
    
    credit: pulled from https://github.com/IgorSusmelj/pytorch-styleguide

`fit_lbfgs(mod, data, parameters=None, optimizer=None, verbose=True, max_iter=1000, lr=1, line_search='strong_wolfe', history_size=100, tolerance_change=1e-07, tolerance_grad=1e-07)`
:   Runs fullbatch LBFGS on a Pytorch model and data dictionary
    Inputs:
        Model: Pytorch model
        data: Dictionary to used with Model.training_step(data)

`fit_lbfgs_batch(model, dataset, batch_size=1000, num_chunks=None, train_inds=None, device=None, verbose=True, max_iter=1000, lr=1, history_size=100, tolerance_change=1e-07, tolerance_grad=1e-07, line_search='strong_wolfe')`
:   Runs LBFGS on a Pytorch model with batching a dataset into chunks. Both model and 
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

`generate_xv_folds(nt, num_folds=5, num_blocks=3, which_fold=None)`
:   Will generate unique and cross-validation indices, but subsample in each block
    NT = number of time steps
    num_folds = fraction of data (1/fold) to set aside for cross-validation
    which_fold = which fraction of data to set aside for cross-validation (default: middle of each block)
    num_blocks = how many blocks to sample fold validation from

`get_fit_versions(data_dir, model_name)`
:   Find versions of the fit model
    Arguments:
        data_dir: directory where the checkpoints are stored
        model_name: name of the model

`get_max_samples(dataset, device, history_size=10, nquad=0, num_cells=None, buffer=1.2)`
:   get the maximum number of samples that fit in memory
    Inputs:
        dataset: the dataset to get the samples from
        device: the device to put the samples on
    Optional:
        history_size: the history size parameter for LBFGS (scales memory usage)
        nquad: the number of quadratic kernels for a gqm (adds # parameters for every new quad filter)
        num_cells: the number of cells in model (n cells * n parameters)
        buffer: extra memory to keep free (in GB)

`grid2pixel(x, L=60, force_int=True, enforce_bounds=False)`
:   Older function: replaced by mu2pixel

`initialize_gaussian_envelope(ws, w_shape)`
:   This assumes a set of filters is passed in, and windows by Gaussian along each non-singleton dimension
    ws is all filters (ndims x nfilters)
    wshape is individual filter shape

`is_int(val)`
:   returns Boolean as to whether val is one of many types of integers

`load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False)`
:   Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    
    credit: pulled from https://github.com/IgorSusmelj/pytorch-styleguide

`load_model(checkpoint_path, model_name=None, version=None, verbose=True, filename=None)`
:   Loads model from checkpoint
    
    Args:
        checkpoint_path
        model_name=''
        version=None
        verbose (default True)
        filename: (default None)
    
    Returns:
        model: NDN model

`mu2pixel(x, L=60, force_int=True, enforce_bounds=True, flip_axes=True)`
:   Converts from mu values back into pixel coordinates, where mus are coordinates used by grid_sample.
    Can be continuous (fractional pixels), but default is rounding to nearest int
    
    Args: 
        x: mu values, assuming num_locations x 2 (although can be 1-dimensional)
        L: size of grid (number of pixels), assuming square (default 60)
        force_int: force to be integer (versus continuous-valued)
        enforce bounds: go to edges if answer ends up bigger than L-1 or smaller than 0
        flip_axes: whether to swap horizontal and vertical axes as grid_sample needs (default: True)
    
    Returns:
        p: pixel values

`pixel2grid(p, L=60)`
:   Pixels are starting with number 0 up to L-1, converted to range of -1 to 1. This is the
    old function, to be replaced by pixel2mu

`pixel2mu(p, L=60, flip_axes=True)`
:   Converts from pixel coordinates to mu values, used by grid_sample. The default will flip
    horizontal and vertical axes automatically, which is what grid_sample requires
    Pixels are starting with number 0 up to L-1, converted to range of -1 to 1
    
    Args:
        p: list of pixel locations, presumably num_locations x 2 (horizontal and vertical coords)
        L: size of grid (number of pixels), assuming square (default 60)
        flip_axes: whether to swap horizontal and vertical axes as grid_sample needs (default: True)
    
    Returns:
        x: mu values

`save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = None)`
:   Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network (use model.state_dict() to get it)
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
    
    credit: pulled from https://github.com/IgorSusmelj/pytorch-styleguide

`set_scaffold_level_reg(ndn, reg_val=None, level_exponent=1.0, core_net=0, readout_net=1)`
:   Sets up regularization for scaffold_level, which requires pulling scaffold structure from 
    core network and passing it into the readout layer at beginning of readout_net
    
    Args:
        ndn: model that has scaffold and readout ffnetworks
        reg_val: scaffold_level reg value to set. None (default) resets
        level_exponent: how much to weight each level, default 1
        core_net: which ffnetwork is the core (default 0)
        readout_net: which ffnetwork is the readout (default 1)
    
    Returns:
        None

`shift_mat_zpad(x, shift, dim=0)`
:   Takes a vector or matrix and shifts it along dimension dim by amount 
    shift using zero-padding. Positive shifts move the matrix right or down.
    
    Args:
        x (type): description
        shift (type): description
        dim (type): description
        
    Returns:
        type: description
            
    Raises:

`summary_string(model, input_size, batch_size=-1, device=device(type='cuda', index=0), dtypes=None)`
:   

`tent_basis_generate(xs=None, num_params=None, doubling_time=None, init_spacing=1, first_lag=0)`
:   Computes tent-bases over the range of 'xs', with center points at each value of 'xs'.
    Alternatively (if xs=None), will generate a list with init_space and doubling_time up to
    the total number of parameters. Must specify xs OR num_params. 
    Note this assumes discrete (binned) variables to be acted on.
    
    Defaults:
        doubling_time = num_params
        init_space = 1

Classes
-------

`CPU_Unpickler(*args, **kwargs)`
:   This takes a binary file for reading a pickle data stream.
    
    The protocol version of the pickle is detected automatically, so no
    protocol argument is needed.  Bytes past the pickled object's
    representation are ignored.
    
    The argument *file* must have two methods, a read() method that takes
    an integer argument, and a readline() method that requires no
    arguments.  Both methods should return bytes.  Thus *file* can be a
    binary file object opened for reading, an io.BytesIO object, or any
    other custom object that meets this interface.
    
    Optional keyword arguments are *fix_imports*, *encoding* and *errors*,
    which are used to control compatibility support for pickle stream
    generated by Python 2.  If *fix_imports* is True, pickle will try to
    map the old Python 2 names to the new names used in Python 3.  The
    *encoding* and *errors* tell pickle how to decode 8-bit string
    instances pickled by Python 2; these default to 'ASCII' and 'strict',
    respectively.  The *encoding* can be 'bytes' to read these 8-bit
    string instances as bytes objects.

    ### Ancestors (in MRO)

    * _pickle.Unpickler

    ### Methods

    `find_class(self, module, name)`
    :   Return an object from a specified module.
        
        If necessary, the module will be imported. Subclasses may override
        this method (e.g. to restrict unpickling of arbitrary classes and
        functions).
        
        This method is called whenever a class or a function object is
        needed.  Both arguments passed are str objects.

`NpEncoder(*, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)`
:   Extensible JSON <http://json.org> encoder for Python data structures.
    
    Supports the following objects and types by default:
    
    +-------------------+---------------+
    | Python            | JSON          |
    +===================+===============+
    | dict              | object        |
    +-------------------+---------------+
    | list, tuple       | array         |
    +-------------------+---------------+
    | str               | string        |
    +-------------------+---------------+
    | int, float        | number        |
    +-------------------+---------------+
    | True              | true          |
    +-------------------+---------------+
    | False             | false         |
    +-------------------+---------------+
    | None              | null          |
    +-------------------+---------------+
    
    To extend this to recognize other objects, subclass and implement a
    ``.default()`` method with another method that returns a serializable
    object for ``o`` if possible, otherwise it should call the superclass
    implementation (to raise ``TypeError``).
    
    Constructor for JSONEncoder, with sensible defaults.
    
    If skipkeys is false, then it is a TypeError to attempt
    encoding of keys that are not str, int, float or None.  If
    skipkeys is True, such items are simply skipped.
    
    If ensure_ascii is true, the output is guaranteed to be str
    objects with all incoming non-ASCII characters escaped.  If
    ensure_ascii is false, the output can contain non-ASCII characters.
    
    If check_circular is true, then lists, dicts, and custom encoded
    objects will be checked for circular references during encoding to
    prevent an infinite recursion (which would cause an RecursionError).
    Otherwise, no such check takes place.
    
    If allow_nan is true, then NaN, Infinity, and -Infinity will be
    encoded as such.  This behavior is not JSON specification compliant,
    but is consistent with most JavaScript based encoders and decoders.
    Otherwise, it will be a ValueError to encode such floats.
    
    If sort_keys is true, then the output of dictionaries will be
    sorted by key; this is useful for regression tests to ensure
    that JSON serializations can be compared on a day-to-day basis.
    
    If indent is a non-negative integer, then JSON array
    elements and object members will be pretty-printed with that
    indent level.  An indent level of 0 will only insert newlines.
    None is the most compact representation.
    
    If specified, separators should be an (item_separator, key_separator)
    tuple.  The default is (', ', ': ') if *indent* is ``None`` and
    (',', ': ') otherwise.  To get the most compact JSON representation,
    you should specify (',', ':') to eliminate whitespace.
    
    If specified, default is a function that gets called for objects
    that can't otherwise be serialized.  It should return a JSON encodable
    version of the object or raise a ``TypeError``.

    ### Ancestors (in MRO)

    * json.encoder.JSONEncoder

    ### Methods

    `default(self, obj)`
    :   Implement this method in a subclass such that it returns
        a serializable object for ``o``, or calls the base implementation
        (to raise a ``TypeError``).
        
        For example, to support arbitrary iterators, you could
        implement default like this::
        
            def default(self, o):
                try:
                    iterable = iter(o)
                except TypeError:
                    pass
                else:
                    return list(iterable)
                # Let the base class default method raise the TypeError
                return JSONEncoder.default(self, o)