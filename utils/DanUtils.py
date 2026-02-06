import numpy as np
from copy import deepcopy
from ..utils import NDNutils
from ..utils import DanUtils as DU
import matplotlib.pyplot as plt


###### GENERAL UTILITIES NOT SPECIFIC TO NDN ######
def subplot_setup(num_rows, num_cols, row_height=3, fig_width=16, fighandle=False):
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
    fig.set_size_inches(fig_width, h=row_height*num_rows)
    fig.tight_layout()
    if fighandle is True:
        return fig


def ss( num_rows=1, num_cols=1, row_height=4, rh=None, fighandle=False):
    # Short-hand for subplot_setup with useful defaults for quick usage
    if rh is not None:
        row_height=rh  # really to just save a bit of typing
    h = subplot_setup(num_rows, num_cols, row_height=row_height, fighandle=fighandle)
    if fighandle:
        return h
    

def imagesc( img, cmap=None, balanced=None, aspect=None, max=None, colrow=True, axis_labels=True, ax=None ):
    """Modifications of plt.imshow that choose reasonable defaults"""
    if balanced is None:
        # Make defaults depending on img
        if np.sign(np.max(img)) == np.sign(np.min(img)):
            balanced = False
        else:
            balanced = True
    if balanced:
        imin = -np.max(abs(img))
        imax = np.max(abs(img))
    else:
        imin = np.min(img)
        imax = np.max(img)

    if max is not None:
        imin = -max
        imax = max

    if aspect is None:
        if img.shape[0] == img.shape[1]:
            aspect = 1
        else:
            aspect = 'auto'

    if colrow:  # then plot with first axis horizontal, second axis vertical
        if ax is None:
            plt.imshow( img.T, cmap=cmap, interpolation='none', aspect=aspect, vmin=imin, vmax=imax)
        else:
            ax.imshow( img.T, cmap=cmap, interpolation='none', aspect=aspect, vmin=imin, vmax=imax)
    else:  # this is like imshow: row, column
        if ax is None:
            plt.imshow( img, cmap=cmap, interpolation='none', aspect=aspect, vmin=imin, vmax=imax)
        else:
            ax.imshow( img, cmap=cmap, interpolation='none', aspect=aspect, vmin=imin, vmax=imax)
    if not axis_labels:
        figgy = plt.gca()
        figgy.axes.xaxis.set_ticklabels([])
        figgy.axes.yaxis.set_ticklabels([])
# END imagesc


def scatterplot( arr2, arrS2=None, clr='b.', alpha=1.0, diag=False, square=False ):
    """
    Generates scatter-plot of 2-d data (arr2) using the following options:

    Args:
        arr2 (array): 2-d array to plot (Nx2)
        arrS2 (array): second array if arr2 is 1-d
        clr (str): symbol to use (default 'b.')
        alpha (float): transparency level (default: 1)
        diag (boolean): whether to draw x-y diagonal line (default False)

    Returns:
        None, simply display to screen
    """
    if square:
        _, ax = plt.subplots(1,1)

    if len(arr2.shape) > 1:
        N, dims = arr2.shape
        if dims == 2:
            assert arrS2 is None, "If arr2 is 2D, arrS2 must be None"
    else:
        dims = 1
    if dims == 1:
        assert arrS2 is not None, "If arr2 is 1D, arrS2 must be provided"
        assert len(arr2) == len(arrS2), "arr2 and arrS2 must have same length"
        plt.plot(arr2, arrS2, clr, alpha=alpha )
    else:
        plt.plot(arr2[:,0], arr2[:,1], clr, alpha=alpha )
    xs = plt.xlim()
    ys = plt.ylim()
    if diag:
        mn = np.minimum(xs[0], ys[0])
        mx = np.maximum(xs[1], ys[1])
        plt.plot([mn,mx], [mn,mx],'k')
        plt.xlim([mn, mx])
        plt.ylim([mn, mx])
    #plt.show()
    if square:
        ax.set_box_aspect(1) 
# END scatterplot()


def find_peaks( x, clearance=10, max_peaks=10, thresh=13.0 ):
    """Find maximum of peaks and then get rid of other points around it for plus/minus some amount"""
    y = deepcopy(x)
    rem = np.arange(len(x))
    pks, amps = [], []
    for counter in range(max_peaks):
        #plt.plot(y,clrs[counter])
        a = rem[np.argmax(y[rem])]
        if y[a] >= thresh:
            pks.append(a)
            amps.append(y[a])
        rem = np.setdiff1d(rem, np.arange(a-clearance, a+clearance+1), assume_unique=True )
        
    pks = np.array(pks, dtype=np.int64)
    amps = np.array(amps, dtype=np.float32)

    return pks, amps
# END find_peaks


def regression2d( x2s, Yobs ):
    """
    2-d regression on data with N examples, where Yobs is predicted by x2s, 
    where x2s is Nx2 and Yobs is Nx1. Yobs can have multiple columns, with each column predicted
    independently from x2s. Prediction(s) are y_i:
    
    y = a + b_1 x_1 + b_2 x_2 (where a, b1, b2 and y are i-specific)
    
    Args:
        x2s: 2-d data inputs (Nx2)
        Yobs: target output(s): (NxM) will regress each column separately
        
    Returns:
        regr_mat: matrix (2xM) for the regression slopes such that x2s@reg_mat + offset
        regr_off: offsets (Mx0), see above
    """
    N, dims = x2s.shape
    assert dims == 2, "x2s argument must be two-dimensional (Nx2)"
    assert Yobs.shape[0] == N, "Yobs must have same number of data points as x2s"
    
    if len(Yobs.shape) == 1:
        Yobs = Yobs[:,None]
    num_vars = Yobs.shape[1]

    regr_mat = np.zeros([2,num_vars], dtype=np.float32)
    regr_off = np.zeros(num_vars)
    x1 = x2s[:, 0] - np.mean(x2s[:, 0])
    x2 = x2s[:, 1] - np.mean(x2s[:, 1])
    denom = np.sum(x1**2) * np.sum(x2**2) - np.sum(x1*x2)**2
    assert denom != 0, "Does not converge: denominator is zero"
    for ii in range(num_vars):
        yi = Yobs[:,ii] - np.mean(Yobs[:,ii])
        regr_mat[0,ii] = ( np.sum(x2**2) * np.sum(x1*yi) - np.sum(x1*x2) * np.sum(x2*yi) ) / denom # b1
        regr_mat[1,ii] = ( np.sum(x1**2) * np.sum(x2*yi) - np.sum(x1*x2) * np.sum(x1*yi) ) / denom # b2
        #regr_off[ii] = np.mean(yi) - np.mean(regr_mat[0,ii]*x1 + regr_mat[1,ii]*x2) # a
        regr_off[ii] = np.mean(Yobs[:,ii]) - np.mean(regr_mat[0,ii]*x2s[:, 0] + regr_mat[1,ii]*x2s[:, 1]) # a
    return regr_mat.squeeze(), regr_off
# END regression2d()


def chunker(seq, size):
    """
    This function chunks a sequence into chunks of size size.

    Args:
        seq: the sequence
        size: the size of the chunks

    Returns:
        a list of chunks
    """
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]


def filename_num2str( n, num_digits=2 ):
    if n == 0:
        num_places = 1
    else:
        num_places = int(np.ceil(np.log10(n)+0.001))
    place_shift = int(np.maximum(num_digits-num_places, 0))
    s = '0'*place_shift + str(n%(10**num_digits))[:]
    return s


def display_matrix( x, prec=3, spacing=4, number_rows=False, number_cols=False ):
    a, b = x.shape
    s = "  %" + str(spacing+prec) + "." + str(prec) + "f"
    if number_cols:
        if number_rows:
            print( "    ", end='')
        for mm in range(b):
            print( " "*(spacing+prec-2)+ "[%2d]"%mm, end='' )
        print('')
    for nn in range(a):
        if number_rows:
            print( "[%2d]"%nn, end='' )
        for mm in range(b):
            print( s%x[nn, mm], end='')
        print('')
# END display_matrix()


def figure_export( fig_handle, filename, bitmap=False, dpi=300):
    """Usage: figure_export( fig_handle, filename, variable_list, bitmap=False, dpi=300)
    if bitmap, will use dpi and export as .png. Otherwise will export PDF"""

    if bitmap:
        fig_handle.savefig( filename, bbox_inches='tight', dpi=dpi, transparent=True )
    else:
        fig_handle.savefig( filename, bbox_inches='tight', transparent=True )


def matlab_export(filename, variable_list):
    """Export list of variables to .mat file"""

    import scipy.io as sio
    if not isinstance(variable_list, list):
        variable_list = [variable_list]

    matdata = {}
    for nn in range(len(variable_list)):
        assert not isinstance(variable_list[nn], list), 'Cant do a list of lists.'
        if nn < 10:
            key_name = 'v0' + str(nn)
        else:
            key_name = 'v' + str(nn)
        matdata[key_name] = variable_list[nn]

    sio.savemat(filename, matdata)


def save_python_data( filename, data ):
    """
    Saves python dictionary in standard binary file

    Args:
        filename: name of the file to save
        data: dictionary to save
    """
    with open( filename, 'wb') as f:
        np.save(f, data)
    print( 'Saved data to', filename )


def load_python_data( filename, show_keys=False ):
    """
    Load python data from standard binary file

    Args:
        filename: name of the file to save
        show_keys: to display components of dictionary (default=False)
    """
    with open( filename, 'rb') as f:
        data = np.load(f, allow_pickle=True)
    print( 'Loaded data from', filename )
    if len(data.shape) == 0:
        data = data.flat[0]  # to rescue dictionaries
    if (type(data) is dict) and show_keys:
        print(data.keys())
    return data
# END load_python_data()


def clean_jupyter_notebook(notebook_filename, new_filename):
    """ 
    Take jupyter notebook file (ipynb) and scrubs the output, in case it takes up too much memory or other
    error running.

    Args:
        notebook_filename: filename of jupyter notebook to be cleaned
        new_filename: filename for new notebook

    Returns:
        Nothing, but writes new file to disk
    """
    from nbformat import read, write
    with open(notebook_filename, 'r', encoding='utf-8') as f:
        nb = read(f, as_version=4)
    for cell in nb.cells:
        if hasattr(cell, 'outputs'):
            cell.outputs = []
        if hasattr(cell, 'execution_count'):
            cell.execution_count = None
    with open(new_filename, 'w', encoding='utf-8') as f:
        write(nb, f)
# END clean_jupyter_notebook


def chunker(seq, size):
    """
    This function chunks a sequence into chunks of size size. (from Matt J)

    Args:
        seq: the sequence
        size: the size of the chunks

    Returns:
        a list of chunks
    """
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]
# END chunker()


def fold_sample( num_items, folds=5, random_gen=False, which_fold=None):
    """Divide fold sample deterministically or randomly distributed over number of items. More options
    can be added, but his captures the basics."""
    if random_gen:
        num_val = int(num_items/folds)
        tmp_seq = np.random.permutation(num_items)
        val_items = np.sort(tmp_seq[:num_val])
        rem_items = np.sort(tmp_seq[num_val:])
    else:
        if which_fold is None:
            offset = int(folds//2) # sample middle folds as test_data
        else:
            assert which_fold < folds
            offset = which_fold
        val_items = np.arange(offset, num_items, folds, dtype='int64')
        rem_items = np.delete(np.arange(num_items, dtype='int64'), val_items)
    return val_items, rem_items
# END fold_sample()


# Generic time-embedding
def time_embedding_simple( stim, nlags ):
    """Simple time embedding: takes the stim and time-embeds with nlags
    If stim is multi-dimensional, it flattens. This is a numpy function"""

    if len(stim.shape) == 1:
        stim = stim[:, None]
    NT, num_dims = stim.shape
    tmp_stim = deepcopy(stim).reshape([NT, -1])  # automatically generates 2-dimensional stim

    X = np.zeros([NT, num_dims, nlags], dtype=np.float32)
    X[:, :, 0] = tmp_stim
    for ii in range(1, nlags):
        X[ii:, :, ii] = tmp_stim[:(-ii),:]
    return X.reshape([NT, -1])
# END time_embedding_simple()


def design_matrix_drift( NT, anchors, zero_left=True, zero_right=False, const_right=False, to_plot=False):
    """Produce a design matrix based on continuous data (s) and anchor points for a tent_basis.
    Here s is a continuous variable (e.g., a stimulus) that is function of time -- single dimension --
    and this will generate apply a tent basis set to s with a basis variable for each anchor point. 
    The end anchor points will be one-sided, but these can be dropped by changing "zero_left" and/or
    "zero_right" into "True".

    Inputs: 
        NT: length of design matrix
        anchors: list or array of anchor points for tent-basis set
        zero_left, zero_right: boolean whether to drop the edge bases (default for both is False)
    Outputs:
        X: design matrix that will be NT x the number of anchors left after zeroing out left and right
    """
    anchors = list(anchors)
    if anchors[0] > 0:
        anchors = [0] + anchors
    #if anchors[-1] < NT:
    #    anchors = anchors + [NT]
    NA = len(anchors)

    X = np.zeros([NT, NA])
    for aa in range(NA):
        if aa > 0:
            dx = anchors[aa]-anchors[aa-1]
            X[range(anchors[aa-1], anchors[aa]), aa] = np.arange(dx)/dx
        if aa < NA-1:
            dx = anchors[aa+1]-anchors[aa]
            X[range(anchors[aa], anchors[aa+1]), aa] = 1-np.arange(dx)/dx

    if zero_left:
        X = X[:, 1:]

    if const_right:
        X[range(anchors[-1], NT), -1] = 1.0

    if zero_right:
        X = X[:, :-1]

    if to_plot:
        plt.imshow(X.T, aspect='auto', interpolation='none')
        plt.show()

    return X

##### GENERAL UTILITY FUNCTIONS? ########
def dist_mean( p ):
    xs = np.arange(len(p))
    return np.sum(xs*p)/np.sum(p)


def max_multiD(k):
    num_dims = len(k.shape)
    if num_dims == 2:
        a,b = k.shape
        d1best = np.argmax( np.max(k, axis=0) )
        d0best = np.argmax( k[:, d1best] )
        return [d0best, d1best] 
    elif num_dims == 3:
        a,b,c = k.shape
        d2best = np.argmax( np.max(np.reshape(k, [a*b, c]), axis=0) )
        d1best = np.argmax( np.max(k[:,:, d2best], axis=0) )
        d0best = np.argmax( k[:,d1best, d2best] )
        return [d0best, d1best, d2best]
    elif num_dims == 4:
        a,b,c,d = k.shape
        d3best = np.argmax( np.max(np.reshape(k, [a*b*c, d]), axis=0) )
        d2best = np.argmax( np.max(np.reshape(k[..., d3best], [a*b, c]), axis=0) )
        d1best = np.argmax( np.max(k[:,:, d2best, d3best], axis=0) )
        d0best = np.argmax( k[:,d1best, d2best, d3best] )
        return [d0best, d1best, d2best, d3best]
    else:
        print('havent figured out how to do this number of dimensions yet.')
# END max_multiD()


def boxcar_smooth( s, win_size=5 ):
    """
    Boxcar smoothing
    """
    from scipy.signal import convolve
    dims = [win_size]
    if len(s.shape) > 1:
        dims += s.shape[1:]
    kernel = np.ones(dims) / win_size
    s_smooth = convolve(s, kernel, mode='same')
    return s_smooth
# END boxcar_smoothing()


def median_smoothing( f, L=5):
    """
    Median smoothing of a 1D signal (or multi-d signal where median smoothing in first dimension), 
    using padding on the ends

    Args:
        f: input signal
        L: window radius (from each time point)

    Returns:
        mout: smoothed signal
    """
    to_squeeze = False
    if len(f.shape) == 1:
        f = f[:,None]
        to_squeeze = True
    other_dims = list(f.shape[1:])
    start_vals = np.median(f[:L,:], axis=0)
    end_vals = np.median(f[-L:,:], axis=0)
    fmed = np.concatenate((
        np.ones([L]+other_dims)*start_vals[None,:], 
        deepcopy(f),
        np.ones([L]+other_dims)*end_vals[None,:]), axis=0)
    mout = deepcopy(f)
    for tt in range(L, len(fmed)-L):
        mout[tt-L] = np.median(fmed[np.arange(tt-L,tt+L)],axis=0)
    if to_squeeze:
        return mout.squeeze()
    else:
        return mout
# END median_smoothing()


def iterate_lbfgs(mod, dat, lbfgs_pars, train_inds=None, val_inds=None, 
                  tol=0.0001, max_iter = 20, verbose=True ):
    
    if train_inds is None:
        train_inds = dat.train_inds
    if val_inds is None:
        val_inds = dat.val_inds

    cont = True
    iter = 0
    while cont and (iter < max_iter):
        LLprev = mod.eval_models(dat[val_inds], null_adjusted=False)[0]
        save_mod = deepcopy(mod)
        mod.fit(dat, train_inds=train_inds, val_inds=val_inds,
                force_dict_training=True, **lbfgs_pars, version=9, verbose=0, seed=iter)
        LLcur = mod.eval_models(dat[val_inds], null_adjusted=False)[0]
        if LLprev-LLcur < tol:
            cont = False
            if LLprev < LLcur:
                mod = save_mod
                LLcur = LLprev
                if verbose:
                    print("    Reverting")
            else:
                if verbose:
                    print("    Below tolerance", LLprev-LLcur)
        else:
            if verbose:
                print( "  Iter %d: %0.6f"%(iter, LLcur) )
        iter += 1
    return mod, LLcur


def binocular_data_import( datadir, expt_num ):
    """Usage: stim, Robs, DFs, used_inds, Eadd_info = binocular_data_import( datadir, expt_num )

    Inputs:
        datadir: directory on local drive where datafiles are
        expt_num: the experiment number (1-12) representing the binocular experiments we currently have. All
                    datafiles are called 'BS2expt?.mat'. Note that numbered from 1 (not zero)
    
    Outputs:
        stim: formatted as NT x 72 (stimuli in each eye are cropped to NX=36). It will be time-shifted by 1 to 
                eliminate 0-latency stim
        Robs: response concatenating all SUs and MUs, NT x NC. NSU is saved as part of Eadd_info
        DFs:  data_filters for experiment, also NT x NC (note MU datafilters are initialized to 1)
        used_inds: indices overwhich data is valid according to initial data parsing (adjusted to python) 
        Eadd_info: dictionary containing all other relevant info for experiment
    """

    # Constants that are pretty much set for our datasets
    stim_trim = np.concatenate( (range(3,39), range(45,81)))
    time_shift = 1
    NX = 36

    # Read all data into memory
    filename = 'B2Sexpt'+ str(expt_num) + '.mat'
    Bmatdat = sio.loadmat(datadir+filename)
    stim = NDNutils.shift_mat_zpad(Bmatdat['stim'][:,stim_trim], time_shift, 0)

    #NX = int(stim.shape[1]) // 2
    RobsSU = Bmatdat['RobsSU']
    RobsMU = Bmatdat['RobsMU']
    #MUA = Bmatdata[nn]['RobsMU']
    #SUA = Bmatdata[nn]['RobsSU']
    NTtot, numSUs = RobsSU.shape

    data_filtersSU = Bmatdat['SUdata_filter']
    data_filtersMU = np.ones( RobsMU.shape, dtype='float32')  # note currently no MUdata filter but their needs to be....

    Robs = np.concatenate( (RobsSU, RobsMU), axis=1 )
    DFs = np.concatenate( (data_filtersSU, data_filtersMU), axis=1 )

    # Valid and cross-validation indices
    used_inds = np.add(np.transpose(Bmatdat['used_inds'])[0,:], -1) # note adjustment for python v matlab indexing
    Ui_analog = Bmatdat['Ui_analog'][:,0]  # these are automaticall in register
    XiA_analog = Bmatdat['XiA_analog'][:,0]
    XiB_analog = Bmatdat['XiB_analog'][:,0]
    # two cross-validation datasets -- for now combine
    Xi_analog = XiA_analog+XiB_analog  # since they are non-overlapping, will make 1 in both places

    # Derive full-dataset Ui and Xi from analog values
    Ui = np.intersect1d(used_inds, np.where(Ui_analog > 0)[0])
    Xi = np.intersect1d(used_inds, np.where(Xi_analog > 0)[0])

    NC = Robs.shape[1]
    NT = len(used_inds)

    dispt_raw = NDNutils.shift_mat_zpad( Bmatdat['all_disps'][:,0], time_shift, 0 ) # time shift to keep consistent
    # this has the actual disparity values, which are at the resolution of single bars, and centered around the neurons
    # disparity (sometime shifted to drive neurons well)
    # Sometimes a slightly disparity is used, so it helps to round the values at some resolution
    dispt = np.round(dispt_raw*100)/100
    frs = NDNutils.shift_mat_zpad( Bmatdat['all_frs'][:,0], time_shift, 0 )
    corrt = NDNutils.shift_mat_zpad( Bmatdat['all_corrs'][:,0], time_shift, 0 )
    # Make dispt consistent with corrt (early experiments had dispt labeled incorrectly)
    corr_funny = np.where((corrt == 0) & (dispt != -1005))[0]
    if len(corr_funny) > 0:
        print( "Warning: %d indices have corr=0 but labeled disparity."%len(corr_funny) )
        dispt[corr_funny] = -1005

    disp_list = np.unique(dispt)
    # where it is -1009 this corresponds to a blank frame
    # where it is -1005 this corresponds to uncorrelated images between the eyes

    if Bmatdat['rep_inds'] is None:
        #rep_inds = [None]*numSUs
        rep_inds = None
    elif len(Bmatdat['rep_inds'][0][0]) < 10:
        rep_inds = None
    else:
        rep_inds = []
        for cc in range(numSUs):
            rep_inds.append( np.add(Bmatdat['rep_inds'][0][cc], -1) ) 

    print( "Expt %d: %d SUs, %d total units, %d out of %d time points used."%(expt_num, numSUs, NC, NT, NTtot))
    #print(len(disp_list), 'different disparities:', disp_list)

    Eadd_info = {
        'Ui_analog': Ui_analog, 'XiA_analog': XiA_analog, 'XiB_analog': XiB_analog, 'Xi_analog': Xi_analog,
        'Ui': Ui, 'Xi': Xi, # these are for use with data_filters and inclusion of whole experiment,
        'NSU': numSUs, 'NC': NC,
        'dispt': dispt, 'corrt': corrt, 'disp_list': disp_list, 
        'frs': frs, 'rep_inds': rep_inds}

    return stim, Robs, DFs, used_inds, Eadd_info


def binocular_data_import_cell( datadir, expt_num, cell_num ):
    """Usage: stim, Robs, used_inds_cell, UiC, XiC = binocular_data_import_cell( datadir, expt_num, cell_num)

    Imports data for one cell: otherwise identitical to binocular_data_import. Takes data_filter for the 
    cell into account so will not need datafilter. Also, all indices will be adjusted accordingly.

    Inputs:
        datadir: directory on local drive where datafiles are
        expt_num: the experiment number (1-12) representing the binocular experiments we currently have. All
                    datafiles are called 'BS2expt?.mat'. Note that numbered from 1 (not zero)
        cell_num: cell number to analyze
    
    Outputs:
        stim_all: formatted as NT x 72 (stimuli in each eye are cropped to NX=36). It will be time-shifted by 1 
                  to eliminate 0-latency stim. Note this is all stim in experiment, val_inds from used_inds...
        Robs: response concatenating all SUs and MUs, NT x NC. NSU and full Robs is saved as part of Eadd_info.
              This is already selected by used_inds_cell, so no need for further reduction
        used_inds_cell: indices overwhich data is valid for that particular cell. Should be applied to stim only
        UiC, XiC: cross-validation indices for that cell, based on used_inds_cell
        Eadd_info: dictionary containing all other relevant info for experiment

    if the full experiment is NT long, stim and Robs are also NT long, 
        stim[used_inds_cell,:] and Robs[used_inds_cell] are the valid data for that cell, and UiC and XiC are
        the proper train and test indices of used_inds_cell to access, i.e. 
    _model.train( input_data=stim[used_inds_cell,:], output_data=Robs[used_inds_cell], train_indxs=UiC, ...)
    """

    assert cell_num > 0, 'cell number must be positive'
    stim_all, Robs_all, DFs, used_inds, Einfo = binocular_data_import( datadir, expt_num )

    cellspecificdata = np.where(DFs[:, cell_num-1] > 0)[0]
    used_cell = np.intersect1d(used_inds, cellspecificdata)
    NT = len(used_cell)
    Ui = np.where(Einfo['Ui_analog'][used_cell] > 0)[0]

    Xi_analog = Einfo['XiA_analog']+Einfo['XiB_analog']
    XVi1 = np.where(Einfo['XiA_analog'][used_cell] > 0)[0]
    XVi2 = np.where(Einfo['XiB_analog'][used_cell] > 0)[0]
    Xi = np.where(Xi_analog[used_cell] > 0)[0]

    Robs = np.expand_dims(Robs_all[:, cell_num-1], 1)

    # Add some things to Einfo
    Einfo['used_inds_all'] = used_inds
    Einfo['XiA_cell'] = XVi1
    Einfo['XiB_cell'] = XVi2
    Einfo['Robs'] = Robs # all cells, full indexing

    print( "Adjusted for cell %d: %d time points"%(cell_num, NT))

    return stim_all, Robs, used_cell, Ui, Xi, Einfo


def monocular_data_import( datadir, exptn, time_shift=1, num_lags=20):
    """
    Usage: stim, Robs, DFs, used_inds, Eadd_info = binocular_data_import( datadir, expt_num )
    Note: expt num is starting with 1
    block_output determines whether to fit using block info, or used_inds info (derived from blocks)
    num_lags is for purposes of used_inds
    """
    import scipy.io as sio

    l_files_to_use = np.add(list(range(16)), 1)  # good laminar-probe experiments
    u_files_to_use = [1, 2, 5, 6, 12]  # good utah-array experiments
    assert exptn <= len(l_files_to_use)+len(u_files_to_use), 'Expt number too high.'
    if exptn <= len(l_files_to_use):
        filename = 'expt'
        ee = l_files_to_use[exptn-1]
        is_utah = False
    else:
        filename = 'Uexpt'
        #utah_array[nn] = True
        ee = u_files_to_use[exptn-1 - len(l_files_to_use)]
        is_utah = True
    if ee < 10:
        filename += '0'
    filename += str(ee) + '.mat'         
    matdata = sio.loadmat( datadir+filename )

    sus = np.squeeze(matdata['goodSUs']) - 1  # switch from matlab indexing
    mus = np.squeeze(matdata['goodMUs']) - 1

    #print('SUs:', sus)
    NC = len(sus)
    layers = matdata['layers'][0,:]
    block_list = matdata['block_inds'] # note matlab indexing
    stim_all = NDNutils.shift_mat_zpad(matdata['stimulus'], time_shift, 0)
    NTtot, NX = stim_all.shape
    DFs = deepcopy(matdata['data_filters'][:,sus])
    Robs = deepcopy(matdata['binned_SU'][:,sus])
    RobsMU = deepcopy(matdata['binned_MUA'][:, mus])
    DFsMU = deepcopy(matdata['data_filters_MUA'][:,mus])

    # Break up into train and test blocks
    # Assemble train and test indices based on BIlist
    NBL = block_list.shape[0]
    Xb = np.arange(2, NBL, 5)  # Every fifth trial is cross-validation
    Ub = np.array(list(set(list(range(NBL)))-set(Xb)), dtype='int')
    
    # Make block indxs #used_inds = make_block_inds( block_list, gap=num_lags )
    # Also further modify DFs to incorporate gaps (for this purpose) 
    used_inds = []
    for nn in range(block_list.shape[0]):
        used_inds = np.concatenate( 
            (used_inds, 
            np.arange(block_list[nn,0]-1+num_lags, block_list[nn,1], dtype='int')),
            axis=0)
        DFs[np.arange(block_list[nn,0]-1, block_list[nn,0]+num_lags, dtype='int'), :] = 0.0
        DFsMU[np.arange(block_list[nn,0]-1, block_list[nn,0]+num_lags, dtype='int'), :] = 0.0
    
    #Ui, Xi = NDNutils.generate_xv_folds( len(used_inds) )
    #Rinds, TEinds = used_inds[Ui].astype(int), used_inds[Xi].astype(int)

    Eadd_info = {
        'expt_name': matdata['cur_expt'][0],
        'cortical_layer': layers, 
        'used_inds': used_inds, 
        'block_list': block_list,
        'sus': sus.astype(int), 
        'mus': mus.astype(int)}
        #'TRinds':TRinds, 'TEinds': TEinds, 
        #'TRblocks': Ub, 'TEblocks': Xb}
    return stim_all, Robs, DFs, RobsMU, DFsMU, Eadd_info

        