Module NDNT.utils.DanUtils
==========================

Functions
---------

`binocular_data_import(datadir, expt_num)`
:   Usage: stim, Robs, DFs, used_inds, Eadd_info = binocular_data_import( datadir, expt_num )
    
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

`binocular_data_import_cell(datadir, expt_num, cell_num)`
:   Usage: stim, Robs, used_inds_cell, UiC, XiC = binocular_data_import_cell( datadir, expt_num, cell_num)
    
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

`chunker(seq, size)`
:   This function chunks a sequence into chunks of size size. (from Matt J)
    
    Args:
        seq: the sequence
        size: the size of the chunks
    
    Returns:
        a list of chunks

`design_matrix_drift(NT, anchors, zero_left=True, zero_right=False, const_right=False, to_plot=False)`
:   Produce a design matrix based on continuous data (s) and anchor points for a tent_basis.
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

`display_matrix(x, prec=3, spacing=4, number_rows=False, number_cols=False)`
:   

`dist_mean(p)`
:   

`figure_export(fig_handle, filename, bitmap=False, dpi=300)`
:   Usage: figure_export( fig_handle, filename, variable_list, bitmap=False, dpi=300)
    if bitmap, will use dpi and export as .png. Otherwise will export PDF

`filename_num2str(n, num_digits=2)`
:   

`find_peaks(x, clearance=10, max_peaks=10, thresh=13.0)`
:   Find maximum of peaks and then get rid of other points around it for plus/minus some amount

`fold_sample(num_items, folds=5, random_gen=False, which_fold=None)`
:   Divide fold sample deterministically or randomly distributed over number of items. More options
    can be added, but his captures the basics.

`imagesc(img, cmap=None, balanced=None, aspect=None, max=None, colrow=True, axis_labels=True, ax=None)`
:   Modifications of plt.imshow that choose reasonable defaults

`iterate_lbfgs(mod, dat, lbfgs_pars, train_inds=None, val_inds=None, tol=0.0001, max_iter=20, verbose=True)`
:   

`load_python_data(filename, show_keys=False)`
:   Load python data from standard binary file
    
    Args:
        filename: name of the file to save
        show_keys: to display components of dictionary (default=False)

`matlab_export(filename, variable_list)`
:   Export list of variables to .mat file

`max_multiD(k)`
:   

`monocular_data_import(datadir, exptn, time_shift=1, num_lags=20)`
:   Usage: stim, Robs, DFs, used_inds, Eadd_info = binocular_data_import( datadir, expt_num )
    Note: expt num is starting with 1
    block_output determines whether to fit using block info, or used_inds info (derived from blocks)
    num_lags is for purposes of used_inds

`save_python_data(filename, data)`
:   Saves python dictionary in standard binary file
    
    Args:
        filename: name of the file to save
        data: dictionary to save

`ss(num_rows=1, num_cols=1, row_height=3, rh=None, fighandle=False)`
:   

`subplot_setup(num_rows, num_cols, row_height=3, fig_width=16, fighandle=False)`
:   

`time_embedding_simple(stim, nlags)`
:   Simple time embedding: takes the stim and time-embeds with nlags
    If stim is multi-dimensional, it flattens. This is a numpy function