import numpy as np
import scipy.io as sio
from copy import deepcopy
import NDNT.utils.NDNutils as NDNutils
import matplotlib.pyplot as plt

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


###### GENERAL UTILITIES NOT SPECIFIC TO NDN ######
def subplot_setup(num_rows, num_cols, row_height=2, fighandle=False):
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
    fig.set_size_inches(16, row_height*num_rows)
    if fighandle is True:
        return fig
    

def filename_num2str( n, num_digits=2 ):
    num_places = int(np.ceil(np.log10(n)+0.001))
    place_shift = int(np.maximum(num_digits-num_places, 0))
    s = '0'*place_shift + str(n%(10**num_digits))[:]
    return s


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
