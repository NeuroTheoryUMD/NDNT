"""
To make this work:
1) edit this file below to include your data paths and dirnames

2) Run in command line: 
<     echo 'export PYTHONPATH="/home/jmch/Code/NDNT/imports:$PYTHONPATH"' >> ~/.bashrc     >
    Where the '/home/jmch/Code' is the path to the folder containing NDNT
    This will write the path to your .bashrc file, so that it is always included
    You will need to restart jupyter notebooks, and VSCode needs additional steps.
    
3) To check that it worked: in any notebook, run [ import os ] 
                                               and [ sys.path ]
     You should see the path to your NDNT/imports in the list of paths.
     
4) For any notebook, run [ from imports import * ]
                      and [ datadir, dirname, device0, device = init_vars() ]
    All imports and initializations should be done.
 
5) VSCode has some really weird problems with updating its python environment variables.
    If it doesn't work automatically, here are some steps:
    a) create a file in your workspace root (where .vscode/ is) named .env
    b) in that file, write the line: <
        PYTHONPATH=/path/from/home/to/NDNT/imports              
    >
    c) open your vscode settings.json file and add the line: <          
    "python.envFile": "$/home/jmch/.env", 
    >     (or similar path to your .env file)
    d) in command line, run:    (while inside of conda environment) <
        echo "export PYTHONPATH=/path/to/NDNT/imports" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
        mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
        echo "unset PYTHONPATH" > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
        >
        (setting /path/to/NDNT/imports to the path where you have NDNT imports)
    e) restart everything, then try step 4
"""

##### BEGIN AUTOMATIC IMPORTS ######
import sys
import os
import torch
import numpy as np
import scipy.io as sio
from copy import deepcopy
import h5py 
from time import time
import matplotlib.pyplot as plt

user = os.getlogin()
myhost = os.uname()[1] # get name of machine

# Default code directory for users in general 
codedir = '/home/' + user + '/Code/'
#device0 = torch.device("cpu")  # the same for any computer

# User-specific cases (if needed)
if user.lower() in ['dbutts', 'dab']:
    if myhost == 'hoser':
        codedir = '/Users/dbutts/Code/'
# Add any other user/hostname specific exceptions to coding directories

sys.path.insert(0, codedir) 

### GENERAL LAB IMPORTS (not project-specific)
# NDNT -- utitilies
import NDNT.utils as utils 
from NDNT.utils import fit_lbfgs
from NDNT.utils.DanUtils import ss
from NDNT.utils.DanUtils import imagesc
# NDNT -- base
import NDNT.NDN as NDN
from NDNT.modules.layers import *
from NDNT.networks import *

# NTdatasets
from NTdatasets.generic import GenericDataset

del user, myhost, codedir
###### END AUTOMATIC IMPORTS #####


def init_vars(globs, GPU = 0, project = None, datadir_input = None, dirname_input = None):
    """
    Initializes several important variables for data and modeling, and imports important libraries.

    Args:
        GPU (int, optional): Which GPU to use (0 or 1). Defaults to 1.
        datadir_input (String, optional): defaults to a definition below, but can be set by argument instead
        dirname_input (String, optional): defaults to a definition below, but can be set by argument instead

    Returns:
        datadir: the directory where data will be pulled from. Governed by computer and user
        dirname: the working directoy. Governed by computer and user
        device: GPU device. defaults to CPU if no GPU is available.
    """
    user = os.getlogin()
    myhost = os.uname()[1] # get name of machine
    print(f"Running on {myhost} with user {user}")

    ########## DEFAULTS ##########
    dirname = '/home/' + user + '/'    
    base_datadir = ''
    if project is None:
        project = 'ColorV1'
        print( "Project: ", project ) # display only if left blank so default is clear

    ########## Computer-specific data directories and GPU settings ##########
    if myhost in ['ca1', 'm1']:
        base_datadir = '/Data/'
    elif myhost == 'ca3':
        base_datadir = '/home/DATA/'
        print('  cuda0:', torch.cuda.get_device_properties(0).name)
        print('  cuda1:', torch.cuda.get_device_properties(1).name)
        GPU = abs((torch.cuda.get_device_properties(0).minor == 9) - (GPU==0))
    
    device0 = torch.device("cpu")
    if GPU == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # then ada = cuda:0
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('Device assigned:', device)  

    ########## SHARED PROJECT EXTRAS (Datadir and Imports) ##########
    ##### ColorV1 project ######
    if project.lower() == 'colorv1':
        #from ColorDataUtils.postprocess import postprocess
        #global CU, DDPIutils, ETutils, readout_fit, pproc, cal, multExpt, RFutils, multidata
        import ColorDataUtils.ConwayUtils as CU
        import ColorDataUtils.DDPIutils as DDPIutils
        import ColorDataUtils.EyeTrackingUtils as ETutils
        import ColorDataUtils.readout_fit as readout_fit
        import ColorDataUtils.postprocessing_utils as pproc
        import ColorDataUtils.CalibrationUtils as cal
        import ColorDataUtils.CloudMultiExpts as multExpt
        import ColorDataUtils.RFutils as RFutils
        import NTdatasets.conway.multi_datasets as multidata
        globs['CU'] = CU
        globs['DDPIutils'] = DDPIutils
        globs['ETutils'] = ETutils
        globs['readout_fit'] = readout_fit
        globs['pproc'] = pproc
        globs['cal'] = cal
        globs['multExpt'] = multExpt
        globs['RFutils'] = RFutils
        globs['multidata'] = multidata
        datadir = base_datadir + 'ColorV1/'

    ##### SimCloud project ######
    elif project.lower() == 'simcloud':
        #global SimCloudData, readout_fit
        import NTdatasets.conway.synthcloud_datasets as syncloud
        import ColorDataUtils.readout_fit as readout_fit
        globs['SimCloudData'] = syncloud
        globs['readout_fit'] = readout_fit

        datadir = base_datadir + 'Antolik/'
    
    ##### OTHER PROJECTS ######

    ########## USER-SPECIFIC INFORMATION ##########
    ##### DAN BUTTS  #####
    if user.lower() in ['dbutts', 'dab']:
        # Overwrite general defaults for my laptop
        if myhost=='hoser': 
            base_datadir = '/Users/dbutts/Data/'
            dirname = '/Users/dbutts/Projects/'
        if project.lower() == 'colorv1':
            datadir = base_datadir + 'ColorV1/'
            dirname = dirname + 'ColorV1/'
        elif project.lower() == 'simcloud':
            datadir = base_datadir + 'Antolik/'
            dirname = dirname + 'Antolik/'

    ##### ISABEL FERNANDEZ #####
    elif user.lower() == 'ifernand':
        if project.lower() == 'cloudsim':
            if myhost == 'sc':
                datadir = '/home/ifernand/Cloud_SynthData_Proj/data/'
                #dirname = ''

    ##### JASPER COLES HOOD #####
    elif user.lower() == 'jmch':
        if myhost=='ca1':
            #datadir = '/Data/ColorV1/'
            dirname = '/home/jmch/'
        if project != 'ColorV1':
            datadir = ""
            dirname = ""

    if datadir_input is not None:
        datadir = datadir_input
    if dirname_input is not None:
        dirname = dirname_input

    print( "Datadir: %s\nDirname: %s"%(datadir, dirname) )
    return datadir, dirname, device, device0
