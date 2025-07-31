#Various
import sys
import os
import torch
import numpy as np
import scipy.io as sio
from scipy import ndimage
from copy import deepcopy
import h5py 
from datetime import date
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
import dill

# NDNT
import NDNT.utils as utils 
from NDNT.utils import fit_lbfgs
from NDNT.utils.DanUtils import ss
from NDNT.utils.DanUtils import imagesc
import NDNT.NDN as NDN
from NDNT.modules.layers import *
from NDNT.networks import *

#NTDatasets
import NTdatasets.conway.multi_datasets as multidata
from NTdatasets.generic import GenericDataset

#ColorDataUtils
import ColorDataUtils.ConwayUtils as CU
from ColorDataUtils.DDPIutils import DDPIutils
import ColorDataUtils.EyeTrackingUtils as ETutils
from ColorDataUtils.preprocessing_utils import *
import ColorDataUtils.CalibrationUtils as cal
from ColorDataUtils import readout_fit
from ColorDataUtils.sync_clocks import convertSamplesToTimestamps, alignTimestampToTimestamp
from ColorDataUtils.CloudMultiExpts import MultiExperiment
from ColorDataUtils import RFutils

"""
Some stuff to do here to make this work:
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

def init_vars(GPU = 0, project = None, datadir_input = None, dirname_input = None):
    """Initializes several important variables for data and modeling, and imports
    many important libraries.

    Args:
        GPU (int, optional): Which GPU to use (0 or 1). Defaults to 1.
        datadir (String, optional): defaults to a definition below, but can be set by argument instead
        dirname (String, optional): defaults to a definition below, but can be set by argument instead

    Returns:
        datadir: the directory where data will be pulled from. Governed by computer and user
        dirname: the working directoy. Governed by computer and user
        editor: used in file_info documents to remember who did editing
        myhost: computer
        device0: cpu device
        device: GPU device. defaults to CPU if no GPU is available.
    """
    user = os.getlogin()
    myhost = os.uname()[1] # get name of machine
    print(f"Running on Computer: {myhost} with user {user}")
    datadir = ""
    dirname = ""
    #Dr. Butts
    if user.lower() == 'dbutts':
        if myhost=='m1':                                        #m1
            datadir = '/home/dbutts/V1/B2data/'
            dirname = '/home/dbutts/V1/Binocular/Bworkspace/'
        if myhost=='ca3':                                       #ca3                                  
            datadir = '/home/DATA/ColorV1/'
            dirname = '/home/dbutts/ColorV1/CLRworkspace/'
        else:                                                   # older computers: MT
            datadir = '/home/dbutts/V1/Binocular/Data/'
            dirname = '/home/dbutts/V1/Binocular/Bworkspace/'  
        if project is not None:                                 #add project directory if desired
            datadir = ""
            dirname = ""
    #Isabel Fernandez
    if user.lower() == '':
        if myhost=='':
            datadir = ''
            dirname = ''
        if project is not None:
            datadir = ""
            dirname = ""
    print(user, myhost)
    #Jasper Coles Hood
    if user.lower() == 'jmch':
        if myhost=='ca1':                                       #ca1              
            datadir = '/Data/ColorV1/'
            dirname = '/home/jmch/'
        if project is not None:
            datadir = ""
            dirname = ""
    if datadir_input is not None:
        datadir = datadir_input
    if dirname_input is not None:
        dirname = dirname_input

    sys.path.insert(0, dirname) 
    device0 = torch.device("cpu")
    if GPU == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # then ada = cuda:0
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('Assigned:', device)  
    print(datadir)
    return datadir, dirname, device0, device