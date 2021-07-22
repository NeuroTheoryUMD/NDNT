## DATASETS that Dan needs
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm


class generic_recording(Dataset):
    """Dataset already in memory of form, stim, Robs, and possibly datafilters
    Note stimulus is not constrained to be any dim, but always NT x dims"""

    def __init__(self, stim, Robs, datafilters=None ):
        
        # map numpy arrays into tensors
        self.x = torch.tensor(stim.astype('float32'))
        self.y = torch.tensor(Robs.astype('float32'))
        if datafilters is not None:
            self.DFs = torch.tensor(datafilters.astype('float32'))
        else:
            self.DFs = torch.ones(Robs.shape, dtype=torch.float32)
        self.NC = Robs.shape[1]
        
    def __getitem__(self, index):
        
        return {'stim': self.x[index,:], 'robs':self.y[index,:], 'dfs': self.DFs[index,:]}
        
    def __len__(self):
        return len(self.x)
