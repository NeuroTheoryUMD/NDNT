## DATASETS that Dan needs
import torch
from torch.utils.data import Dataset

class generic_recording(Dataset):
    """Dataset already in memory of form, stim, Robs, and possibly datafilters
    Note stimulus is not constrained to be any dim, but always NT x dims"""

    def __init__(self, stim, Robs, datafilters=None, device=None, dtype=torch.float32):
        '''
        Arguments:
            stim: torch tensor of shape NT x dims
            Robs: torch tensor of shape NT x NC
            datafilters: torch tensor of shape NT x NC
            device: torch device
                default: None, will use the current device
            dtype: torch dtype
                default: torch.float32
        '''
        
        # map numpy arrays into tensors
        self.x = torch.tensor(stim, dtype=dtype)
        self.y = torch.tensor(Robs, dtype=dtype)
        if datafilters is not None:
            self.DFs = torch.tensor(datafilters, dtype=dtype)
        else:
            self.DFs = torch.ones(Robs.shape, dtype=torch.float32)
        
        if device:
            self.x =self.x.to(device)
            self.y =self.y.to(device)
            self.DFs =self.DFs.to(device)

        self.NC = Robs.shape[1]
        
    def __getitem__(self, index):
        
        return {'stim': self.x[index,:], 'robs':self.y[index,:], 'dfs': self.DFs[index,:]}
        
    def __len__(self):
        return len(self.x)
