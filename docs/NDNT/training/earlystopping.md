Module NDNT.training.earlystopping
==================================

Classes
-------

`EarlyStopping(patience=7, verbose=0, delta=0, trace_func=<built-in function print>)`
:   Pytorch module that gets plugged into the trainer by NDN in order to early stops the training 
    if validation loss doesn't improve after a given patience.
    
    Args:
        patience (int): How long to wait after last time validation loss improved. (Default: 7)
        verbose (int): If >2, prints a message for each validation loss improvement. (Default: 0)
        delta (float): Minimum change in the monitored quantity to qualify as an improvement. (Default: 0)
        trace_func (function): trace print function. (Default: print)