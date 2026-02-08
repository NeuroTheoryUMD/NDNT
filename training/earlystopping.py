import numpy as np

class EarlyStopping:
    """
    Pytorch module that gets plugged into the trainer by NDN in order to early stops the training 
    if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=0, delta=0, trace_func=print):
        """            
        Args:
            patience (int): How long to wait after last time validation loss improved. (Default: 7)
            verbose (int): If >2, prints a message for each validation loss improvement. (Default: 0)
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. (Default: 0)
            trace_func (function): trace print function. (Default: print)
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 1.0e8 #np.Inf
        self.delta = delta
        self.trace_func = trace_func
    
    def __call__(self, val_loss):

        score = -val_loss
        #if self.verbose:
        #    print("EarlyStopping score = {}".format(-score))

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose > 2:
                self.trace_func(f'=> EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
