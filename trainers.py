
'''
Trainers for NDN
'''
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from tqdm import tqdm # progress bar
from NDNutils import ModelSummary, save_checkpoint, ensure_dir


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    
    def __call__(self, val_loss):

        score = -val_loss
        if self.verbose:
            print("EarlyStopping score = {}".format(score))

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class Trainer:
    '''
    This is the most basic trainer. There are fancier things we could add (hooks, callbacks, etc.), but I don't understand them well enough yet.
    '''
    def __init__(self, model=None, optimizer=None, scheduler=None,
            device=None,
            optimize_graph=False,
            dirpath=os.path.join('.', 'checkpoints'),
            num_gpus=None,
            version=None,
            max_epochs=100,
            early_stopping=None):
        '''
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.

            optimizer (torch.optim): Pytorch optimizer.

            device (torch.device): Device to train on
                            Default: will use CUDA if available
            scheduler (torch.scheduler): learning rate scheduler
                            Default: None
            dirpath (str): Path to save checkpoints
                            Default: current directory
            multi_gpu (bool): Whether to use multiple GPUs
                            Default: False
            max_epochs (int): Maximum number of epochs to train
                            Default: 100
            early_stopping (EarlyStopping): If not None, will use this as the early stopping callback.
                            Default: None
            optimize_graph (bool): Whether to optimize graph before training
        '''
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimize_graph = optimize_graph
        
        ensure_dir(dirpath)

        # auto version if version is None
        if version is None:
            # try to find version number
            import re
            dirlist = os.listdir(dirpath)            
            versionlist = [re.findall('(?!version)\d+', x) for x in dirlist]
            versionlist = [int(x[0]) for x in versionlist if not not x]
            if versionlist:
                max_version = max(versionlist)
            else:
                max_version = 0
            version = max_version + 1

        self.dirpath = os.path.join(dirpath, "version%d" % version)
        if num_gpus:
            if num_gpus > 1:
                self.multi_gpu = True
            else:
                self.multi_gpu = False
        else:
            self.multi_gpu = False
        self.early_stopping = early_stopping

        # ensure_dir(self.dirpath)
        
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.logger = SummaryWriter(log_dir=self.dirpath, comment="version%d" % version) # use tensorboard to keep track of experiments
        self.version = version
        self.model = model # initialize model attribute
        self.epoch = 0
        self.max_epochs = max_epochs
        self.n_iter = 0
        self.val_loss_min = np.Inf
        
        # scheduler defaults
        self.step_scheduler_after = 'epoch' # this is the only option for now
        self.step_scheduler_metric = None


    def fit(self, model, train_loader, val_loader, seed=None):
        
        self.model = model # can we just assign this here? --> makes it look more like the way lightning trainer is called (with model passed into fit)

        GPU_FLAG = torch.cuda.is_available()
        GPU_USED = self.device.type == 'cuda'
        print("\nGPU Available: %r, GPU Used: %r" %(GPU_FLAG, GPU_USED))

        # main training loop
        if self.optimize_graph:
            torch.backends.cudnn.benchmark = True # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
            # Note: On Nvidia GPUs you can add the following line at the beginning of our code.
            # This will allow the cuda backend to optimize your graph during its first execution.
            # However, be aware that if you change the network input/output tensor size the graph
            # will be optimized each time a change occurs. This can lead to very slow runtime and out of memory errors.
            # Only set this flag if your input and output have always the same shape.
            # Usually, this results in an improvement of about 20%.
        
        if seed is not None:
            # set flags / seeds    
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # if more than one device, use parallel training
        if torch.cuda.device_count() > 1 and self.multi_gpu:
            print("Using", torch.cuda.device_count(), "GPUs!") # this should be specified in requewstee
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)
        
        # self.model.to(self.device) # move model to device
        #if next(self.model.parameters()).device != self.device:
        print("Moving model to %s" %self.device)
        self.model.to(self.device)

        # Print Model Summary: has to happen after model is moved to device
        # _ = ModelSummary(self.model, train_loader.dataset[0]['stim'].shape, batch_size=train_loader.batch_size, device=self.device, dtypes=None)

        # if we wrap training in a try/except block, can have a graceful exit upon keyboard interrupt
        try:            
            self.fit_loop(self.max_epochs, train_loader, val_loader)
            
        except KeyboardInterrupt: # user aborted training
            
            self.graceful_exit()
            return

        self.graceful_exit()
    
    def fit_loop(self, epochs, train_loader, val_loader):
        # main loop for training
        for epoch in range(epochs):
            self.epoch = epoch
            # train one epoch
            out = self.train_one_epoch(train_loader, epoch)
            self.logger.add_scalar('Loss/Train (Epoch)', out['train_loss'], epoch)

            # validate every epoch
            if epoch % 1 == 0:
                out = self.validate_one_epoch(val_loader)
                self.val_loss_min = out['val_loss']
                self.logger.add_scalar('Loss/Validation (Epoch)', self.val_loss_min, epoch)
            
            # scheduler if scheduler steps at epoch level
            if self.scheduler:
                if self.step_scheduler_after == "epoch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            
            # checkpoint
            self.checkpoint_model(epoch)

            # callbacks: e.g., early stopping
            if self.early_stopping:
                self.early_stopping(out['val_loss'])
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

    def validate_one_epoch(self, val_loader):
        # validation step for one epoch

        # bring models to evaluation mode
        self.model.eval()
        runningloss = 0
        nsteps = len(val_loader)
        pbar = tqdm(val_loader, total=nsteps, bar_format=None)
        pbar.set_description("Validating ver=%d" %self.version)
        with torch.no_grad():
            for data in pbar:
                
                # Data to device if it's not already there
                for dsub in data:
                    if data[dsub].device != self.device:
                        data[dsub] = data[dsub].to(self.device)
                
                if isinstance(self.model, nn.DataParallel):
                    out = self.model.module.validation_step(data)
                else:
                    out = self.model.validation_step(data)

                runningloss += out['val_loss'].item()/nsteps
                pbar.set_postfix({'val_loss': runningloss})

        return {'val_loss': runningloss}
            
    def train_one_epoch(self, train_loader, epoch=0):
        # train for one epoch
        
        self.model.train() # set model to training mode

        runningloss = 0
        nsteps = len(train_loader)
        pbar = tqdm(train_loader, total=nsteps, bar_format=None) # progress bar for looping over data
        pbar.set_description("Epoch %i" %epoch)
        for data in pbar:
            # Data to device if it's not already there
            moved_to_device = False
            for dsub in data:
                if data[dsub].device != self.device:
                    moved_to_device = True
                    data[dsub] = data[dsub].to(self.device)
            
            # handle optimization step
            if isinstance(self.optimizer, torch.optim.LBFGS):
                out = self.train_lbfgs_step(data)
            else:
                out = self.train_one_step(data)
            
            # Move Data off device
            if moved_to_device:
                for dsub in data:
                    data[dsub] = data[dsub].to('cpu')

            self.n_iter += 1
            self.logger.add_scalar('Loss/Train', out['train_loss'], self.n_iter)

            runningloss += out['train_loss']/nsteps
            # update progress bar
            pbar.set_postfix({'train_loss': runningloss})
        
        return {'train_loss': runningloss} # should this be an aggregate out?

    def train_lbfgs_step(self, data):
        # # Version 1: This version is based on the torch.optim.lbfgs implementation
        self.optimizer.zero_grad()

        def closure():
            self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                out = self.model.training_step(data)
            
            loss = out['loss']
            loss.backward()

            return loss
            
        self.optimizer.step(closure)
            
        # calculate the loss again for monitoring
        loss = closure()
    
        return {'train_loss': loss.detach().item()}


    def train_one_step(self, data):

        self.optimizer.zero_grad() # zero the gradients
        if isinstance(self.model, nn.DataParallel):
            out = self.model.module.training_step(data)
        else:
            out = self.model.training_step(data)

        loss = out['loss']
        # with torch.set_grad_enabled(True):
        loss.backward()
        self.optimizer.step()
            
        if self.scheduler:
            if self.step_scheduler_after == "batch":
                if self.step_scheduler_metric is None:
                    self.scheduler.step()
                else:
                    step_metric = self.name_to_metric(self.step_scheduler_metric)
                    self.scheduler.step(step_metric)
        
        return {'train_loss': loss.detach().item()}
    
    def checkpoint_model(self, epoch=None):
        if isinstance(self.model, nn.DataParallel):
            state = self.model.module.state_dict()
        else:
            state = self.model.state_dict()
        
        if epoch is None:
            epoch = self.epoch

        # check point the model
        cpkt = {
            'net': state, # the model state puts all the parameters in a dict
            'epoch': epoch,
            'optim': self.optimizer.state_dict()
        } # probably also want to track n_ter =>  'n_iter': n_iter,

        save_checkpoint(cpkt, os.path.join(self.dirpath, 'model_checkpoint.ckpt'))
    
    def graceful_exit(self):
        print("Done fitting")
        # to run upon keybord interrupt
        self.checkpoint_model() # save checkpoint

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module # get the non-data-parallel model

        self.model.eval()

        # save model
        torch.save(self.model, os.path.join(self.dirpath, 'model.pt'))

        # log final value of loss along with hyperparameters
        defopts = dict()
        defopts['model'] = self.model.__class__.__name__
        defopts['optimizer'] = self.optimizer.__class__.__name__
        defopts.update(self.optimizer.defaults)
        newopts = dict()
        for k in defopts.keys():
            if isinstance(defopts[k], (int, float, str, bool, torch.Tensor)):
                newopts[k] = defopts[k]
    
        # self.logger.export_scalars_to_json(os.path.join(self.dirpath, "all_scalars.json"))
        self.logger.close()