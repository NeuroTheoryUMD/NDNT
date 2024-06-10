import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # progress bar
from ..utils import save_checkpoint, ensure_dir
from .trainer import Trainer

class DualTrainer(Trainer):
    '''
    This is the most basic trainer. There are fancier things we could add (hooks, callbacks, etc.), but I don't understand them well enough yet.
    '''
    def __init__(self, 
                 #optimizer=None, scheduler=None, version=None, 
                 #device=None, optimize_graph=False,
                 #dirpath=os.path.join('.', 'checkpoints'),
                 #max_epochs=100, early_stopping=None, accumulate_grad_batches=1,
                 #log_activations=False, scheduler_after='batch',
                 #scheduler_metric=None,
                 #verbose=True, set_grad_to_none=False,
                 optimizer2=None,
                 **kwargs):
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

        # PASS THIS ALL ONTO REGULAR TRAINER INIT -- the fit is what is diff
        super().__init__(**kwargs)
        self.epoch_optimizer = optimizer2
    # END DualTrainer.__init__

    def prepare_fit(self, model, seed=None):
        """
        This is called before fit_loop, and is used to set up the model and optimizer(s).

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            seed (int): Random seed for reproducibility.

        Returns:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
        """
        super().prepare_fit(model=model,seed=seed)
        self.epoch_optimizer.zero_grad() # set gradients to zero

        return model

    #def fit(self, model, train_loader, val_loader, seed=None):
    #def fit_loop(self, model, par_group1, par_group2, epochs, train_loader, val_loader):
       
    def train_one_epoch(self, model, train_loader, epoch=0):
        """
        Train for one epoch.

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            train_loader (DataLoader): Training data.
            epoch (int): Current epoch.

        Returns:
            dict: training loss
        """
        assert not self.use_closure, "Trainer: LBFGS and LBFGSNew are supported by LBFGSTrainer"
        
        model.train() # set model to training mode

        runningloss = 0
        if self.verbose == 2:
            pbar = tqdm(train_loader, total=self.nbatch, bar_format=None) # progress bar for looping over data
            pbar.set_description("Epoch %i" %epoch)
        else:
            pbar = train_loader
        
        for batch_idx, data in enumerate(pbar):
            
            # handle optimization step
            out = self.train_one_step(model, data, batch_idx=batch_idx)

            if self.use_gpu:
                torch.cuda.empty_cache()

            if np.isnan(out['train_loss']):
                break
            runningloss += out['train_loss']
            if self.verbose > 1:
                # update progress bar
                pbar.set_postfix({'train_loss': str(np.round(runningloss/(batch_idx + 1), 6))})

        # Step with second optimizer (epoch-level) here
        self.epoch_optimizer.step()
        self.epoch_optimizer.zero_grad(set_to_none=self.set_to_none)

        return {'train_loss': runningloss/(batch_idx + 1)} # should this be an aggregate out?

    def train_one_step(self, model, par_group1, par_group2, data, batch_idx=None):
        """
        Train for one step.

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            data (dict): Training data.
            batch_idx (int): Current batch index.

        Returns:
            dict: training loss
        """
        # Data to device if it's not already there
        if self.use_gpu:
            for dsub in data:
                if data[dsub].device != self.device:
                    data[dsub] = data[dsub].to(self.device)
                # if trainer concatentates batches incorrectly --- this is a kluge
                #if len(data[dsub].shape) > 2:
                #    data[dsub] = data[dsub].flatten(end_dim=1) # kluge to get around collate_fn
        out = model.training_step(data)
        
        self.n_iter += 1
        self.logger.add_scalar('Loss/Loss', out['loss'].item(), self.n_iter)
        self.logger.add_scalar('Loss/Train', out['train_loss'].item(), self.n_iter)
        try:
            self.logger.add_scalar('Loss/Reg', out['reg_loss'].item(), self.n_iter)
        except:
            pass

        loss = out['loss']
        # with torch.set_grad_enabled(True):
        loss.backward()
        # loss.backward(create_graph=True)
        
        # optimization step
        if ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (batch_idx + 1 == self.nbatch):
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=self.set_to_none)
            # self.optimizer.zero_grad() # zero the gradients for the next batch
            
            if self.scheduler:
                if self.step_scheduler_after == "batch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = out[self.step_scheduler_metric]
                        self.scheduler.step(step_metric)
        
        return {'train_loss': loss.detach().item()}
    
    def checkpoint_model(self, model, epoch=None, is_best=False):
        """
        Checkpoint the model.

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            epoch (int): Current epoch.
            is_best (bool): Whether this is the best model.

        Returns:
            None
        """
        state = model.state_dict()
        
        if epoch is None:
            epoch = self.epoch

        # check point the model
        cpkt = {
            'net': state, # the model state puts all the parameters in a dict
            'epoch': epoch,
            'optim': self.optimizer.state_dict()
        } # probably also want to track n_ter =>  'n_iter': n_iter,

        if is_best:
            torch.save(model, os.path.join(self.dirpath, 'model_best.pt'))

        save_checkpoint(cpkt, os.path.join(self.dirpath, 'model_checkpoint.ckpt'), is_best=is_best)
    
    def graceful_exit(self, model):
        """
        Graceful exit. This is called at the end of training or it if gets interrupted.

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.

        Returns:
            None
        """
        if self.verbose > 0:
            print("Done fitting")
        # to run upon keybord interrupt
        self.checkpoint_model(model) # save checkpoint

        if self.device.type == 'cuda':
            model.cpu()
            torch.cuda.empty_cache()

        model.eval()

        if self.log_activations:
            for hook in self.hooks:
                hook.remove()

        # save model
        try:
            torch.save(model, os.path.join(self.dirpath, 'model.pt'))
        except:
            torch.save({'state_dict': model.state_dict()}, os.path.join(self.dirpath, 'model.pt'))

        # log final value of loss along with hyperparameters
        defopts = dict()
        defopts['model'] = model.__class__.__name__
        defopts['optimizer'] = self.optimizer.__class__.__name__
        defopts.update(self.optimizer.defaults)
        newopts = dict()
        for k in defopts.keys():
            if isinstance(defopts[k], (int, float, str, bool, torch.Tensor)):
                newopts[k] = defopts[k]
    
        # self.logger.export_scalars_to_json(os.path.join(self.dirpath, "all_scalars.json"))
        self.logger.close()

        # if best model checkpoint exists, load that and save it
        
        # if os.path.exists(os.path.join(self.dirpath, 'best_model.ckpt')):
        #     os.path.exists(os.path.join(self.dirpath, 'best_model.ckpt'))                
        #     'best_model.pt'
