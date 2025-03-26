import os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # progress bar
from ..utils import save_checkpoint, ensure_dir
from .lbfgsnew import LBFGSNew

class Trainer:
    '''
    This is the most basic trainer. There are fancier things we could add (hooks, callbacks, etc.), but I don't understand them well enough yet.
    '''
    def __init__(self, optimizer=None, scheduler=None,
            device=None,
            optimize_graph=False,
            dirpath=os.path.join('.', 'checkpoints'),
            version=None,
            max_epochs=100,
            early_stopping=None,
            log_activations=False,
            scheduler_after='batch',
            scheduler_metric=None,
            accumulate_grad_batches=1,
            verbose=1,
            set_grad_to_none=False,
            save_epochs=False,
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
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimize_graph = optimize_graph
        self.log_activations = log_activations
        self.accumulate_grad_batches = accumulate_grad_batches
        self.verbose = verbose
        self.set_to_none = set_grad_to_none
        self.fullbatch = False
        
        ensure_dir(dirpath)

        self.save_epochs = save_epochs

        # auto version if version is None
        if version is None:
            # try to find version number
            import re
            dirlist = os.listdir(dirpath)            
            versionlist = [re.findall(r'(?!version)\d+', x) for x in dirlist]
            versionlist = [int(x[0]) for x in versionlist if not not x]
            if versionlist:
                max_version = max(versionlist)
            else:
                max_version = 0
            version = max_version + 1

        self.dirpath = os.path.join(dirpath, "version%d" % version)
        self.early_stopping = early_stopping

        # ensure_dir(self.dirpath)
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.logger = SummaryWriter(log_dir=self.dirpath, comment="version%d" % version) # use tensorboard to keep track of experiments
        self.version = version
        self.epoch = 0
        self.max_epochs = max_epochs
        self.n_iter = 0
        self.val_loss_min = np.Inf

        
        # scheduler defaults
        self.step_scheduler_after = scheduler_after # this is the only option for now
        self.step_scheduler_metric = scheduler_metric


    def fit(self, model, train_loader, val_loader, seed=None):
        """
        Fit the model to the data.

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            seed (int): Random seed for reproducibility.

        Returns:
            None
        """
        model = self.prepare_fit(model, seed=seed)

        # if we wrap training in a try/except block, can have a graceful exit upon keyboard interrupt
        try:            
            self.fit_loop(model, self.max_epochs, train_loader, val_loader)
            
        except KeyboardInterrupt: # user aborted training
            
            self.graceful_exit(model)
            return

        self.graceful_exit(model)

    def prepare_fit(self, model, seed=None):
        """
        This is called before fit_loop, and is used to set up the model and optimizer.

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            seed (int): Random seed for reproducibility.

        Returns:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
        """
        GPU_FLAG = torch.cuda.is_available()
        GPU_USED = self.device.type == 'cuda'
        if self.verbose > 0:
            print("\nGPU Available: %r, GPU Used: %r" %(GPU_FLAG, GPU_USED))

        if GPU_FLAG and GPU_USED:
            self.use_gpu = True
            torch.cuda.empty_cache()
        else:
            self.use_gpu = False

        self.use_closure = isinstance(self.optimizer, torch.optim.LBFGS) or isinstance(self.optimizer, LBFGSNew)

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
        
        model = model.to(self.device) # move model to device

        self.optimizer.zero_grad() # set gradients to zero

        return model
    # END trainer.prepare_fit()
    
    def fit_loop(self, model, epochs, train_loader, val_loader):
        """
        Main training loop. This is where the model is trained.

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            epochs (int): Number of epochs to train.
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.

        Returns:
            None
        """
        
        self.nbatch = len(train_loader)

        if self.fullbatch:
            self.accumulate_grad_batches = len(train_loader)
        self.accumulate_grad_batches = min([self.nbatch, self.accumulate_grad_batches])
        
        if self.log_activations:
            activations = {}
            self.hooks = []
            self.logged_parameters = []

            def hook(name):
                def hook_fn(m, i, o):
                    activations[name] = o

                return hook_fn

            # Hook the activations
            for name, layer in model.named_modules():
                if 'NL' in name or 'reg' in name or 'loss' in name:
                    continue

                self.logged_parameters.append(name)
                self.hooks.append(layer.register_forward_hook(hook(name)))

        # main loop for training
        for epoch in range(epochs):
            self.epoch += 1
            # train one epoch
            out = self.train_one_epoch(model, train_loader, self.epoch)
            self.logger.add_scalar('Loss/Train (Epoch)', out['train_loss'], self.epoch)
            train_loss = out['train_loss']
            #if self.verbose > 1:
            #    print('Train loss: %0.8f'%out['train_loss'])
            if np.isnan(train_loss):
                self.graceful_exit(model)

            if self.log_activations:
                for name in self.logged_parameters:
                    try:
                        self.logger.add_histogram(
                                    f"Activations/{name.replace('.', ' ')}/hist",
                                    activations[name].view(-1),
                                    self.epoch,
                                )

                        self.logger.add_scalar(
                                    f"Activations/{name.replace('.', ' ')}/mean",
                                    activations[name].mean(),
                                    self.epoch,
                                )

                        self.logger.add_scalar(
                                    f"Activations/{name.replace('.', ' ')}/std",
                                    activations[name]
                                    .std(dim=0)
                                    .mean(),
                                    self.epoch,
                                )
                    except:
                        pass

            # validate every epoch
            #if self.epoch % 1 == 0:
            out = self.validate_one_epoch(model, val_loader)
            if out['val_loss'] < self.val_loss_min:
                is_best=True
                self.val_loss_min = out['val_loss']
            else:
                is_best=False
            #self.val_loss_min = out['val_loss']
            #self.logger.add_scalar('Loss/Validation (Epoch)', self.val_loss_min, self.epoch)
            self.logger.add_scalar('Loss/Validation (Epoch)', out['val_loss'], self.epoch)
            
            if self.verbose==1:
                print("Epoch %d: train loss %.6f val loss %.6f" %(self.epoch, train_loss, out['val_loss']))
                
            # scheduler if scheduler steps at epoch level
            if self.scheduler:
                if self.step_scheduler_after == "epoch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = out[self.step_scheduler_metric]
                        self.scheduler.step(step_metric)
            
            # checkpoint
            self.checkpoint_model(model, self.epoch, is_best=is_best)

            # callbacks: e.g., early stopping
            if self.early_stopping:
                self.early_stopping(out['val_loss'])
                if self.early_stopping.early_stop:
                    if self.verbose > 0:
                        print("Early stopping")
                    break

    def validate_one_epoch(self, model, val_loader):
        """
        validation step for one epoch

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            val_loader (DataLoader): Validation data.

        Returns:
            dict: validation loss
        """
        # bring models to evaluation mode
        model.eval()
        runningloss = 0
        nsteps = len(val_loader)
        if self.verbose > 1:
            pbar = tqdm(val_loader, total=nsteps, bar_format=None)
            pbar.set_description("Validating ver=%d" %self.version)
        else:
            pbar = val_loader

        with torch.no_grad():
            for batch_idx, data in enumerate(pbar):
                
                # Data to device if it's not already there
                for dsub in data:
                    if data[dsub].device != self.device:
                        data[dsub] = data[dsub].to(self.device)
                    #if len(data[dsub].shape) > 2:  # kluge to get around collate_fn
                    #    data[dsub] = data[dsub].flatten(end_dim=1)

                
                out = model.validation_step(data)

                runningloss += out['val_loss'].item()

                if self.verbose > 1:
                    pbar.set_postfix({'val_loss': str(np.round(runningloss/(batch_idx+1), 6))})

        return {'val_loss': runningloss/nsteps}
            
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
        if self.verbose > 1:
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

        return {'train_loss': runningloss/(batch_idx + 1)} # should this be an aggregate out?


    def train_one_step(self, model, data, batch_idx=None):
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
        
        loss = out['loss']
        # with torch.set_grad_enabled(True):
        loss.backward()
        # loss.backward(create_graph=True)

        # optimization step
        if (((batch_idx + 1) % self.accumulate_grad_batches) == 0) or (batch_idx + 1 == self.nbatch):
            self.n_iter += 1
            self.logger.add_scalar('Loss/Loss', out['loss'].item(), self.n_iter)
            self.logger.add_scalar('Loss/Train', out['train_loss'].item(), self.n_iter)
            try:
                self.logger.add_scalar('Loss/Reg', out['reg_loss'].item(), self.n_iter)
            except:
                pass

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
            'net': deepcopy(state), # the model state puts all the parameters in a dict
            'epoch': epoch,
            'optim': self.optimizer.state_dict()
        } # probably also want to track n_ter =>  'n_iter': n_iter,

        if is_best:
            torch.save(model, os.path.join(self.dirpath, 'model_best.pt'))
        
        if self.save_epochs:
            #ensure_dir(self.dirpath + 'epochs/')
            torch.save(model, os.path.join(self.dirpath, f'model_epoch{epoch}.pt'))

        save_checkpoint(cpkt, os.path.join(self.dirpath, 'model_checkpoint.ckpt'), is_best=is_best)
    # END Trainer.checkpoint_model()
    
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

class LBFGSTrainer(Trainer):
    """
    This class is for training with the LBFGS optimizer. It is a subclass of Trainer.
    """

    def __init__(self,
            full_batch=False,
            **kwargs,
            ):
        """
        Args:
            full_batch (bool): Whether to use full batch optimization.
        """
        super().__init__(**kwargs)

        self.fullbatch = full_batch
    
    def fit(self, model, train_loader, val_loader=None, seed=None):
        """
        Fit the model to the data.

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            seed (int): Random seed for reproducibility.

        Returns:
            None
        """
        self.prepare_fit(model, seed=seed)

        if isinstance(train_loader, dict):
            ''' fit entire dataset at once'''
            self.fit_data_dict(model, train_loader)

            if val_loader is None:
                val_loader = train_loader
                
            if isinstance(val_loader, dict):
                # for dsub in val_loader:
                #     val_loader[dsub] = val_loader[dsub].to(self.device)
                
                # copy the model to the validation set device
                dsubs = list(val_loader.keys())
                model = model.to(val_loader[dsubs[0]].device)
                out = model.validation_step(val_loader)
                # copy the model back
                model = model.to(self.device)
            else:
                out = self.validate_one_epoch(val_loader)
                if np.isnan(out['val_loss']):
                    return
            self.val_loss_min = out['val_loss'].item()
            self.logger.add_scalar('Loss/Validation (Epoch)', self.val_loss_min, self.epoch)

        else:
            ''' fit one epoch at a time (can still accumulate the grads across epochs, but is much slower. handles datasets that cannot fit in memory'''
            if self.fullbatch:
                self.accumulate_grad_batches = len(train_loader)
                print('Accumulated batches set to %d'%self.accumulate_grad_batches)

            # if we wrap training in a try/except block, can have a graceful exit upon keyboard interrupt
            try:            
                self.fit_loop(model, self.max_epochs, train_loader, val_loader)
                
            except KeyboardInterrupt: # user aborted training
                
                self.graceful_exit(model)
                return

        self.graceful_exit(model)

    def fit_data_dict(self, model, train_data):
        """
        Fit data that is provided in a dictionary.

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            train_data (dict): Training data.

        Returns:
            None
        """
        for dsub in train_data:
            train_data[dsub] = train_data[dsub].to(self.device)
            
        def closure():
        
            self.optimizer.zero_grad(set_to_none=self.set_to_none)
            loss = torch.zeros(1, device=self.device)

            out = model.training_step(train_data)
            loss = out['loss']
            if np.isnan(loss.item()):
                return loss
            loss.backward()
            
            # torch.cuda.empty_cache()
            if self.verbose > 1:
                print('Iteration: {} | Loss: {}'.format(self.optimizer.state_dict()['state'][0]['n_iter'], loss.item()))
            
            return loss

        loss = self.optimizer.step(closure)
        self.optimizer.zero_grad(set_to_none=self.set_to_none)
        if self.use_gpu:
                torch.cuda.empty_cache()

        return {'train_loss': loss}
        
        
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
        model.train() # set model to training mode

        self.iter = 1
        max_iter = self.optimizer.param_groups[0]['max_iter']

        def closure():

            # pbar = tqdm(train_loader, total=self.nbatch, bar_format=None) # progress bar for looping over data
            # pbar.set_description("Epoch %i, iter %d/%d" %(epoch, self.iter, max_iter))
            
            self.optimizer.zero_grad(set_to_none=self.set_to_none)
            loss = torch.zeros(1, device=self.device)
            if self.verbose > 1:
                pbar = tqdm(train_loader, total=len(train_loader), bar_format=None)
                pbar.set_description("Training ver=%d" %self.version)
            else:
                pbar = train_loader

            for batch_idx, data in enumerate(pbar):
            
                if batch_idx < self.accumulate_grad_batches:

                    if self.use_gpu:
                        for dsub in data:
                            if data[dsub].device != self.device:
                                data[dsub] = data[dsub].to(self.device)

                    out = model.training_step(data)
                    if torch.isnan(out['loss']):
                        break
                    loss += out['loss']
                    self.n_iter += 1
                    self.logger.add_scalar('Loss/Loss', out['loss'].item(), self.n_iter)
                    self.logger.add_scalar('Loss/Train', out['train_loss'].item(), self.n_iter)
                    self.logger.add_scalar('Loss/Reg', out['reg_loss'].item(), self.n_iter)
                    torch.cuda.empty_cache()
                    if self.verbose > 1:
                        # update progress bar
                        pbar.set_postfix({'train_loss': str(np.round(loss.detach().item()/(batch_idx + 1), 6)),
                            'fevals': self.optimizer.state_dict()['state'][0]['func_evals'],
                            'n_iter': self.optimizer.state_dict()['state'][0]['n_iter']})

                else:
                    break

            loss/=self.accumulate_grad_batches  # note this assumes constant batch_size to be fully accurate
            loss.backward()

            self.iter += 1

            return loss

        loss = self.optimizer.step(closure)
        print("Epoch %d (%d iter): loss = %02.3f" %(epoch, self.iter, loss.detach().item()))
        #self.optimizer.zero_grad(set_to_none=self.set_to_none) # zero the gradients for the next batch
        
        if self.use_gpu:
                torch.cuda.empty_cache()

        return {'train_loss': loss} # should this be an aggregate out?


class TemperatureCalibratedTrainer(Trainer):
    """
    This class is for training with temperature calibration. It is a subclass of Trainer.
    """

    def __init__(self,
            **kwargs,
            ):
        """
        Args:
            None
        """
        super().__init__(**kwargs)

    def validate_one_epoch(self, model, val_loader):
        """
        Validation step for one epoch.

        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            val_loader (DataLoader): Validation data.

        Returns:
            dict: validation loss
        """
        # bring models to evaluation mode
        model.eval()
        assert hasattr(model, 'temperature'), 'Model must have a temperature attribute'
        assert hasattr(model, 'set_temperature'), 'Model must have a set_temperature method'

        model.temperature.requries_grad = True
        after_temperature_nll = model.set_temperature(val_loader, self.device)
        model.temperature.requries_grad = False

        return {'val_loss': after_temperature_nll}