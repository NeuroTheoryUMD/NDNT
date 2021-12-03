import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # progress bar
from NDNT.utils import save_checkpoint, ensure_dir
from .lbfgsnew import LBFGSNew
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
            early_stopping=None,
            log_activations=True,
            step_scheduler_after='batch',
            accumulate_grad_batches=1,
            **kwargs
            ):
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
        self.step_scheduler_after = step_scheduler_after # this is the only option for now
        self.step_scheduler_metric = None


    def fit(self, model, train_loader, val_loader, seed=None):
        
        self.model = model # can we just assign this here? --> makes it look more like the way lightning trainer is called (with model passed into fit)

        self.prepare_fit(seed=seed)
        # Print Model Summary: has to happen after model is moved to device
        # _ = ModelSummary(self.model, train_loader.dataset[0]['stim'].shape, batch_size=train_loader.batch_size, device=self.device, dtypes=None)

        # if we wrap training in a try/except block, can have a graceful exit upon keyboard interrupt
        try:            
            self.fit_loop(self.max_epochs, train_loader, val_loader)
            
        except KeyboardInterrupt: # user aborted training
            
            self.graceful_exit()
            return

        self.graceful_exit()

    def prepare_fit(self, seed=None):
        '''
        This is called before fit_loop, and is used to set up the model and optimizer.
        '''
        GPU_FLAG = torch.cuda.is_available()
        GPU_USED = self.device.type == 'cuda'
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
        
        # if more than one device, use parallel training
        if torch.cuda.device_count() > 1 and self.multi_gpu:
            print("Using", torch.cuda.device_count(), "GPUs!") # this should be specified in requewstee
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)
        
        self.model.to(self.device) # move model to device

        self.optimizer.zero_grad() # set gradients to zero

    
    def fit_loop(self, epochs, train_loader, val_loader):
        
        self.nbatch = len(train_loader)
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
            for name, layer in self.model.named_modules():
                if 'NL' in name or 'reg' in name or 'loss' in name:
                    continue

                self.logged_parameters.append(name)
                self.hooks.append(layer.register_forward_hook(hook(name)))

        # main loop for training
        for epoch in range(epochs):
            self.epoch += 1
            # train one epoch
            out = self.train_one_epoch(train_loader, epoch)
            self.logger.add_scalar('Loss/Train (Epoch)', out['train_loss'], epoch)

            if self.log_activations:
                for name in self.logged_parameters:
                    try:
                        self.logger.add_histogram(
                                    f"Activations/{name.replace('.', ' ')}/hist",
                                    activations[name].view(-1),
                                    epoch,
                                )

                        self.logger.add_scalar(
                                    f"Activations/{name.replace('.', ' ')}/mean",
                                    activations[name].mean(),
                                    epoch,
                                )

                        self.logger.add_scalar(
                                    f"Activations/{name.replace('.', ' ')}/std",
                                    activations[name]
                                    .std(dim=0)
                                    .mean(),
                                    epoch,
                                )
                    except:
                        pass

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
            for batch_idx, data in enumerate(pbar):
                
                # Data to device if it's not already there
                for dsub in data:
                    if data[dsub].device != self.device:
                        data[dsub] = data[dsub].to(self.device)
                
                if isinstance(self.model, nn.DataParallel):
                    out = self.model.module.validation_step(data)
                else:
                    out = self.model.validation_step(data)

                runningloss += out['val_loss'].item()
                pbar.set_postfix({'val_loss': runningloss/(batch_idx+1)})

        return {'val_loss': runningloss/nsteps}
            
    def train_one_epoch(self, train_loader, epoch=0):
        # train for one epoch
        
        assert not self.use_closure, "Trainer: LBFGS and LBFGSNew are supported by LBFGSTrainer"
        
        self.model.train() # set model to training mode

        runningloss = 0

        pbar = tqdm(train_loader, total=self.nbatch, bar_format=None) # progress bar for looping over data
        pbar.set_description("Epoch %i" %epoch)

        for batch_idx, data in enumerate(pbar):
            
            # handle optimization step
            out = self.train_one_step(data, batch_idx=batch_idx)

            if self.use_gpu:
                torch.cuda.empty_cache()

            runningloss += out['train_loss']
            # update progress bar
            pbar.set_postfix({'train_loss': runningloss/(batch_idx + 1)})

        return {'train_loss': runningloss/self.nbatch} # should this be an aggregate out?


    def train_one_step(self, data, batch_idx=None):
        
        # Data to device if it's not already there
        if self.use_gpu:
            for dsub in data:
                if data[dsub].device != self.device:
                    data[dsub] = data[dsub].to(self.device)

        if isinstance(self.model, nn.DataParallel):
            out = self.model.module.training_step(data)
        else:
            out = self.model.training_step(data)
        
        self.n_iter += 1
        self.logger.add_scalar('Loss/Loss', out['loss'].item(), self.n_iter)
        self.logger.add_scalar('Loss/Train', out['train_loss'].item(), self.n_iter)
        self.logger.add_scalar('Loss/Reg', out['reg_loss'].item(), self.n_iter)

        loss = out['loss']
        # with torch.set_grad_enabled(True):
        loss.backward()
        
        # optimization step
        if ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (batch_idx + 1 == self.nbatch):
            self.optimizer.step()
            self.optimizer.zero_grad() # zero the gradients for the next batch
            
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

        if self.device.type == 'cuda':
            self.model.cpu()
            torch.cuda.empty_cache()

        self.model.eval()

        if self.log_activations:
            for hook in self.hooks:
                hook.remove()

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

class LBFGSTrainer(Trainer):

    def __init__(self,
            full_batch=False,
            **kwargs,
            ):

        super().__init__(**kwargs)

        self.fullbatch = full_batch
        
        
    def train_one_epoch(self, train_loader, epoch=0):
        # train for one epoch
        if self.fullbatch:
            self.accumulate_grad_batches = len(train_loader)
        
        self.model.train() # set model to training mode

        self.iter = 1
        max_iter = self.optimizer.param_groups[0]['max_iter']

        def closure():

            # pbar = tqdm(train_loader, total=self.nbatch, bar_format=None) # progress bar for looping over data
            # pbar.set_description("Epoch %i, iter %d/%d" %(epoch, self.iter, max_iter))
            
            self.optimizer.zero_grad()
            loss = torch.zeros(1, device=self.device)

            for batch_idx, data in enumerate(train_loader):
            
                if batch_idx < self.accumulate_grad_batches:

                    if self.use_gpu:
                        for dsub in data:
                            if data[dsub].device != self.device:
                                data[dsub] = data[dsub].to(self.device)

                    out = self.model.training_step(data)
                    loss += out['loss']
                    self.n_iter += 1
                    self.logger.add_scalar('Loss/Train', out['train_loss'].detach().item(), self.n_iter)
                    # update progress bar
                    # pbar.set_postfix({'train_loss': loss.detach().item()/(batch_idx + 1)})
                else:
                    break

            loss/=self.accumulate_grad_batches
            loss.backward()

            self.iter += 1

            return loss

        loss = self.optimizer.step(closure)
        print("Epoch %d (%d iter): loss = %02.3f" %(epoch, self.iter, loss.detach().item()))
        self.optimizer.zero_grad() # zero the gradients for the next batch
        
        if self.use_gpu:
                torch.cuda.empty_cache()

        return {'train_loss': loss} # should this be an aggregate out?

