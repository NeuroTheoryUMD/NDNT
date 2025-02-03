# from sklearn.utils import shuffle
import torch
from torch import nn
from functools import reduce

import numpy as np # TODO: we can get rid of this and just use torch for math
import tqdm
import json
import zipfile
import os

from NDNT.metrics import poisson_loss as plosses
from NDNT.metrics import mse_loss as glosses
from NDNT.metrics import rmse_loss as rlosses
from NDNT.utils import create_optimizer_params, NpEncoder, chunker
from NDNT.modules.experiment_sampler import ExperimentSampler

from NDNT import networks as NDNnetworks

FFnets = {
    'normal': NDNnetworks.FFnetwork,
    'add': NDNnetworks.FFnetwork,  # just controls how inputs are concatenated
    'mult': NDNnetworks.FFnetwork, # just controls how inputs are concatenated
    'scaffold': NDNnetworks.ScaffoldNetwork, # forward concatenates all layers (default: convolutional)
    'scaffold3d': NDNnetworks.ScaffoldNetwork3D, # forward concatenates all layers (default: convolutional)
    'readout': NDNnetworks.ReadoutNetwork
}

class NDN(nn.Module):
    """
    Initializes an instance of the NDN (Neural Deep Network) class.

    Args:
        ffnet_list (list): A list of FFnetwork objects. If None, a single ffnet specified by layer_list will be used.
        layer_list (list): A list specifying the layers of the FFnetwork. Only used if ffnet_list is None.
        external_modules (list): A list of external modules to be used in the FFnetwork.
        loss_type (str): The type of loss function to be used. Default is 'poisson'.
        ffnet_out (int or list): The index(es) of the output layer(s) of the FFnetwork(s). If None, the last layer will be used.
        optimizer_params (dict): Parameters for the optimizer. If None, default parameters will be used.
        model_name (str): The name of the model. If None, a default name will be assigned.
        seed (int): The random seed to be used for reproducibility.
        working_dir (str): The directory to save checkpoints and other files.

    Returns:
        None
    """
    
    def __init__(self,
        ffnet_list = None,
        layer_list = None,  # can just import layer list if consisting of single ffnetwork (simple ffnet)
        external_modules = None,
        loss_type = 'poisson', 
        ffnet_out = None,
        optimizer_params = None,
        model_name=None,
        seed=None,
        working_dir='./checkpoints'):

        super().__init__()

        # save the ffnet_list for serialization purposes
        from copy import deepcopy
        self.loss_type = loss_type
        self.ffnet_out = ffnet_out
        
        self.seed = seed
        if seed is not None:
            # set flags / seeds    
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        if ffnet_list is None:
            # then must be a single ffnet specified by layer_list
            assert layer_list is not None, "NDN: must specify either ffnet_list or layer_list."
            from .networks import FFnetwork
            ffnet_list = [FFnetwork.ffnet_dict(layer_list=layer_list)]
        else:
            assert layer_list is None, "NDN: cannot specify ffnet_list and layer_list at same time."
        assert type(ffnet_list) is list, 'FFnetwork list in NDN constructor must be a list.'
        self.ffnet_list = deepcopy(ffnet_list)
 
        if (ffnet_out is None) or (ffnet_out == -1):
            ffnet_out = len(ffnet_list)-1

        self.working_dir = working_dir

        # Configure loss
        if isinstance(loss_type, str):
            self.loss_type = loss_type
            if loss_type == 'poisson' or loss_type == 'poissonT':
                self.loss_module = plosses.PoissonLoss_datafilter()  # defined below, but could be in own Losses.py
            elif loss_type in ['simple', 'poissonS']:
                self.loss_module = plosses.SimplePoissonLoss()
            elif loss_type in ['gaussian', 'mse']:
                self.loss_module = glosses.MseLoss_datafilter()
            elif loss_type in ['rmse']:
                self.loss_module = rlosses.RmseLoss()
            else:
                print('Invalid loss function.')
                self.loss_module = None
        else: # assume passed in loss function directly
            self.loss_type = 'custom'
            self.loss_module = loss_type

        # module can also be called as a function (since it has a forward)
        self.loss = self.loss_module
        self.val_loss = self.loss_module
        self.trainer = None  # to stash last trainer used in fitting
        self.register_buffer( '_block_sample', torch.zeros(1, dtype=torch.int8) )
        self.block_sample = False  # if need time-contiguous blocks

        # Assemble FFnetworks from if passed in network-list -- else can embed model
        networks = self.assemble_ffnetworks(ffnet_list, external_modules)
        
        # Check and record output of network
        if not isinstance(ffnet_out, list):
            ffnet_out = [ffnet_out]
        
        for ii in range(len(ffnet_out)):
            if ffnet_out[ii] == -1:
                ffnet_out[ii] = len(networks)-1
                
        # Assemble ffnetworks
        self.networks = networks
        self.ffnet_out = ffnet_out

        # Set model_name if unspecified
        if model_name is None:
            self.model_name = self.model_string()
        else:
            self.model_name = model_name

        # Assign optimizer params
        if optimizer_params is None:
            optimizer_params = create_optimizer_params()
        self.opt_params = optimizer_params
        self.speckled_flag = False
    # END NDN.__init__

    @property
    def block_sample(self):
        if self._block_sample > 0:
            return True
        else:
            return False

    @block_sample.setter
    def block_sample(self, value):
        assert isinstance(value, bool), "NDN.block_sample: boolean value required"
        if value:
            self._block_sample.data[:] = 1
        else:
            self._block_sample.data[:] = 0
    # END NDN.block_sample property

    def assemble_ffnetworks(self, ffnet_list, external_nets=None):
        """
        This function takes a list of ffnetworks and puts them together 
        in order. This has to do two steps for each ffnetwork: 

        1. Plug in the inputs to each ffnetwork as specified
        2. Builds the ff-network with the input
        
        This returns the 'network', which is (currently) a module with a 
        'forward' and 'reg_loss' function specified.

        When multiple ffnet inputs are concatenated, it will always happen in the first
        (filter) dimension, so all other dimensions must match

        Args:
            ffnet_list (list): A list of ffnetworks to be assembled.
            external_nets (optional): External networks to be passed into the 'external' type ffnetworks.

        Returns:
            networks (nn.ModuleList): A module list containing the assembled ff-networks.

        Raises:
            AssertionError: If ffnet_list is not a list.
        """

        assert type(ffnet_list) is list, "ffnet_list must be a list."
        
        num_networks = len(ffnet_list)
        # Make list of pytorch modules
        networks = nn.ModuleList()

        for mm in range(num_networks):

            # Determine internal network input to each subsequent network (if exists)
            if ffnet_list[mm]['xstim_n'] is None:
                input_dims_list = []
            else:
                input_dims_list = [ffnet_list[mm]['layer_list'][0]['input_dims']]

            if ffnet_list[mm]['ffnet_n'] is not None:
                nets_in = ffnet_list[mm]['ffnet_n']
                for ii in range(len(nets_in)):
                    assert nets_in[ii] < mm, "FFnet%d (%d): input networks must come earlier"%(mm, ii)
                    input_dims_list.append(networks[nets_in[ii]].output_dims)
                ffnet_list[mm]['input_dims_list'] = input_dims_list
        
            # Create corresponding FFnetwork
            net_type = ffnet_list[mm]['ffnet_type']
            if net_type == 'external':  # separate case because needs to pass in external modules directly
                networks.append( NDNnetworks.FFnet_external(ffnet_list[mm], external_nets))
            else:
                networks.append( FFnets[net_type](**ffnet_list[mm]) )

        return networks
    # END NDNT.assemble_ffnetworks

    def compute_network_outputs(self, Xs):
        """
        Computes the network outputs for the given input data.

        Args:
            Xs (list): The input data.

        Returns:
            tuple: A tuple containing the network inputs and network outputs.

        Raises:
            AssertionError: If no networks are defined in this NDN.

        Note:
            This method currently saves only the network outputs and does not return the network inputs.
        """

        assert 'networks' in dir(self), "compute_network_outputs: No networks defined in this NDN"

        net_ins, net_outs = [], []

        for ii in range(len(self.networks)):
            # First, assemble inputs to network
            if self.networks[ii].xstim_n is not None:
                # then getting external input
                inputs = [Xs[self.networks[ii].xstim_n]]
            else:
                inputs = []
            if self.networks[ii].ffnets_in is not None:
                in_nets = self.networks[ii].ffnets_in
                # Assemble network inputs in list, which will be used by FFnetwork
                for mm in range(len(in_nets)):
                    inputs.append( net_outs[in_nets[mm]] )
            net_ins.append(inputs)
            # Compute outputs
            net_outs.append( self.networks[ii](inputs) ) 
        return net_ins, net_outs
    # END NDNT.compute_network_outputs

    def forward(self, Xs):
            """
            Applies the forward pass of each network in sequential order.

            Args:
                Xs (list): List of input data.

            Returns:
                ndarray: Output of the forward pass.

            Notes:
                The tricky thing is concatenating multiple-input dimensions together correctly.
                Note that the external inputs are actually in principle a list of inputs.
            """

            net_ins, net_outs = self.compute_network_outputs( Xs )
            # For now assume it's just one output, given by the first value of self.ffnet_out
            return net_outs[self.ffnet_out[0]]
    # END NDNT.forward

    def training_step(self, batch, batch_idx=None):  # batch_indx not used, right?
        """
        Performs a single training step.

        Args:
            batch (dict): A dictionary containing the input data batch.
            batch_idx (int, optional): The index of the current batch. Defaults to None.

        Returns:
            dict: A dictionary containing the loss values for training, total loss, and regularization loss.
        """

        y = batch['robs']

        if self.speckled_flag:
            dfs = batch['dfs'] * batch['Mtrn']
        else:
            dfs = batch['dfs']

        y_hat = self(batch)

        loss = self.loss(y_hat, y, dfs)

        regularizers = self.compute_reg_loss()

        return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}
    # END Encoder.training_step

    def validation_step(self, batch, batch_idx=None):
        """
        Performs a validation step for the model.

        Args:
            batch (dict): A dictionary containing the input batch data.
            batch_idx (int, optional): The index of the current batch. Defaults to None.

        Returns:
            dict: A dictionary containing the computed losses for the validation step.
        """
        
        y = batch['robs']

        if self.speckled_flag:
            dfs = batch['dfs'] * batch['Mval']
        else:
            dfs = batch['dfs']

        y_hat = self(batch)
        loss = self.val_loss(y_hat, y, dfs)
        
        reg_loss = self.compute_reg_loss()
        
        return {'loss': loss, 'val_loss': loss, 'reg_loss': reg_loss}
    
    def compute_reg_loss(self):
            """
            Computes the regularization loss for the NDNT model.

            Args:
                None

            Returns:
                The total regularization loss.
            """

            rloss = []
            for network in self.networks:
                rloss.append(network.compute_reg_loss())
            return reduce(torch.add, rloss)
    
    def get_trainer(self,
        version=None,
        save_dir='./checkpoints',
        name='jnkname',
        optimizer=None,
        scheduler=None,
        device=None,
        optimizer_type='AdamW',
        early_stopping=False,
        early_stopping_patience=5,
        early_stopping_delta=0.0,
        optimize_graph=False,
        **kwargs):
        """
        Returns a trainer object.

        Args:
            version (str): The version of the trainer.
            save_dir (str): The directory to save the trainer checkpoints.
            name (str): The name of the trainer.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to use for adjusting the learning rate during training.
            device (str or torch.device): The device to use for training. If not specified, it will default to 'cuda:0' if available, otherwise 'cpu'.
            optimizer_type (str): The type of optimizer to use. Default is 'AdamW'.
            early_stopping (bool): Whether to use early stopping during training. Default is False.
            early_stopping_patience (int): The number of epochs to wait for improvement before stopping early. Default is 5.
            early_stopping_delta (float): The minimum change in the monitored metric to be considered as improvement. Default is 0.0.
            optimize_graph (bool): Whether to optimize the computation graph during training. Default is False.
            **kwargs: Additional keyword arguments to be passed to the trainer.

        Returns:
            trainer (Trainer): The trainer object.
        """

        from NDNT.training import Trainer, EarlyStopping, LBFGSTrainer
        import os

        #trainers = {'step': Trainer, 'AdamW': Trainer, 'Adam': Trainer, 'sgd': Trainer, 'lbfgs': LBFGSTrainer}
        trainers = {'step': Trainer, 'lbfgs': LBFGSTrainer}

        if optimizer is None:  # then specified through optimizer inputs
            optimizer = self.get_optimizer(optimizer_type=optimizer_type, **kwargs)

        if early_stopping:
            earlystopper = EarlyStopping( patience=early_stopping_patience, delta= early_stopping_delta )
            #if isinstance(opt_params['early_stopping'], EarlyStopping):
            #    earlystopping = opt_params['early_stopping']
            #elif isinstance(opt_params['early_stopping'], dict):
            #    earlystopping = EarlyStopping(patience=opt_params['early_stopping']['patience'],delta=opt_params['early_stopping']['delta'])
            #else:
            #    earlystopping = EarlyStopping(patience=opt_params['early_stopping_patience'],delta=0.0)
        else:
            earlystopper = None

        # Check for device assignment in opt_params
        if device is None:
            # Assign default device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            if isinstance(device, str):
                device = torch.device(device)
            elif not isinstance(device, torch.device):
                raise ValueError("opt_params['device'] must be a string or torch.device")

        trainer_type = 'step'
        if optimizer is not None:
            if isinstance(optimizer, torch.optim.LBFGS):
                trainer_type = 'lbfgs'
        else:
            if optimizer_type == 'LBFGS':
                trainer_type = 'lbfgs'

        trainer = trainers[trainer_type](model=self,
                                         optimizer=optimizer,
                                         early_stopping=earlystopper,
                                         dirpath=os.path.join(save_dir, name),
                                         optimize_graph=optimize_graph,
                                         device=device,
                                         scheduler=scheduler,
                                         version=version,
                                         **kwargs
                                         )

        return trainer

    def get_dataloaders(self,
            dataset,
            train_inds=None,
            val_inds=None,
            batch_size=10, 
            num_workers=0,
            #is_fixation=False,
            is_multiexp=False,
            full_batch=False,
            pin_memory=False,
            data_seed=None,
            #device=None,
            **kwargs):
        
        """
        Creates dataloaders for training and validation.

        Args:
            dataset (Dataset): The dataset to use for training and validation.
            train_inds (list): The indices of the training data.
            val_inds (list): The indices of the validation data.
            batch_size (int): The batch size to use.
            num_workers (int): The number of workers to use for data loading.
            is_multiexp (bool): Whether the dataset is a multi-experiment dataset.
            full_batch (bool): Whether to use the full batch for training.
            pin_memory (bool): Whether to pin memory for faster data loading.
            data_seed (int): The seed to use for the data loader.
            **kwargs: Additional keyword arguments.

        Returns:
            train_dl (DataLoader): The training data loader.
            valid_dl (DataLoader): The validation data loader.
        """

        from torch.utils.data import DataLoader, random_split, Subset

        # get the verbose flag if it is provided, default to False if not
        verbose = kwargs.get('verbose', False)

        covariates = list(dataset[0].keys())
        #print('Dataset covariates:', covariates)

        if train_inds is None or val_inds is None:
            n_val = len(dataset)//5
            n_train = len(dataset)-n_val

            if data_seed is None:
                train_ds, val_ds = random_split(dataset, lengths=[n_train, n_val], generator=torch.Generator()) # .manual_seed(42)
            else:
                train_ds, val_ds = random_split(dataset, lengths=[n_train, n_val], generator=torch.Generator().manual_seed(data_seed))
        
            if train_inds is None:
                train_inds = train_ds.indices
            if val_inds is None:
                val_inds = val_ds.indices
            
        # build dataloaders:
        if self.block_sample:
            # New method: just change collate function to collate_blocks provided by dataset
            #train_ds = Subset(dataset, train_inds)
            #val_ds = Subset(dataset, val_inds)

            #train_dl = DataLoader(
            #    train_ds, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_blocks,
            #    num_workers=num_workers, pin_memory=pin_memory)
            #valid_dl = DataLoader(
            #    val_ds, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_blocks,
            #    num_workers=num_workers, pin_memory=pin_memory)

            # we use a batch sampler to sample the data because it generates indices for the whole batch at one time
            # instead of iterating over each sample. This is both faster (probably) for our cases, and it allows us
            # to use the "Fixation" datasets and concatenate along a variable-length batch dimension
            train_sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SubsetRandomSampler(train_inds),
                batch_size=batch_size,
                drop_last=False)
            
            val_sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SubsetRandomSampler(val_inds),
                batch_size=batch_size,
                drop_last=False)

            train_dl = DataLoader(dataset, sampler=train_sampler, batch_size=None, num_workers=num_workers)
            valid_dl = DataLoader(dataset, sampler=val_sampler, batch_size=None, num_workers=num_workers)
        elif is_multiexp:
            train_sampler = ExperimentSampler(dataset, batch_size=batch_size, indices=train_inds, shuffle=True, verbose=verbose)
            val_sampler = ExperimentSampler(dataset, batch_size=batch_size, indices=val_inds, shuffle=True, verbose=verbose)

            train_dl = DataLoader(dataset, sampler=train_sampler, batch_size=None, num_workers=num_workers)
            valid_dl = DataLoader(dataset, sampler=val_sampler, batch_size=None, num_workers=num_workers)
        else:
            train_ds = Subset(dataset, train_inds)
            val_ds = Subset(dataset, val_inds)

            shuffle_data = True  # no reason not to shuffle -- samples everything over epoch
            if pin_memory:
                print('Pinning memory')
            train_dl = DataLoader(
                train_ds, batch_size=batch_size, shuffle=shuffle_data,
                num_workers=num_workers, pin_memory=pin_memory)
            valid_dl = DataLoader(
                val_ds, batch_size=batch_size, shuffle=shuffle_data,
                num_workers=num_workers, pin_memory=pin_memory)
            
        return train_dl, valid_dl
    # END NDN.get_dataloaders

    def get_optimizer(self,
            optimizer_type='AdamW',
            learning_rate=0.001,
            weight_decay=0.01,
            amsgrad=False,
            betas=(0.9, 0.999),
            momentum=0.9,  # for SGD
            max_iter=10,
            history_size=4,
            tolerance_change=1e-3,
            tolerance_grad=1e-4,
            line_search_fn=None,
            **kwargs):
        
        """
        Returns an optimizer object.

        Args:
            optimizer_type (str): The type of optimizer to use. Default is 'AdamW'.
            learning_rate (float): The learning rate to use. Default is 0.001.
            weight_decay (float): The weight decay to use. Default is 0.01.
            amsgrad (bool): Whether to use the AMSGrad variant of Adam. Default is False.
            betas (tuple): The beta values to use for the optimizer. Default is (0.9, 0.999).
            max_iter (int): The maximum number of iterations for the LBFGS optimizer. Default is 10.
            history_size (int): The history size to use for the LBFGS optimizer. Default is 4.

        Returns:
            optimizer (torch.optim.Optimizer): The optimizer object.
        """
        
        # Assign optimizer
        if optimizer_type=='AdamW':

            # weight decay only affects certain parameters
            decay = []
            
            decay_names = []
            no_decay_names = []
            no_decay = []
            for name, m in self.named_parameters():
                if 'weight' in name:
                    decay.append(m)
                    decay_names.append(name)
                else:
                    no_decay.append(m)
                    no_decay_names.append(name)

            optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0}, {'params': decay, 'weight_decay': weight_decay}],
                    lr=learning_rate,
                    betas=betas,
                    amsgrad=amsgrad)

        elif optimizer_type=='Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                    lr=learning_rate,
                    betas=betas)

        elif (optimizer_type=='SGD') | (optimizer_type=='sgd'):
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=learning_rate, 
                momentum=momentum, 
                weight_decay=weight_decay)
            #torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, 
            #  weight_decay=0, nesterov=False, *, maximize=False, foreach=None, 
            #  differentiable=False, fused=None)
            
        elif optimizer_type=='LBFGS':

            optimizer = torch.optim.LBFGS(
                self.parameters(), 
                history_size=history_size, max_iter=max_iter, 
                tolerance_change=tolerance_change,
                tolerance_grad=tolerance_grad,
                line_search_fn=line_search_fn)

        else:
            raise ValueError('optimizer [%s] not supported' %optimizer_type)
        
        return optimizer
    # END NDN.get_optimizer
    
    def prepare_regularization(self):
        """
        Prepares the regularization for the model.

        Returns:
            None
        """

        for network in self.networks:
            network.prepare_regularization(device=self.device)

    def fit(self,
        dataset,
        train_inds=None,
        val_inds=None,
        speckledXV=False,
        seed=None, # this currently seed for data AND optimization
        save_dir:str=None,
        version:int=None,
        #name:str=None,
        optimizer=None,  # this can pass in full optimizer (rather than keyword)
        scheduler=None,
        batch_size=None,
        force_dict_training=False,  # will force dict-based training instead of using data-loaders for LBFGS
        block_sample=None, # to pass in flag to use dataset's block-sampler (and choose appropriate dataloader)
        reuse_trainer=False,
        device=None,
        **kwargs  # kwargs replaces explicit opt_params, which can list some or all of the following
        ):

        """
        Trains the model using the specified dataset.

        Args:
            dataset (Dataset): The dataset to use for training and validation.
            train_inds (list): The indices of the training data.
            val_inds (list): The indices of the validation data.
            speckledXV (bool): Whether to use speckled cross-validation. Default is False.
            seed (int): The seed to use for reproducibility.
            save_dir (str): The directory to save the model checkpoints.
            version (int): The version of the trainer.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to use for adjusting the learning rate during training.
            batch_size (int): The batch size to use.
            force_dict_training (bool): Whether to force dictionary-based training. Default is False.
            block_sample (bool): Whether to use block sampling. Default is None.
            reuse_trainer (bool): Whether to reuse the trainer. Default is False.
            device (str or torch.device): The device to use for training. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        Raises:
            AssertionError: If batch_size is not specified.

        Notes:
            The fit method is the main training loop for the model.
            Steps:
            1. Get a trainer and dataloaders
            2. Prepare regularizers
            3. Run the main fit loop from the trainer, checkpoint, and save model
        """

        import time

        assert batch_size is not None, "NDN.fit() must be passed batch_size in optimization params."

        if save_dir is None:
            save_dir = self.working_dir

        if block_sample is not None:
            self.block_sample = block_sample
        if self.block_sample:
            assert dataset.trial_sample, "dataset trial_sample does not match model_block_sample"
        else:
            assert dataset.trial_sample == False, "dataset trial_sample does not match model_block_sample"

        # Should be set ahead of time
        name = self.model_name

        # Determine train_inds and val_inds: read from dataset if not specified, or warn
        self.speckled_flag = speckledXV  # This determines training/test loop
        if speckledXV:
            assert train_inds is None, "SPECKLED: no train_inds"
            assert val_inds is None, "SPECKLED: no val_inds"
            train_inds = range(len(dataset))
            val_inds = range(len(dataset))
        else:
            if train_inds is None:
                # check dataset itself
                if self.block_sample:
                    if hasattr(dataset, 'train_blks'):
                        if dataset.train_blks is not None:
                            train_inds = dataset.train_blks
                        else:
                            print( "Warning: no train_bkls specified in dataset (block_sample)")
                    else:
                        print( "Warning: no train_bkls specified (block_sample)")                
                else:
                    if hasattr(dataset, 'train_inds'):
                        if dataset.train_inds is not None:
                            train_inds = dataset.train_inds
                        else:
                            print( "Warning: no train_inds specified in dataset" )
                    else:
                        train_inds = range(len(dataset))
                        if 'verbose' in kwargs.keys():
                            if kwargs['verbose'] > 0:
                                print( "Warning: no train_inds specified. Using full dataset passed in.")
            if val_inds is None:
                if self.block_sample:
                    if hasattr(dataset, 'val_blks'):
                        if dataset.val_blks is not None: # changed from if dataset.val_inds is not None:
                            val_inds = dataset.val_blks
                        else:
                            print( "Warning: no val_bkls specified in dataset (block_sample)")
                    else:
                        print( "Warning: no val_bkls specified (block_sample)")
                else:
                    if hasattr(dataset, 'val_inds'):
                        if dataset.val_inds is not None:
                            val_inds = dataset.val_inds
                        else:
                            print( "Warning: no val_inds specified in dataset." )
                    else:
                        print( "No val_inds specified.")

        if force_dict_training:
            batch_size = len(train_inds)

        # Check to see if loss-flags require any initialization using dataset information 
        if self.loss_module.unit_weighting or (self.loss_module.batch_weighting == 2):
            self.initialize_loss(dataset, batch_size=batch_size, data_inds=train_inds) 

        # Prepare model regularization (will build reg_modules)
        self.prepare_regularization()

        if 'verbose' in kwargs:
            if kwargs['verbose'] > 0:
                print( 'Model:', self.model_name)

        # Make trainer 
        if reuse_trainer & (self.trainer is not None):
            trainer = self.trainer
            print("  Reusing existing trainer")
        else:
            trainer = self.get_trainer(
                version=version,
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=save_dir,
                name=name,
                device=device,
                **kwargs)

        t0 = time.time()
        if force_dict_training:
            from NDNT.training import LBFGSTrainer
            assert isinstance(trainer, LBFGSTrainer), "force_dict_training will not work unless using LBFGS." 

            #trainer.fit(self, train_dl.dataset[:], valid_dl.dataset[:], seed=seed)
            trainer.fit(self, dataset[train_inds], dataset[val_inds], seed=seed)
        else:
            # Create dataloaders / 
            train_dl, valid_dl = self.get_dataloaders(
                dataset, batch_size=batch_size, 
                train_inds=train_inds, val_inds=val_inds, data_seed=seed, **kwargs)
    
            trainer.fit(self, train_dl, valid_dl, seed=seed)
        t1 = time.time()

        self.trainer = trainer

        if 'verbose' in kwargs.keys():
            if kwargs['verbose'] > 0:
                print('  Fit complete:', t1-t0, 'sec elapsed')
        else:  # default behavior
            print('  Fit complete:', t1-t0, 'sec elapsed')
    # END NDN.fit

    def fit_dl(self,
        train_ds, val_ds, 
        seed=None, # this currently seed for data AND optimization
        save_dir:str=None,
        version:int=None,
        name:str=None,
        optimizer=None,  # this can pass in full optimizer (rather than keyword)
        scheduler=None,
        batch_size=None,
        num_workers=0,
        force_dict_training=False,  # will force dict-based training instead of using data-loaders for LBFGS
        device=None,
        **kwargs  # kwargs replaces explicit opt_params, which can list some or all of the following
        ):
        """
        Fits the model using deep learning training.

        Args:
            train_ds (Dataset): The training dataset.
            val_ds (Dataset): The validation dataset.
            seed (int, optional): The seed for data and optimization. Defaults to None.
            save_dir (str, optional): The directory to save the model. Defaults to None.
            version (int, optional): The version of the model. Defaults to None.
            name (str, optional): The name of the model. Defaults to None.
            optimizer (Optimizer, optional): The optimizer for training. Defaults to None.
            scheduler (Scheduler, optional): The scheduler for training. Defaults to None.
            batch_size (int, optional): The batch size for training. Defaults to None.
            num_workers (int, optional): The number of workers for data loading. Defaults to 0.
            force_dict_training (bool, optional): Whether to force dictionary-based training instead of using data loaders for LBFGS. Defaults to False.
            device (str, optional): The device to use for training. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        Raises:
            AssertionError: If batch_size is not provided.

        Notes:
            This is the main training loop.
            Steps:
            1. Get a trainer and data loaders.
            2. Prepare regularizers.
            3. Run the main fit loop from the trainer, checkpoint, and save the model.
        """
    
        import time

        assert batch_size is not None, "NDN.fit() must be passed batch_size in optimization params."

        if save_dir is None:
            save_dir = self.working_dir
        
        if name is None:
            name = self.model_name

        # Check to see if loss-flags require any initialization using dataset information 
        if self.loss_module.unit_weighting or (self.loss_module.batch_weighting == 2):
            if force_dict_training:
                batch_size = len(train_ds)
            self.initialize_loss(train_ds, batch_size=batch_size) 

        # Prepare model regularization (will build reg_modules)
        self.prepare_regularization()

        # Create dataloaders 
        shuffle_data=True  # no condition when we don't want it shuffled, right?

        from torch.utils.data import DataLoader
        
        if seed is not None:
            torch.manual_seed(seed)
        train_dl = DataLoader( train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_data) 
        valid_dl = DataLoader( val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_data) 

        # Make trainer
        trainer = self.get_trainer(
            version=version,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=save_dir,
            name=name,
            device=device,
            **kwargs)

        t0 = time.time()
        if force_dict_training:
            from NDNT.training import LBFGSTrainer
            assert isinstance(trainer, LBFGSTrainer), "force_dict_training will not work unless using LBFGS." 

            trainer.fit(self, train_dl.dataset[:], valid_dl.dataset[:], seed=seed)
        else:
            trainer.fit(self, train_dl, valid_dl, seed=seed)
        t1 = time.time()

        if 'verbose' in kwargs.keys():
            if kwargs['verbose'] > 0:
                print('  Fit complete:', t1-t0, 'sec elapsed')
        else:  # default behavior
            print('  Fit complete:', t1-t0, 'sec elapsed')
        self.trainer = trainer
    # END NDN.fit_dl
    
    def initialize_loss( self, dataset=None, batch_size=None, data_inds=None, batch_weighting=None, unit_weighting=None ):
        """
        Interacts with loss_module to set loss flags and/or pass in dataset information.
        
        Args:
            dataset (optional): The dataset to be used for computing average batch size and unit weights.
            batch_size (optional): The batch size to be used for computing average batch size.
            data_inds (optional): The indices of the data in the dataset to be used.
            batch_weighting (optional): The batch weighting value to be set. Must be one of [-1, 0, 1, 2].
            unit_weighting (optional): The unit weighting value to be set.
        
        Returns:
            None

        Notes:
            Without a dataset, this method will only set the flags 'batch_weighting' and 'unit_weighting'.
            With a dataset, this method will set 'av_batch_size' and 'unit_weights' based on the average rate.
        """

        if batch_weighting is None:
            batch_weighting = self.loss_module.batch_weighting
        else:
            assert batch_weighting in [0,1,2,-1], "Initialize loss: invalid batch_weighting value: %d"%batch_weighting
        if unit_weighting is None:
            unit_weighting = self.loss_module.unit_weighting
        
        unit_weights, av_batch_size = None, None
        if dataset is not None:
            assert batch_size is not None, "Initialize loss: need to have batch_size defined with dataset"
            if data_inds is None:
                data_inds = range(len(dataset))

            # Compute average batch size
            T = len(data_inds)
            num_batches = np.ceil(T/batch_size)
            av_batch_size = T/num_batches

            # Compute unit weights based on number of spikes in dataset (and/or firing rate) if requested
            unit_weights = 1.0/np.maximum( self.compute_average_responses(dataset, data_inds=data_inds), 1e-8 )

        self.loss_module.set_loss_weighting( 
            batch_weighting=batch_weighting, unit_weighting=unit_weighting, 
            unit_weights=unit_weights, av_batch_size=av_batch_size )
    # END NDNT.initialize_loss

    def compute_average_responses( self, dataset, data_inds=None ):
        """
        Computes the average responses for the specified dataset.

        Args:
            dataset: The dataset to use for computing the average responses.
            data_inds: The indices of the data in the dataset.

        Returns:
            ndarray: The average responses.
        """

        if data_inds is None:
            data_inds = range(len(dataset))
        
        # Check if dataset has specified function
        if hasattr( dataset, 'avrates' ):
            return dataset.avrates()

        # Iterate through dataset to compute average rates
        NC = dataset[0]['robs'].shape[-1]
        Rsum, Tsum = torch.zeros(NC), torch.zeros(NC)
        for tt in data_inds:
            sample = dataset[tt]
            if len(sample['robs'].shape) == 1:  # then one datapoint at a time
                Tsum += sample['dfs'].cpu()
                Rsum += torch.mul(sample['dfs'], sample['robs']).cpu()
            else: # then each sample contains multiple datapoints and need to sum
                Tsum += torch.sum(sample['dfs'], axis=0).cpu()
                Rsum += torch.sum(torch.mul(sample['dfs'], sample['robs']), axis=0).cpu()

        return torch.divide( Rsum, Tsum.clamp(1) ).cpu().detach().numpy()
    # END NDNT.compute_average_responses

    def initialize_biases( self, dataset, data_inds=None, ffnet_target=-1, layer_target=-1 ):
        """
        Initializes the biases for the specified dataset.

        Args:
            dataset: The dataset to use for initializing the biases.
            data_inds: The indices of the data in the dataset.
            ffnet_target: The target feedforward network.
            layer_target: The target layer.

        Returns:
            None

        Notes:
            This method initializes the biases for the specified feedforward network and layer using the average responses.
        """

        avRs = self.compute_average_responses( dataset=dataset, data_inds=data_inds )
        # Invert last layer nonlinearity
        assert len(avRs) == len(self.networks[ffnet_target].layers[layer_target].bias), "Need to specify output NL correctly: layer confusion."
        if self.networks[ffnet_target].layers[layer_target].NL is None:
            print( 'Initializing biases given linear NL')
            biases = avRs
            
        elif isinstance(self.networks[ffnet_target].layers[layer_target].NL, nn.Softplus):
            print( 'Initializing biases given Softplus NL')
            # Invert softplus
            beta = self.networks[ffnet_target].layers[layer_target].NL.beta
            biases = np.log(np.exp(np.maximum(avRs, 1e-6))-1)/beta
        else:
            biases = np.zeros(len(avRs))
        
        self.networks[ffnet_target].layers[layer_target].bias.data = torch.tensor(biases, dtype=torch.float32)
    # otherwise not initializing biases, even if desired

    def eval_models(
        self, data, data_inds=None, bits=False, null_adjusted=False, speckledXV=False, train_val=1,
        batch_size=1000, num_workers=0, device=None, **kwargs ):
        '''
        Evaluate the neural network models on the given data.

        get null-adjusted log likelihood (if null_adjusted = True)
        bits=True will return in units of bits/spike

        Note that data will be assumed to be a dataset, and data_inds will have to be specified batches
        from dataset.__get_item__()

        Args:
            data (dict or Dataset): The input data to evaluate the models on. If a dictionary is provided, it is assumed to be a single sample. If a Dataset object is provided, it is assumed to contain multiple samples.
            data_inds (list, optional): The indices of the data samples to evaluate. Only applicable if `data` is a Dataset object. Defaults to None.
            bits (bool, optional): If True, the result will be returned in units of bits/spike. Defaults to False.
            null_adjusted (bool, optional): If True, the null-adjusted log likelihood will be calculated. Defaults to False.
            speckledXV (bool, optional): If True, speckled cross-validation will be used. Defaults to False.
            train_val (int, optional): The train/validation split ratio. Only applicable if `speckledXV` is True. Defaults to 1.
            batch_size (int, optional): The batch size for data loading. Defaults to 1000.
            num_workers (int, optional): The number of worker processes for data loading. Defaults to 0.
            device (torch.device, optional): The device to perform the evaluation on. If None, the device of the model will be used. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: The evaluated log likelihood values for each neuron.

        Note:
            If `data` is a dictionary, it is assumed to contain the following keys:
                'robs': The observed responses.
                'Mval': The validation mask.
                'dfs': The data filters.
            If `data` is a Dataset object, it is assumed to have the following attributes:
                'robs': The observed responses.
                'dfs': The data filters.

        Raises:
            AssertionError: If `data_inds` is not None when `data` is a dictionary.
        '''

        # Switch into evaluation mode
        self.eval()

        if isinstance(data, dict): 
            # Then assume that this is just to evaluate a sample: keep original here
            assert data_inds is None, "Cannot use data_inds if passing in a dataset sample."
            dev0 = data['robs'].device
            if device is not None:
                if device != dev0:
                    "For dictionary-based evaluation, constrained to the device the data is on."

            d0 = next(self.parameters()).device  # device the model is currently on
            m0 = self.to(dev0)
            yhat = m0(data)
            y = data['robs']
            if speckledXV:
                if train_val == 1:
                    dfs = data['Mval']*data['dfs']
                else:
                    dfs = data['Mtrn']*data['dfs']
            else:
                dfs = data['dfs']
            
            if 'poisson' in m0.loss_type:
                loss = m0.loss_module.lossNR
            else:
                print("This loss-type is not supported for eval_models.")
                loss = None

            with torch.no_grad():
                LLraw = torch.sum( 
                    torch.multiply( 
                        dfs, 
                        loss(yhat, y)),
                        axis=0).detach().cpu().numpy()
                obscnt = torch.sum(
                    torch.multiply(dfs, y), axis=0).detach().cpu().numpy()
                
                Ts = np.maximum(torch.sum(dfs, axis=0).detach().cpu().numpy(), 1)

                LLneuron = LLraw / np.maximum(obscnt,1) # note making positive

                if null_adjusted:
                    #predcnt = torch.sum(
                    #    torch.multiply(dfs, yhat), axis=0).detach().cpu().numpy()
                    #rbar = np.divide(predcnt, Ts)
                    #LLnulls = np.log(rbar)-np.divide(predcnt, np.maximum(obscnt,1))
                    rbar = np.divide(obscnt, np.maximum(Ts, 1))
                    LLnulls = np.log(np.maximum(rbar, 1e-6))-1

                    LLneuron = -LLneuron - LLnulls             
            self = self.to(d0)
            return LLneuron  # end of the old method

        else:
            # This will be the 'modern' eval_models using already-defined self.loss_module
            # In this case, assume data is dataset
            if data_inds is None:
                data_inds = list(range(len(data)))

            from torch.utils.data import DataLoader, Subset

            data_dl, _ = self.get_dataloaders(
                data, batch_size=batch_size, num_workers=num_workers, 
                train_inds=data_inds, val_inds=data_inds)
            #data_ds = Subset(data, data_inds)
            #data_dl = DataLoader(data_ds, batch_size=batch_size, num_workers=num_workers)

            LLsum, Tsum, Rsum = 0, 0, 0
            from tqdm import tqdm
            d0 = next(self.parameters()).device  # device the model is currently on
            self = self.to(device)

            for data_sample in tqdm(data_dl, desc='Eval models'):
                # data_sample = data[tt]
                for dsub in data_sample.keys():
                    if data_sample[dsub].device != device:
                        data_sample[dsub] = data_sample[dsub].to(device)
                with torch.no_grad():
                    pred = self(data_sample)
                    if speckledXV:
                        if train_val == 1:
                            dfs = data_sample['dfs']*data_sample['Mval']
                        else:
                            dfs = data_sample['dfs']*data_sample['Mtrn']
                    else:
                        dfs = data_sample['dfs']
                    
                    LLsum += self.loss_module.unit_loss( 
                        pred, data_sample['robs'], data_filters=dfs, temporal_normalize=False)
                    Tsum += torch.sum(data_sample['dfs'], axis=0)
                    Rsum += torch.sum(torch.mul(dfs, data_sample['robs']), axis=0)
            LLneuron = torch.divide(LLsum, Rsum.clamp(1) )
            self = self.to(d0)  # move back to original device

            # Null-adjust
            if null_adjusted:
                rbar = torch.divide(Rsum, Tsum.clamp(1))
                LLnulls = torch.log(rbar)-1
                LLneuron = -LLneuron - LLnulls 
        if bits:
            LLneuron/=np.log(2)

        self = self.to(d0)  # put back on original device

        return LLneuron.detach().cpu().numpy()

    def predictions(self, data, data_inds=None, block_inds=None, batch_size=None, num_lags=0, ffnet_target=None, device=None):
        """
        Generate predictions for the model for a dataset. Note that will need to send to device if needed, and enter 
        batch_size information. 

        Args:
            data: The dataset.
            data_inds: The indices to use from the dataset.
            block_inds: The blocks to use from the dataset (if block_sample).
            batch_size: The batch size to use for processing. Reduce if running out of memory.
            num_lags: The number of lags to use for prediction.
            ffnet_target: The index of the feedforward network to use for prediction. Defaults to None.
            device: The device to use for prediction

        Returns:
            torch.Tensor: The predictions array (detached, on cpu)
        """
        
        self.eval()    
        num_cells = self.networks[-1].output_dims[0]
        model_device = self.device
        if self.block_sample:
            if device is None:
                device = data['robs'].device
            self = self.to(device)

            assert data_inds is None, "block_sample currently does not handle data_inds"
            if batch_size is None:
                batch_size = 10   # default batch size for block_sample
            if block_inds is None:
                block_inds = np.arange(len(data.block_inds))
            
            total = len(block_inds)
            data_subset_NT = np.sum([len(data.block_inds[ii]) for ii in block_inds])
            pred = torch.zeros((data_subset_NT, num_cells)).to(device)
            with torch.no_grad():
                #for i in tqdm.tqdm(range(0, total, batch_size)):
                for trs in tqdm.tqdm(chunker(np.arange(total), batch_size)):
                    #batch_block_inds = [data.block_inds[ii] for ii in block_inds[i:i+batch_size]]
                    batch_block_inds = [data.block_inds[ii] for ii in trs]
                    batch_block_inds = np.concatenate(batch_block_inds)
                    batch_block_inds = batch_block_inds[batch_block_inds < data.NT]
                    #data_batch = data[i:np.minimum(i+batch_size, total)]
                    data_batch = data[block_inds[trs]]
                    if device is not None:
                        for key in data_batch.keys():
                            data_batch[key] = data_batch[key].to(device)
                    if ffnet_target is not None:
                        pred_batch = self.networks[ffnet_target](data_batch)
                    else:
                        pred_batch = self(data_batch)
                    pred[batch_block_inds] = pred_batch
            self = self.to(model_device)
            return pred.detach().cpu()
    
        if isinstance(data, dict):
            # Then assume that this is just to evaluate a sample: keep original here
            assert data_inds is None, "Cannot use data_inds if passing in a dataset sample."
            dev0 = data['robs'].device
            m0 = self.to(dev0)
            with torch.no_grad():
                if ffnet_target is not None:
                    pred = self.networks[ffnet_target](data)
                else:
                    pred = self(data)
    
        else:
            if batch_size is None:
                batch_size = 500   # default batch size for non-block_sample
            dev0 = data[0]['robs'].device
            self = self.to(dev0)
    
            if data_inds is None:
                data_inds = np.arange(len(data))
            NT = len(data_inds)
    
            pred = torch.zeros([NT, num_cells], device=dev0)
    
            # Must do contiguous sampling despite num_lags
            nblks = np.ceil(NT/batch_size).astype(int)
            for bb in tqdm.tqdm(range(nblks)):
                trange = np.arange(batch_size*bb-num_lags, batch_size*(bb+1))
                trange = trange[trange < NT]
                with torch.no_grad():
                    if ffnet_target is not None:
                        pred_tmp = self.networks[ffnet_target](data[data_inds[trange]])
                    else:
                        pred_tmp = self(data[data_inds[trange]])
                pred[batch_size*bb + np.arange(len(trange)-num_lags), :] = pred_tmp[num_lags:, :]
            
            self = self.to(model_device)
    
        return pred.cpu().detach()

    def change_loss( self, new_loss_type, dataset=None ):
        """
        Change the loss function for the model.

        Swaps in new loss module for model.
        Include dataset if wish to initialize loss (which happens during fit).

        Args:
            new_loss_type: The new loss type.
            dataset: The dataset to use for initializing the loss.

        Returns:
            None

        Notes:
            This method swaps in a new loss module for the model.
            If a dataset is provided, the loss will be initialized during the fit.
        """

        if isinstance(new_loss_type, str):
            self.loss_type = new_loss_type
            if self.loss_type == 'poisson' or self.loss_type == 'poissonT':
                self.loss_module = plosses.PoissonLoss_datafilter()  # defined below, but could be in own Losses.py
            elif self.loss_type in ['simple', 'poissonS']:
                self.loss_module = plosses.SimplePoissonLoss()
            elif self.loss_type in ['gaussian', 'mse']:
                self.loss_module = glosses.MseLoss_datafilter()
            else:
                print('Invalid loss function.')
                self.loss_module = None
        else: # assume passed in loss function directly
            self.loss_type = 'custom'
            self.loss_module = new_loss_type

        # module can also be called as a function (since it has a forward)
        self.loss = self.loss_module
        self.val_loss = self.loss_module

        if dataset is not None:
            if self.loss_module.unit_weighting or (self.loss_module.batch_weighting == 2):
                if dataset.trial_sample:
                    inds = dataset.train_blks
                else:
                    inds = dataset.train_inds
                self.initialize_loss(dataset, batch_size=self.batch_size, data_inds=inds) 
    # END change_loss()

    def get_weights(self, ffnet_target=0, **kwargs):
        """
        Get the weights for the specified feedforward network.

        Passed down to layer call, with optional arguments conveyed.

        Args:
            ffnet_target: The target feedforward network.
            **kwargs: Additional keyword arguments.

        Returns:
            list: The weights for the specified feedforward network.

        Notes:
            This method is passed down to the layer call with optional arguments conveyed.
        """

        assert ffnet_target < len(self.networks), "Invalid ffnet_target %d"%ffnet_target
        return self.networks[ffnet_target].get_weights(**kwargs)

    def get_readout_locations(self):
        """
        Get the readout locations and sigmas.

        Returns:
            list: The readout locations and sigmas.

        Notes:
            This method currently returns a list of readout locations and sigmas set in the readout network.
        """

        # Find readout network
        net_n = -1
        for ii in range(len(self.networks)):
            if self.networks[ii].network_type == 'readout':
                net_n = ii
        assert net_n >= 0, 'No readout network found.'
        return self.networks[net_n].get_readout_locations()

    def passive_readout(self):
        """
        Applies the passive readout function to the readout layer.

        Returns:
            None

        Notes:
            This method finds the readout layer and applies its passive_readout function.
        """

        from NDNT.modules.layers.readouts import ReadoutLayer
        assert isinstance(self.networks[-1].layers[-1], ReadoutLayer), "NDNT: Last layer is not a ReadoutLayer"
        self.networks[-1].layers[-1].passive_readout()

    def list_parameters(self, ffnet_target=None, layer_target=None):
        """
        List the parameters for the specified feedforward network and layer.

        Args:
            ffnet_target: The target feedforward network.
            layer_target: The target layer.

        Returns:
            None
        """

        if ffnet_target is None:
            ffnet_target = np.arange(len(self.networks), dtype='int32')
        elif not isinstance(ffnet_target, list):
            ffnet_target = [ffnet_target]
        for ii in ffnet_target:
            assert(ii < len(self.networks)), 'Invalid network %d.'%ii
            print("Network %d:"%ii)
            self.networks[ii].list_parameters(layer_target=layer_target)

    def set_parameters(self, ffnet_target=None, layer_target=None, name=None, val=None ):
        """
        Set the parameters for the specified feedforward network and layer.

        Args:
            ffnet_target: The target feedforward network.
            layer_target: The target layer.
            name: The name of the parameter.
            val: The value to set the parameter to.

        Returns:
            None

        Notes:
            This method sets the parameters for the specified feedforward network and layer.
        """

        if ffnet_target is None:
            ffnet_target = np.arange(len(self.networks), dtype='int32')
        elif not isinstance(ffnet_target, list):
            ffnet_target = [ffnet_target]
        for ii in ffnet_target:
            assert(ii < len(self.networks)), 'Invalid network %d.'%ii
            self.networks[ii].set_parameters(layer_target=layer_target, name=name, val=val)

    def set_reg_val(self, reg_type=None, reg_val=None, ffnet_target=None, layer_target=None ):
        """
        Set reg_values for listed network and layer targets (default 0,0).

        Args:
            reg_type: The regularization type.
            reg_val: The regularization value.
            ffnet_target: The target feedforward network.
            layer_target: The target layer.

        Returns:
            None

        Notes:
            This method sets the regularization values for the specified feedforward network and layer.
        """

        if ffnet_target is None:
            ffnet_target = 0
        assert ffnet_target < len(self.networks), "ffnet target too large (max = %d)"%len(self.networks)
        self.networks[ffnet_target].set_reg_val( reg_type=reg_type, reg_val=reg_val, layer_target=layer_target )

    def get_network_info(self, abbrev=False):
            """
            Prints out a decreiption of the model structure.
            """
            for n, N in enumerate(self.networks):
                print(f'N{n}: {N.network_type}')
                N.get_network_info(abbrev=abbrev)
    # END NDN.get_network_info()

    def info( self, expand=False):
        print("NDN %s, output net #%d:"%(self.loss_type, self.ffnet_out[0]) )
        for ii in range(len(self.networks)):
            self.networks[ii].info(ffnet_n=ii, expand=expand)
    # END NDN.info()

    def plot_filters(self, ffnet_target=0, **kwargs):
        """
        Plot the filters for the specified feedforward network.

        Args:
            ffnet_target: The target feedforward network.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.networks[ffnet_target].plot_filters(**kwargs)
    # END NDN.plot_filters()

    def update_ffnet_list(self):
        """
        the ffnet_list builds the NDN, but ideally holds a record of the regularization (a dictionary)
        since dictionaries are not saveable in a checkpoint. So, this function pulls the dictionaries
        from each reg_module back into the ffnet_list

        Args:
            None

        Returns:
            None, but updates self.ffnet_list
        """
        from copy import deepcopy

        for ii in range(len(self.networks)):
            for jj in range(len(self.networks[ii].layers)):
                # gets rid of reg_modules so not saved
                self.networks[ii].layers[jj].reg.reg_modules = nn.ModuleList()  
                self.ffnet_list[ii]['layer_list'][jj]['reg_vals'] = deepcopy(
                    self.networks[ii].layers[jj].reg.vals)
    # END NDN.update_ffnet_list()

    def save_model(self, 
                       path,
                       ffnet_list=None,
                       ffnet_out=None,
                       loss_type='poisson',
                       model_name=None,
                       working_dir='./checkpoints'):
        """
        Save the model as a zip file (with extension .ndn) containing a json file with the model parameters
        and a .ckpt file with the state_dict.

        Args:
            path: The name of the zip file to save.
            ffnet_list: The list of feedforward networks. (if this is set, it uses the ffnet_list, ffnet_out, loss_type, model_name, and working_dir arguments)
            ffnet_out: The output layer of the feedforward network.
            loss_type: The loss type.
            model_name: The name of the model.
            working_dir: The working directory.

        Returns:
            None
        """

        # the directory and name of the desired file to save
        file_dir = os.path.dirname(path)
        file_name = os.path.basename(path).split('.')[0] # remove the .zip extension (if it exists)

        # make a temporary directory in the same location as the zip file
        # where we will extract the two files
        import uuid
        temp_dir = os.path.join(file_dir, str(uuid.uuid4()))
        temp_name = os.path.join(temp_dir, file_name)

        # make the temporary directory
        os.mkdir(temp_dir)

        # package params
        if ffnet_list is None:
            self.update_ffnet_list()  # pulls current reg_values into self.ffnet_list
            ndn_params = {'ffnet_list': self.ffnet_list, 
                          'loss_type':self.loss_type,
                          'ffnet_out': self.ffnet_out,
                          'model_name': self.model_name,
                          'working_dir': self.working_dir}
        else:
            ndn_params = {'ffnet_list': ffnet_list, 
                          'loss_type':loss_type,
                          'ffnet_out': ffnet_out,
                          'model_name': model_name,
                          'working_dir': working_dir}

        # make zip_filename and ckpt_filename in the same directory as the filename
        json_filename = temp_name + '.json'
        ckpt_filename = temp_name + '.ckpt'
        with open(json_filename, 'w') as f:
            json.dump(ndn_params, f, cls=NpEncoder)
        torch.save(self.state_dict(), ckpt_filename)

        
        # zip up the two files and save to the file_dir
        fn = os.path.join(file_dir, file_name) + '.ndn'
        with zipfile.ZipFile(fn, 'w') as myzip:
            myzip.write(json_filename, os.path.basename(json_filename))
            myzip.write(ckpt_filename, os.path.basename(ckpt_filename))

        print( '  Model saved: ', fn)

        # remove the two files and the temporary directory
        os.remove(json_filename)
        os.remove(ckpt_filename)
        os.rmdir(temp_dir)

    @classmethod
    def load_model(cls, path):
        """
        Load the model from an a zip file (with extension .ndn) containing a json file with the model parameters
        and a .ckpt file with the state_dict.

        Args:
            path: The name of the zip file to load.

        Returns:
            NDN: The loaded model.
        """

        # get the file directory and name
        file_dir = os.path.dirname(path)
        file_name = os.path.basename(path).split('.')[0] # remove the .ndn extension (if it exists)

        # make a temporary directory in the same location as the zip file
        # where we will extract the two files
        import uuid
        temp_dir = os.path.join(file_dir, str(uuid.uuid4()))
        temp_name = os.path.join(temp_dir, file_name)

        # open the zip file
        with zipfile.ZipFile(path, 'r') as myzip:
            myzip.extractall(temp_dir)
        with open(temp_name+'.json', 'r') as f:
            params = json.load(f)
        ffnet_list = params['ffnet_list']
        loss_type = params['loss_type']
        ffnet_out = params['ffnet_out']
        model_name = params['model_name']
        working_dir = params['working_dir']

        # make an NDN object with the ffnet_list
        model = cls(ffnet_list=ffnet_list, 
                    loss_type=loss_type,
                    ffnet_out=ffnet_out,
                    model_name=model_name,
                    working_dir=working_dir)
        model.prepare_regularization() # this will build the reg_modules
        # load the state_dict
        state_dict = torch.load(temp_name+'.ckpt')
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # remove the two files and the temporary directory
        os.remove(temp_name+'.json')
        os.remove(temp_name+'.ckpt')
        os.rmdir(temp_dir)

        return model

    def save_model_pkl(self, filename=None, pt=False ):
        """
        Models will be saved using dill/pickle in as the filename, which can contain
        the directory information. Will be put in the CPU first
        
        Args:
            filename: The name of the file to save.
            pt: Whether to use torch.save instead of dill.

        Returns:
            None
        """

        import dill
        if filename is None:
            fn += self.model_name + '.pkl'
        else:
            fn = filename

        # Eliminate nested objects
        if self.trainer is not None:
            del self.trainer
            self.trainer = None

        print( '  Saving model at', fn)

        with open(fn, 'wb') as f:
            if pt:
                torch.save(self.to(torch.device('cpu')), f)
            else:
                dill.dump(self.to(torch.device('cpu')), f)

    def save_model_chk(self, filename=None, alt_dirname=None):
        """
        Models will be saved using dill/pickle in the directory above the version
        directories, which happen to be under the model-name itself. This assumes the
        current save-directory (notebook specific) and the model name
        
        Args:
            filename: The name of the file to save.
            alt_dirname: The alternate directory to save the file.

        Returns:
            None
        """

        import dill
        if alt_dirname is None:
            fn = './checkpoints/'
        else:
            fn = alt_dirname
            if alt_dirname[-1] != '/':
                fn += '/'
        if filename is None:
            fn += self.model_name + '.pkl'
        else :
            fn += filename
        print( '  Saving model at', fn)

        with open(fn, 'wb') as f:
            dill.dump(self, f)

    def get_null_adjusted_ll(self, sample, bits=False):
        """
        Get the null-adjusted log likelihood.

        Args:
            sample: The sample data from a dataset.
            bits: Whether to return the result in units of bits/spike.

        Returns:
            float: The null-adjusted log likelihood.
        """

        m0 = self.cpu()
        if self.loss_type == 'poisson':
            #loss = nn.PoissonNLLLoss(log_input=False, reduction='none')
            loss = self.loss_module.lossNR
        else:
            print('Whatever loss function you want is not yet here.')
        
        lnull = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy().sum(axis=0)
        #yhat = m0(sample['stim'], shifter=sample['eyepos'])
        yhat = m0(sample)
        llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy().sum(axis=0)
        rbar = sample['robs'].sum(axis=0).numpy()
        ll = (llneuron - lnull)/rbar
        if bits:
            ll/=np.log(2)
        return ll
    
    def get_activations(self, sample, ffnet_target=0, layer_target=0, NL=False):
        """
        Returns the inputs and outputs of a specified ffnet and layer
        
        Args:
            sample: dictionary of sample data from a dataset
            ffnet_target: which network to target (default: 0)
            layer_target: which layer to target (default: 0)
            NL: get activations using the nonlinearity as the module target

        Returns:
            activations dict
            with keys:
                'input' : input to layer
                'output' : output of layer
        """
        
        activations = {}

        def hook_fn(m, i, o):
            activations['input'] = i
            activations['output'] = o

        if NL:
            if self.networks[ffnet_target].layers[layer_target].NL:
                handle = self.networks[ffnet_target].layers[layer_target].NL.register_forward_hook(hook_fn)
            else:
                raise ValueError('This layer does not have a non-linearity. Call with NL=False')
        else:
            handle = self.networks[ffnet_target].layers[layer_target].register_forward_hook(hook_fn)

        out = self(sample)
        handle.remove()
        return activations

    @property
    def device( self ):
        return next(self.parameters()).device
        # just for checking -   but not definitive/great to do in general, apparently

    @classmethod
    def load_model_pkl(cls, filename=None, pt=False ):
        """
        Load a pickled model from disk.

        Args:
            filename: The path and filename.
            pt: Whether to use torch.load instead of dill.

        Returns:
            model: The loaded model on CPU.
        """

        from NDNT.utils.NDNutils import CPU_Unpickler as unpickler

        #import dill 
        #model = dill.load(filename)
        with open(filename, 'rb') as f:
            if pt:
                model = torch.load(f)
            else:
                model = unpickler(f).load()
        
        return model

    @classmethod
    def load_model_chk(cls, checkpoint_path=None, model_name=None, version=None):
        """
        Load a model from disk.

        Args:
            checkpoint_path: The path to the directory containing model checkpoints.
            model_name: The name of the model (from model.model_name).
            version: The checkpoint version (default: best).

        Returns:
            model: The loaded model.
        """
        
        from NDNT.utils.NDNutils import load_model as load

        assert checkpoint_path is not None, "Need to provide a checkpoint_path"
        assert model_name is not None, "Need to provide a model_name"

        model = load(checkpoint_path, model_name, version)
        
        return model

    def model_string( self ):
        """
        Automated way to name model

        Returns:
            str: The model name.
        """
        
        from NDNT.utils import filename_num2str

        label_chart = {
            'NDNLayer': 'N',
            'ConvLayer': 'C', 
            'TconvLayer': 'Ct',
            'STconvLayer': 'Cs',
            'ReadoutLayer':'R',
            'FixationLayer': 'F', 
            'Dim0Layer': 'D',
            'ChannelLayer': 'A'}

        Noutputs = np.prod(self.networks[-1].layers[-1].output_dims)
        name = 'M' + filename_num2str(Noutputs, num_digits=3)
        for ii in range(len(self.networks)):
            name += '_'
            for jj in range(len(self.networks[ii].layers)):
                t = "%s"%type(self.networks[ii].layers[jj])
                p, q = t.rfind('.'), t.rfind("'")
                Ltype = t[(p+1):q]
                if Ltype in label_chart:
                    name += label_chart[Ltype]
                else:
                    name += 'X'
        return name

