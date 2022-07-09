# from sklearn.utils import shuffle
import torch
from torch import nn
from functools import reduce


import numpy as np # TODO: we can get rid of this and just use torch for math

import NDNT.metrics.poisson_loss as plosses
import NDNT.metrics.mse_loss as glosses
from NDNT.utils import create_optimizer_params

import NDNT.networks as NDNnetworks

FFnets = {
    'normal': NDNnetworks.FFnetwork,
    'add': NDNnetworks.FFnetwork,  # just controls how inputs are concatenated
    'mult': NDNnetworks.FFnetwork, # just controls how inputs are concatenated
    'scaffold': NDNnetworks.ScaffoldNetwork, # forward concatenates all layers (default: convolutional)
    'readout': NDNnetworks.ReadoutNetwork
}

class NDN(nn.Module):

    def __init__(self,
        ffnet_list = None,
        layer_list = None,  # can just import layer list if consisting of single ffnetwork (simple ffnet)
        external_modules = None,
        loss_type = 'poisson', 
        ffnet_out = None,
        optimizer_params = None,
        model_name=None,
        working_dir='./checkpoints'):

        super().__init__()
        
        if ffnet_list is None:
            # then must be a single ffnet specified by layer_list
            assert layer_list is not None, "NDN: must specify either ffnet_list or layer_list."
            from .networks import FFnetwork
            ffnet_list = [FFnetwork.ffnet_dict(layer_list=layer_list)]
        else:
            assert layer_list is None, "NDN: cannot specify ffnet_list and layer_list at same time."
 
        assert type(ffnet_list) is list, 'FFnetwork list in NDN constructor must be a list.'

        if (ffnet_out is None) or (ffnet_out == -1):
            ffnet_out = len(ffnet_list)-1

        self.working_dir = working_dir

        # Configure loss
        if isinstance(loss_type, str):
            self.loss_type = loss_type
            if loss_type == 'poisson' or loss_type == 'poissonT':
                self.loss_module = plosses.PoissonLoss_datafilter()  # defined below, but could be in own Losses.py
            elif loss_type in ['gaussian', 'mse']:
                self.loss_module = glosses.MseLoss_datafilter()
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
    # END NDN.__init__

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
        """
        assert type(ffnet_list) is list, "ffnet_list must be a list."
        
        num_networks = len(ffnet_list)
        # Make list of pytorch modules
        networks = nn.ModuleList()

        for mm in range(num_networks):

            # Determine internal network input to each subsequent network (if exists)
            if ffnet_list[mm]['ffnet_n'] is not None:
                nets_in = ffnet_list[mm]['ffnet_n']
                input_dims_list = []
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
        """Note this could return net_ins and net_outs, but currently just saving net_outs (no reason for net_ins yet"""
        
        assert 'networks' in dir(self), "compute_network_outputs: No networks defined in this NDN"

        net_ins, net_outs = [], []
        for ii in range(len(self.networks)):
            if self.networks[ii].ffnets_in is None:
                # then getting external input
                net_ins.append( [Xs[self.networks[ii].xstim_n]] )
                net_outs.append( self.networks[ii]( net_ins[-1] ) )
            else:
                in_nets = self.networks[ii].ffnets_in
                # Assemble network inputs in list, which will be used by FFnetwork
                inputs = []
                for mm in range(len(in_nets)):
                    inputs.append( net_outs[in_nets[mm]] )
                
                net_ins.append(inputs)
                net_outs.append( self.networks[ii](inputs) ) 
        return net_ins, net_outs
    # END NDNT.compute_network_outputs

    def forward(self, Xs):
        """This applies the forwards of each network in sequential order.
        The tricky thing is concatenating multiple-input dimensions together correctly.
        Note that the external inputs is actually in principle a list of inputs"""

        net_ins, net_outs = self.compute_network_outputs( Xs )
        # For now assume its just one output, given by the first value of self.ffnet_out
        return net_outs[self.ffnet_out[0]]
    # END NDNT.forward

    def training_step(self, batch, batch_idx=None):  # batch_indx not used, right?
     
        y = batch['robs']
        dfs = batch['dfs']

        y_hat = self(batch)

        loss = self.loss(y_hat, y, dfs)

        regularizers = self.compute_reg_loss()

        return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}
    # END Encoder.training_step

    def validation_step(self, batch, batch_idx=None):
        
        y = batch['robs']
        dfs = batch['dfs']

        y_hat = self(batch)
        loss = self.val_loss(y_hat, y, dfs)
        
        reg_loss = self.compute_reg_loss()
        
        return {'loss': loss, 'val_loss': loss, 'reg_loss': reg_loss}
    
    def compute_reg_loss(self):
        rloss = []
        for network in self.networks:
            rloss.append(network.compute_reg_loss())
        return reduce(torch.add, rloss)
    
    def get_trainer(self,
        version=None,
        save_dir='./checkpoints',
        name='jnkname',
        optimizer = None,
        scheduler = None,
        device = None,
        optimizer_type='AdamW',
        early_stopping=False,
        early_stopping_patience=5,
        early_stopping_delta=0.0,
        optimize_graph=False,
        **kwargs):
        """
            Returns a trainer and object splits the training set into "train" and "valid"
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
            num_workers=1,
            is_fixation=False,
            full_batch=False,
            data_seed=None,
            device=None,
            **kwargs):

        from torch.utils.data import DataLoader, random_split, Subset

        covariates = list(dataset[0].keys())
        #print('Dataset covariates:', covariates)

        if train_inds is None or val_inds is None:
            # check dataset itself
            if hasattr(dataset, 'val_inds') and \
                (dataset.train_inds is not None) and (dataset.val_inds is not None):
                train_inds = dataset.train_inds
                val_inds = dataset.val_inds
            else:
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
        if is_fixation:
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
        else:
            train_ds = Subset(dataset, train_inds)
            val_ds = Subset(dataset, val_inds)

            shuffle_data = True  # no reason not to shuffle -- samples everything over epoch

            train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_data)
            valid_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_data)
            
        return train_dl, valid_dl
    # END NDN.get_dataloaders

    def get_optimizer(self,
            optimizer_type='AdamW',
            learning_rate=0.001,
            weight_decay=0.01,
            amsgrad=False,
            betas=(0.9, 0.999),
            max_iter=10,
            history_size=4,
            tolerance_change=1e-3,
            tolerance_grad=1e-4,
            line_search_fn=None,
            **kwargs):
        
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

        for network in self.networks:
            network.prepare_regularization()

    def fit(self,
        dataset,
        train_inds=None,
        val_inds=None,
        seed=None, # this currently seed for data AND optimization
        save_dir:str=None,
        version:int=None,
        #name:str=None,
        optimizer=None,  # this can pass in full optimizer (rather than keyword)
        scheduler=None,
        batch_size=None,
        force_dict_training=False,  # will force dict-based training instead of using data-loaders for LBFGS
        device=None,
        **kwargs  # kwargs replaces explicit opt_params, which can list some or all of the following
        ):

        '''
        This is the main training loop.
        Steps:
            1. Get a trainer and dataloaders
            2. Prepare regularizers
            3. Run the main fit loop from the trainer, checkpoint, and save model
        '''
        import time

        assert batch_size is not None, "NDN.fit() must be passed batch_size in optimization params."

        if save_dir is None:
            save_dir = self.working_dir
        
        # Should be set ahead of time
        name = self.model_name

        # Determine train_inds and val_inds: read from dataset if not specified, or warn
        if train_inds is None:
            # check dataset itself
            if hasattr(dataset, 'train_inds'):
                if dataset.train_inds is not None:
                    train_inds = dataset.train_inds
                if dataset.val_inds is not None:
                    val_inds = dataset.val_inds
            else:
                train_inds = range(len(dataset))
                print( "Warning: no train_inds specified. Using full dataset passed in.")

        # Check to see if loss-flags require any initialization using dataset information 
        if self.loss_module.unit_weighting or (self.loss_module.batch_weighting == 2):
            if force_dict_training:
                batch_size = len(train_inds)
            self.initialize_loss(dataset, batch_size=batch_size, data_inds=train_inds) 

        # Prepare model regularization (will build reg_modules)
        self.prepare_regularization()

        if 'verbose' in kwargs:
            if kwargs['verbose'] > 0:
                print( 'Model:', self.model_name)
        # Create dataloaders / 
        train_dl, valid_dl = self.get_dataloaders(
            dataset, batch_size=batch_size, train_inds=train_inds, val_inds=val_inds, data_seed=seed, **kwargs)

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

        '''
        This is the main training loop.
        Steps:
            1. Get a trainer and dataloaders
            2. Prepare regularizers
            3. Run the main fit loop from the trainer, checkpoint, and save model
        '''
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
        from torch.utils.data import DataLoader
        train_dl = DataLoader( train_ds, batch_size=batch_size, num_workers=num_workers) 
        valid_dl = DataLoader( val_ds, batch_size=batch_size, num_workers=num_workers) 

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

        print('  Fit complete:', t1-t0, 'sec elapsed')
        self.trainer = trainer
    # END NDN.fit_dl
    
    def initialize_loss( self, dataset=None, batch_size=None, data_inds=None, batch_weighting=None, unit_weighting=None ):
        """
        Interacts with loss_module to set loss flags and/or pass in dataset information
        Without dataset, it will just set flags 'batch_weighting' [-1, 0,1,2] and 'unit_weighting'
        With dataset, it will set av_batch_size and unit_weights based on average rate
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
        self, data, data_inds=None, bits=False, null_adjusted=True,
        batch_size=1000, num_workers=0, **kwargs ):
        '''
        get null-adjusted log likelihood (if null_adjusted = True)
        bits=True will return in units of bits/spike

        Note that data will be assumed to be a dataset, and data_inds will have to be specified batches
        from dataset.__get_item__()
        '''
        
        # Switch into evalulation mode
        self.eval()

        if isinstance(data, dict): 
            # Then assume that this is just to evaluate a sample: keep original here
            assert data_inds is None, "Cannot use data_inds if passing in a dataset sample."
            dev0 = data['robs'].device
            m0 = self.to(dev0)
            yhat = m0(data)
            y = data['robs']
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
                    rbar = np.divide(obscnt, Ts)
                    LLnulls = np.log(rbar)-1

                    LLneuron = -LLneuron - LLnulls             

            return LLneuron  # end of the old method

        else:
            # This will be the 'modern' eval_models using already-defined self.loss_module
            # In this case, assume data is dataset
            if data_inds is None:
                data_inds = list(range(len(data)))

            data_dl, _ = self.get_dataloaders(data, batch_size=batch_size, num_workers=num_workers, train_inds=data_inds, val_inds=data_inds)

            LLsum, Tsum, Rsum = 0, 0, 0
            from tqdm import tqdm
            d = next(self.parameters()).device  # device the model is on
            for data_sample in tqdm(data_dl, desc='Eval models'):
                # data_sample = data[tt]
                for dsub in data_sample.keys():
                    if data_sample[dsub].device != d:
                        data_sample[dsub] = data_sample[dsub].to(d)
                with torch.no_grad():
                    pred = self(data_sample)
                    LLsum += self.loss_module.unit_loss( 
                        pred, data_sample['robs'], data_filters=data_sample['dfs'], temporal_normalize=False)
                    Tsum += torch.sum(data_sample['dfs'], axis=0)
                    Rsum += torch.sum(torch.mul(data_sample['dfs'], data_sample['robs']), axis=0)
            LLneuron = torch.divide(LLsum, Rsum.clamp(1) )

            # Null-adjust
            if null_adjusted:
                rbar = torch.divide(Rsum, Tsum.clamp(1))
                LLnulls = torch.log(rbar)-1
                LLneuron = -LLneuron - LLnulls 
        if bits:
            LLneuron/=np.log(2)

        return LLneuron.detach().cpu().numpy()

    ### NOTE THIS FUNCTION IS NOT DONE YET -- it would ideally step through all relevant batches like eval_model 
    def generate_predictions( self, data, data_inds=None, batch_size=1000, num_workers=0, **kwargs ):
        '''
        Note that data will be assumed to be a dataset, and data_inds will have to be specified batches
        from dataset.__get_item__()
        '''
        
        # Switch into evalulation mode
        self.eval()

        if isinstance(data, dict): 
            # Then assume that this is just to evaluate a sample: keep original here
            assert data_inds is None, "Cannot use data_inds if passing in a dataset sample."
            dev0 = data['robs'].device
            m0 = self.to(dev0)
            yhat = m0(data)
            y = data['robs']
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
                
            return pred  # end of the old method

        else:
            # This will be the 'modern' eval_models using already-defined self.loss_module
            # In this case, assume data is dataset
            if data_inds is None:
                data_inds = list(range(len(data)))

            data_dl, _ = self.get_dataloaders(data, batch_size=batch_size, num_workers=num_workers, train_inds=data_inds, val_inds=data_inds)

            LLsum, Tsum, Rsum = 0, 0, 0
            from tqdm import tqdm
            d = next(self.parameters()).device  # device the model is on
            for data_sample in tqdm(data_dl, desc='Eval models'):
                # data_sample = data[tt]
                for dsub in data_sample.keys():
                    if data_sample[dsub].device != d:
                        data_sample[dsub] = data_sample[dsub].to(d)
                with torch.no_grad():
                    pred = self(data_sample)
                    LLsum += self.loss_module.unit_loss( 
                        pred, data_sample['robs'], data_filters=data_sample['dfs'], temporal_normalize=False)
                    Tsum += torch.sum(data_sample['dfs'], axis=0)
                    Rsum += torch.sum(torch.mul(data_sample['dfs'], data_sample['robs']), axis=0)
            LLneuron = torch.divide(LLsum, Rsum.clamp(1) )

            # Null-adjust
            if null_adjusted:
                rbar = torch.divide(Rsum, Tsum.clamp(1))
                LLnulls = torch.log(rbar)-1
                LLneuron = -LLneuron - LLnulls 
        if bits:
            LLneuron/=np.log(2)

        return LLneuron.detach().cpu().numpy()



    def get_weights(self, ffnet_target=0, **kwargs):
        """passed down to layer call, with optional arguments conveyed"""
        assert ffnet_target < len(self.networks), "Invalid ffnet_target %d"%ffnet_target
        return self.networks[ffnet_target].get_weights(**kwargs)

    def get_readout_locations(self):
        """This currently retuns list of readout locations and sigmas -- set in readout network"""
        # Find readout network
        net_n = -1
        for ii in range(len(self.networks)):
            if self.networks[ii].network_type == 'readout':
                net_n = ii
        assert net_n >= 0, 'No readout network found.'
        return self.networks[net_n].get_readout_locations()

    def passive_readout(self):
        """This finds the readout layer and applies its passive_readout function"""
        from NDNT.modules.layers.readouts import ReadoutLayer
        assert isinstance(self.networks[-1].layers[-1], ReadoutLayer), "NDNT: Last layer is not a ReadoutLayer"
        self.networks[-1].layers[-1].passive_readout()

    def list_parameters(self, ffnet_target=None, layer_target=None):
        if ffnet_target is None:
            ffnet_target = np.arange(len(self.networks), dtype='int32')
        elif not isinstance(ffnet_target, list):
            ffnet_target = [ffnet_target]
        for ii in ffnet_target:
            assert(ii < len(self.networks)), 'Invalid network %d.'%ii
            print("Network %d:"%ii)
            self.networks[ii].list_parameters(layer_target=layer_target)

    def set_parameters(self, ffnet_target=None, layer_target=None, name=None, val=None ):
        """Set parameters for listed layer or for all layers."""
        if ffnet_target is None:
            ffnet_target = np.arange(len(self.networks), dtype='int32')
        elif not isinstance(ffnet_target, list):
            ffnet_target = [ffnet_target]
        for ii in ffnet_target:
            assert(ii < len(self.networks)), 'Invalid network %d.'%ii
            self.networks[ii].set_parameters(layer_target=layer_target, name=name, val=val)

    def set_reg_val(self, reg_type=None, reg_val=None, ffnet_target=None, layer_target=None ):
        """Set reg_values for listed network and layer targets (default 0,0)."""
        if ffnet_target is None:
            ffnet_target = 0
        assert ffnet_target < len(self.networks), "ffnet target too large (max = %d)"%len(self.networks)
        self.networks[ffnet_target].set_reg_val( reg_type=reg_type, reg_val=reg_val, layer_target=layer_target )

    def plot_filters(self, ffnet_target=0, **kwargs):
        self.networks[ffnet_target].plot_filters(**kwargs)

    def save_model(self, filename=None, alt_dirname=None):
        """Models will be saved using dill/pickle in the directory above the version
        directories, which happen to be under the model-name itself. This assumes the
        current save-directory (notebook specific) and the model name"""

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
        '''
        get null-adjusted log likelihood
        bits=True will return in units of bits/spike
        '''
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
        Output:
            activations: dictionary of activations
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
        # just for checking -- but not definitive/great to do in general, apparently
    
    @classmethod
    def load_model(cls, checkpoint_path=None, model_name=None, version=None):
        '''
            Load a model from disk.
            Arguments:
                checkpoint_path: path to directory containing model checkpoints
                model_name: name of model (from model.model_name)
                version: which checkpoint to load (default: best)
            Returns:
                model: loaded model
        '''
        
        from NDNT.utils.NDNutils import load_model as load

        assert checkpoint_path is not None, "Need to provide a checkpoint_path"
        assert model_name is not None, "Need to provide a model_name"

        model = load(checkpoint_path, model_name, version)
        
        return model

    def model_string( self ):
        """Automated way to name model"""
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
