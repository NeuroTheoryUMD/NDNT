import torch
from torch import nn

import numpy as np # TODO: we can get rid of this and just use torch for math

import NDNT.metrics.poisson_loss as losses
from NDNT.utils import create_optimizer_params

import NDNT.networks as NDNnetworks

FFnets = {
    'normal': NDNnetworks.FFnetwork,
    'readout': NDNnetworks.ReadoutNetwork
}

class NDN(nn.Module):

    def __init__(self,
        ffnet_list = None,
        external_modules = None,
        loss_type = 'poisson',  # note poissonT will not use unit_normalization
        ffnet_out = None,
        optimizer_params = None,
        model_name='NDN_model',
        data_dir='./checkpoints'):

        super().__init__()
        
        if ffnet_list is not None:
            self.networks = self.assemble_ffnetworks(ffnet_list)
        else:
            self.networks = None

        if (ffnet_out is None) or (ffnet_out == -1):
            ffnet_out = len(ffnet_list)-1

        self.model_name = model_name
        self.data_dir = data_dir

        self.configure_loss(loss_type)

        # Assemble FFnetworks from if passed in network-list -- else can embed model
        assert type(ffnet_list) is list, 'FFnetwork list in NDN constructor must be a list.'
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
        
        # Assign optimizer params
        if optimizer_params is None:
            optimizer_params = create_optimizer_params()
        self.opt_params = optimizer_params
    # END NDN.__init__

    def configure_loss(self, loss_type):
        # Assign loss function (from list)
        if isinstance(loss_type, str):
            self.loss_type = loss_type
            if loss_type == 'poisson' or loss_type == 'poissonT':
                loss_func = losses.PoissonLoss_datafilter()  # defined below, but could be in own Losses.py
                if loss_type == 'poissonT':
                    loss_func.unit_normalization = False

            elif loss_type == 'gaussian':
                print('Gaussian loss_func not implemented yet.')
                loss_func = None
            else:
                print('Invalid loss function.')
                loss_func = None
        else: # assume passed in loss function directly
            self.loss_type = 'custom'
            loss_func = loss_type
            # Loss function defaults to Poisson loss with data filters (requires dfs field in dataset batch)

        # Has both reduced and non-reduced for other eval functions
        self.loss_module = loss_func
        self.loss = loss_func
        self.val_loss = self.loss

    def assemble_ffnetworks(self, ffnet_list, external_nets=None):
        """
        This function takes a list of ffnetworks and puts them together 
        in order. This has to do two steps for each ffnetwork: 

        1. Plug in the inputs to each ffnetwork as specified
        2. Builds the ff-network with the input
        
        This returns the a 'network', which is (currently) a module with a 
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
                    input_dims_list.append(networks[nets_in[ii]].layers[-1].output_dims)
                ffnet_list[mm]['input_dims_list'] = input_dims_list
        
            # Create corresponding FFnetwork
            net_type = ffnet_list[mm]['ffnet_type']
            if net_type == 'external':  # separate case because needs to pass in external modules directly
                networks.append( NDNnetworks.FFnet_external(ffnet_list[mm], external_nets))
            else:
                networks.append( FFnets[net_type](**ffnet_list[mm]) )

        return networks
    # END assemble_ffnetworks

    def compute_network_outputs(self, Xs):
        """Note this could return net_ins and net_outs, but currently just saving net_outs (no reason for net_ins yet"""
        
        assert 'networks' in dir(self), "compute_network_outputs: No networks defined in this NDN"

        net_ins, net_outs = [], []
        for ii in range(len(self.networks)):
            if self.networks[ii].ffnets_in is None:
                # then getting external input
                net_outs.append( self.networks[ii]( [Xs[self.networks[ii].xstim_n]] ) )
                net_ins.append( [Xs[self.networks[ii].xstim_n]] )
            else:
                in_nets = self.networks[ii].ffnets_in
                # Assemble network inputs in list, which will be used by FFnetwork
                inputs = []
                for mm in range(len(in_nets)):
                    inputs.append( net_outs[in_nets[mm]] )
                
                net_ins.append(inputs)
                net_outs.append( self.networks[ii](inputs) ) 
        return net_ins, net_outs
    # END compute_network_outputs

    def forward(self, Xs):
        """This applies the forwards of each network in sequential order.
        The tricky thing is concatenating multiple-input dimensions together correctly.
        Note that the external inputs is actually in principle a list of inputs"""

        net_ins, net_outs = self.compute_network_outputs( Xs )
        # For now assume its just one output, given by the first value of self.ffnet_out
        return net_outs[self.ffnet_out[0]]
    # END Encoder.forward

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
        rloss = 0
        for network in self.networks:
            rloss += network.compute_reg_loss()
        return rloss
    
    def get_trainer(self,
        version=None,
        save_dir='./checkpoints',
        name='jnkname',
        optimizer = None,
        scheduler = None,
        device = None,
        accumulated_grad_batches=None,  # default=1 if not set in opt_params
        log_activations=True,
        opt_params = None, 
        full_batch=False):
        """
            Returns a trainer and object splits the training set into "train" and "valid"
        """
        from NDNT.training import Trainer, EarlyStopping, LBFGSTrainer
        import os
    
        model = self

        trainers = {'step': Trainer, 'lbfgs': LBFGSTrainer}
        
        if opt_params is None:
                opt_params = create_optimizer_params()
                print('WARNING: using default optimization parameters.')

        if optimizer is None:
            optimizer = self.get_optimizer(**opt_params)

        if opt_params['early_stopping']:
            if isinstance(opt_params['early_stopping'], EarlyStopping):
                earlystopping = opt_params['early_stopping']
            elif isinstance(opt_params['early_stopping'], dict):
                earlystopping = EarlyStopping(patience=opt_params['early_stopping']['patience'],delta=opt_params['early_stopping']['delta'])
            else:
                earlystopping = EarlyStopping(patience=opt_params['early_stopping_patience'],delta=0.0)
        else:
            earlystopping = None

        if accumulated_grad_batches is None:
            if opt_params is None:
                accumulated_grad_batches = 1
            else:
                accumulated_grad_batches = opt_params['accumulated_grad_batches']

        # Check for device assignment in opt_params
        if device is None:
            if opt_params['device'] is not None:
                if isinstance(opt_params['device'], str):
                    device = torch.device(opt_params['device'])
                elif isinstance(opt_params['device'], torch.device):
                    device = opt_params['device']
                else:
                    raise ValueError("opt_params['device'] must be a string or torch.device")
            else:
                # Assign default device
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert isinstance(device, torch.device), "NDN.get_trainer: device must be a torch.device by this point"

        if 'optimize_graph' in opt_params.keys(): # specifies whether to attempt to optimize the graph
            optimize_graph = opt_params['optimize_graph']
            # NOTE: will cause big problems if the batch size is variable
        else:
            optimize_graph = False

        if isinstance(optimizer, torch.optim.LBFGS):
            trainer_type = 'lbfgs'
        else:
            trainer_type = 'step'
        
        trainer = trainers[trainer_type](model=model,
                optimizer=optimizer,
                early_stopping=earlystopping,
                dirpath=os.path.join(save_dir, name),
                optimize_graph=optimize_graph,
                device=device,
                scheduler=scheduler,
                accumulate_grad_batches=accumulated_grad_batches,
                max_epochs=opt_params['max_epochs'],
                log_activations=log_activations,
                version=version,
                full_batch=full_batch)

        return trainer

    def get_dataloaders(self,
            dataset,
            opt_params=None,
            batch_size=None,  # default 10
            num_workers=None, # default 1
            train_inds=None,
            val_inds=None,
            is_fixation=False,
            data_seed=None):

        from torch.utils.data import DataLoader, random_split, Subset

        if opt_params is not None:
            if batch_size is None:
                batch_size = opt_params['batch_size']
            if num_workers is None:
                num_workers = opt_params['num_workers']
        else:
            if batch_size is None:
                batch_size = 10
            if num_workers is None:
                num_workers = 1
        
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

            shuffle_data = True
            if opt_params is not None:
                if (opt_params['optimizer'] == 'LBFGS') & opt_params['full_batch']:
                    shuffle_data = False

            train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_data)
            valid_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_data)
                
        return train_dl, valid_dl
    # END NDN.get_dataloaders

    def get_optimizer(self,
            optimizer='AdamW',
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
        if optimizer=='AdamW':

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

        elif optimizer=='Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                    lr=learning_rate,
                    betas=betas)

        elif optimizer=='LBFGS':

            optimizer = torch.optim.LBFGS(
                self.parameters(), 
                history_size=history_size, max_iter=max_iter, 
                tolerance_change=tolerance_change,
                tolerance_grad=tolerance_grad,
                line_search_fn=line_search_fn)

        else:
            raise ValueError('optimizer [%s] not supported' %optimizer)
        
        return optimizer
    # END NDN.get_optimizer
    
    def prepare_regularization(self):

        for network in self.networks:
            network.prepare_regularization()

    def fit(self,
        dataset,
        version:int=None,
        save_dir:str=None,
        name:str=None,
        optimizer=None,  # this can pass in full optimizer (rather than keyword)
        scheduler=None, 
        train_inds=None,
        val_inds=None,
        opt_params=None, # Currently this params will 
        device=None,
        #accumulated_grad_batches=None, # 1 default unless overruled by opt_params # does ADAM use this? Can pass in through opt_pars, as with many others
        #full_batch=None,  # False default unless overruled by opt-params
        #log_activations=None, # True default
        seed=None):
        '''
        This is the main training loop.
        Steps:
            1. Get a trainer and dataloaders
            2. Prepare regularizers
            3. Run the main fit loop from the trainer, checkpoint, and save model
        '''
        import time

        if save_dir is None:
            save_dir = self.data_dir
        
        if name is None:
            name = self.model_name

        # Precalculate any normalization needed from the data
        if self.loss_module.unit_normalization: # where should unit normalization go?
            # compute firing rates given dataset
            avRs = self.compute_average_responses(dataset) # use whole dataset seems best versus any specific inds
            self.loss_module.set_unit_normalization(avRs) 

        # Make reg modules
        self.prepare_regularization()

        if opt_params is None:
            opt_params = self.opt_params
        else:
            # replace optimizer parameters
            self.opt_params = opt_params

        # Create dataloaders
        batchsize = opt_params['batch_size']
        train_dl, valid_dl = self.get_dataloaders(
            dataset, batch_size=batchsize, train_inds=train_inds, val_inds=val_inds, opt_params=opt_params)
            
        # get trainer 
        trainer = self.get_trainer(
            version=version,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=save_dir,
            name=name,
            device=device,
            accumulated_grad_batches=opt_params['accumulated_grad_batches'],
            log_activations=opt_params['log_activations'],
            opt_params=opt_params,
            full_batch=opt_params['full_batch'])

        t0 = time.time()
        trainer.fit(self, train_dl, valid_dl, seed=seed)
        t1 = time.time()

        print('  Fit complete:', t1-t0, 'sec elapsed')
    # END NDN.train
    
    def compute_average_responses( self, dataset, data_inds=None ):
        if data_inds is None:
            data_inds = range(len(dataset))

        #if hasattr( dataset, 'use_units') and (len(dataset.use_units) > 0):
        #    rselect = dataset.use_units
        #else:
        #    rselect = dataset.num
        
        # Iterate through dataset to compute average rates
        Rsum, Tsum = 0, 0
        for tt in data_inds:
            sample = dataset[tt]
            Tsum += torch.sum(sample['dfs'], axis=0)
            Rsum += torch.sum(torch.mul(sample['dfs'], sample['robs']), axis=0)

        return torch.divide( Rsum, Tsum.clamp(1))

    def eval_models(self, data, data_inds=None, bits=False, null_adjusted=True, batch_size=1000, num_workers=8):
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
            m0 = self.cpu()
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
                    predcnt = torch.sum(
                        torch.multiply(dfs, yhat), axis=0).detach().cpu().numpy()
                    rbar = np.divide(predcnt, Ts)
                    LLnulls = np.log(rbar)-np.divide(predcnt, np.maximum(obscnt,1))
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

    def get_weights(self, ffnet_target=0, layer_target=0, to_reshape=True, time_reverse=False):
        return self.networks[ffnet_target].layers[layer_target].get_weights(to_reshape, time_reverse=time_reverse)

    def get_readout_locations(self):
        """This currently retuns list of readout locations and sigmas -- set in readout network"""
        # Find readout network
        net_n = -1
        for ii in range(len(self.networks)):
            if self.networks[ii].network_type == 'readout':
                net_n = ii
        assert net_n >= 0, 'No readout network found.'
        return self.networks[net_n].get_readout_locations()

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
        self.networks[ffnet_target].set_reg_val( reg_type=None, reg_val=None, layer_target=None )

    def plot_filters(self, cmaps=None, ffnet_target=0, layer_target=0, num_cols=8):
        self.networks[ffnet_target].plot_filters(layer_target=layer_target, cmaps=cmaps, num_cols=num_cols)

    def save_model(self, filename=None, alt_dirname=None):
        """Models will be saved using dill/pickle in the directory above the version
        directories, which happen to be under the model-name itself. This assumes the
        current save-directory (notebook specific) and the model name"""

        import dill
        if alt_dirname is None:
            fn = './checkpoints/'
        else:
            fn = alt_dirname
            if alt_dirname != '/':
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