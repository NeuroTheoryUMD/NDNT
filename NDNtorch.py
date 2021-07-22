### NDNtorch.py
# this defines NDNclass
import numpy as np
from copy import deepcopy
import torch
from torch import nn

# from pytorch_lightning import LightningModule
from torch.nn import functional as F

#from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
#from torch.nn.common_types import _size_2_t, _size_3_t # for conv2,conv3 default
#from torch.nn.modules.utils import _triple # for posconv3

# Imports from my code
from .NDNLosses import *
from .NDNencoders import Encoder
from .NDNencoders import get_trainer
from .NDNlayer import *
from .FFnetworks import *
from .NDNutils import create_optimizer_params


class NDN:

    def __init__(self,
        ffnet_list = None,
        loss_type = 'poisson',
        ffnet_out = [-1], # Default output is last
        # not sure what val_loss does and if we need....
        reg_params = None, 
        optimizer_params = None,
        model_name='model',
        data_dir='',
        detach_core=False  # what does this do?
        ):

        # Default reg_params
        if reg_params is None:
            weight_decay = 1e-2
        else:
            weight_decay = reg_params['weight_decay']

        # Assign optimizer params
        if optimizer_params is None:
            optimizer_params = create_optimizer_params()

        # Assign loss function (from list)
        
        if isinstance(loss_type, str):
            self.loss_type = loss_type
            if loss_type == 'poisson':
                loss_func = PoissonLoss_datafilter()  # defined below, but could be in own Losses.py
            elif loss_type == 'gaussian':
                print('Gaussian loss_func not implemented yet.')
                loss_func = None
            else:
                print('Invalid loss function.')
                loss_func = None
        else: # assume passed in loss function directly
            self.loss_type = 'custom'
            loss_func = loss_type

        # Assemble FFnetworks and put into encoder (f passed in network-list)
        if type(ffnet_list) is list:
            network_list = self.assemble_ffnetworks(ffnet_list)
        else:  # assume passed in external module
            # can check type here if we want
            network_list = [ffnet_list]  # list of single network with forward

        # Check and record output of network
        if type(ffnet_out) is not list:
            ffnet_out = [ffnet_out]
        for nn in range(len(ffnet_out)):
            if ffnet_out[nn] == -1:
                ffnet_out[nn] = len(network_list)-1

        self.encoder = Encoder(
            network_list=network_list,
            loss=loss_func,
            #val_loss=val_loss, # what is this here for?
            detach_core=detach_core,
            learning_rate=optimizer_params['learning_rate'],
            batch_size=optimizer_params['batch_size'],
            num_workers=optimizer_params['num_workers'],
            data_dir=data_dir,
            optimizer=optimizer_params['optimizer'],
            weight_decay=weight_decay,
            amsgrad=optimizer_params['amsgrad'],
            betas=optimizer_params['betas'],
            max_iter=optimizer_params['max_iter'],
            ffnet_out=ffnet_out)
   
        self.opt_params = optimizer_params
        self.reg_params = reg_params
        self.name = model_name
    # END NDN.__init__

    def assemble_ffnetworks(self, ffnet_list):
        """This function takes a list of ffnetworks and puts them together 
        in order. This has to do two steps for each ffnetwork: 
        1. Plug in the inputs to each ffnetwork as specified
        2. Builds the ff-network with the input
        This returns the a 'network', which is (currently) a [lightning] module with a 
        'forward' and 'reg_loss' function specified.

        When multiple ffnet inputs are concatenated, it will always happen in the first
        (filter) dimension, so all other dimensions must match
        """
        assert type(ffnet_list) is list, "Yo ffnet_list is screwy."
        
        num_networks = len(ffnet_list)
        # Make list of lightning modules
        network_list = nn.ModuleList()

        for mm in range(num_networks):

            # Determine network input
            if ffnet_list[mm]['ffnet_n'] is None:
                # then external input (assume from one source)
                input_dims = ffnet_list[mm]['input_dims']
                assert input_dims is not None, "FFnet%d: External input dims must be specified"%mm
            else: 
                nets_in = ffnet_list[mm]['ffnet_n']
                num_input_networks = len(nets_in)

                # Concatenate input dimensions into first filter dims and make sure valid
                input_dim_list, valid_concat = [], True
                for ii in range(num_input_networks):
                    assert nets_in[ii] < mm, "FFnet%d (%d): input networks must come earlier"%(mm, ii)
                    
                    # How reads input networks depends on what type of network this is
                    if ffnet_list[mm]['ffnet_type'] == 'normal':
                        # this means that just takes output of last layer of input network
                        input_dim_list.append(network_list[nets_in[ii]].layers[-1].output_dims)
                    else:
                        print('currently no dim combo rules for non-normal ffnetworks')
                    if ii == 0:
                        num_cat_filters = input_dim_list[0][0]
                    else:
                        if input_dim_list[ii][1:] == input_dim_list[0][1:]:
                            num_cat_filters += input_dim_list[ii][0]
                        else:
                            valid_concat = False
                            print("FFnet%d: invalid concatenation %d:"%(mm,ii), input_dim_list[ii][1:], input_dim_list[0][1:] )
                assert valid_concat, "Dim concat error. Exiting."
                input_dims = [num_cat_filters] + input_dim_list[0][1:]

            ffnet_list[mm]['input_dims'] = input_dims

            # Create corresponding FFnetwork
            network_list.append( FFnetwork(ffnet_list[mm]) )

            return network_list
    # END assemble_ffnetworks

    def out( self, x, shifter=None):
        return self.encoder(x)
        #return self.encoder(x, shifter=shifter)
    
    def train( self, dataset, version=1, save_dir='./checkpoints/', name='test', save_pkl=True):

        import time

        trainer, train_dl, valid_dl = get_trainer(
            dataset, version=version,
            save_dir=save_dir, name = name,
            opt_params = self.opt_params)
            #auto_lr=self.opt_params['auto_lr'],
            #earlystopping = self.opt_params['early_stopping'],
            #batchsize= self.opt_params['batch_size'])

        t0 = time.time()
        trainer.fit( self.encoder, train_dl, valid_dl)
        t1 = time.time()

        print('  Fit complete:', t1-t0, 'sec elapsed')
        if save_pkl:
            # Pickle model-structure in checkpoint_directory
            self.save_model()
    # END NDN.train
        
    def eval_models(self, sample, bits=False, null_adjusted=True):
        '''
        get null-adjusted log likelihood
        bits=True will return in units of bits/spike
        '''
        m0 = self.encoder.cpu()
        yhat = m0(sample['stim'])
        y = sample['robs']
        dfs = sample['dfs']

        if self.loss_type == 'poisson':
            loss = nn.PoissonNLLLoss(log_input=False, reduction='none')
        else:
            print("This loss-type is not supported for eval_models.")
            loss = None

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

        if bits:
            LLneuron/=np.log(2)
        return LLneuron

    def get_filters(self):
        return self.encoder.core.get_filters()

    def get_readout_weights(self):
        return self.encoder.network.features[0].detach().cpu().numpy().squeeze()

    def get_readout_positions(self):
        return self.encoder.network.mu.detach().cpu().numpy().squeeze()


    def plot_filters(self, cmaps):
        import matplotlib.pyplot as plt
        self.encoder.network.plot_filters(cmaps=cmaps)
        plt.show()

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
            fn += self.name + '.pkl'
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
            loss = nn.PoissonNLLLoss(log_input=False, reduction='none')
        else:
            print('Whatever loss function you want is not yet here.')
        
        lnull = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy().sum(axis=0)
        #yhat = m0(sample['stim'], shifter=sample['eyepos'])
        yhat = m0(sample['stim'])
        llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy().sum(axis=0)
        rbar = sample['robs'].sum(axis=0).numpy()
        ll = (llneuron - lnull)/rbar
        if bits:
            ll/=np.log(2)
        return ll

    @classmethod
    def load_model(cls, model_name=None, filename=None, alt_dirname=None, version=None):
        """If version number given, then will load checkpointed model corresponding
        to that version"""

        import os
        import dill
        if alt_dirname is None:
            fn = './checkpoints/'
        else:
            fn = alt_dirname
            if alt_dirname != '/':
                fn += '/'
        if filename is None:
            assert model_name is not None, 'Need model_name or filename.'
            fn += model_name + '.pkl'
        else :
            fn += filename

        if not os.path.isfile(fn):
            raise ValueError(str('%s is not a valid filename' %fn))

        print( 'Loading model:', fn)
        with open(fn, 'rb') as f:
            model = dill.load(f)
        model.encoder = None
        if version is not None:
            from pathlib import Path
            assert filename is None, 'Must recover version from checkpoint dir.'
            # Then load checkpointed encoder on top of model
            chkpntdir = fn[:-4] + '/version_' + str(version) + '/'
            chkpath = Path(chkpntdir) / 'checkpoints'
            ckpt_files = list(chkpath.glob('*.ckpt'))
            model.encoder = Encoder.load_from_checkpoint(str(ckpt_files[0]))
            nn.utils.remove_weight_norm(model.encoder.core.features.layer0.conv)
            print( '-> Updated with', str(ckpt_files[0]))

        return model
