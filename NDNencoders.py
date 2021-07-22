### NDNencoders.py
# This currently holds "Encoder" class from Jake's code, as well as 
# specific trainer information and checkpointing stuff: all specific
# to whether we use lightning and related organization

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

# import regularizers
from .NDNlayer import *
from .FFnetworks import *


class Encoder(LightningModule):
    """Note that I'm using the "Encoder" here as a placeholder (for the NDN) since the forward is 
    essentially built with other code and passed in here. I'm leaving this in place with the intent:
    1. this can be passed in directly as a whole input-output Lightning module and could integrate
       with the rest of the code (maybe?)
    2. So I can inherit a good chunk of your (Jake's) existing code that runs this. It really just
       replaces the explicit core/readout/nonlinearity structure with something else.
    3. Should be able to be easy to switch out when we decide a better structure and minimal work 
       to assemble this version"""
    
    def __init__(self,
        network_list=None,
        loss=None,
        val_loss=None,
        detach_core=False,
        learning_rate=1e-3,
        batch_size=1000,
        num_workers=0,
        data_dir='',
        optimizer='AdamW',
        weight_decay=1e-2,
        amsgrad=False,
        betas=[.9,.999],
        max_iter=10000,
        ffnet_out = [-1],
        **kwargs):

        assert network_list is not None, 'Missing encoder' 
        assert loss is not None, 'Missing loss_function' 

        super().__init__()

        # Assemble ffnetworks
        #self.core = core
        #self.readout = readout
        self.networks = network_list
        self.ffnet_out = ffnet_out
        self.detach_core = detach_core
        self.save_hyperparameters('learning_rate','batch_size',
            'num_workers', 'data_dir', 'optimizer', 'weight_decay', 'amsgrad', 'betas',
            'max_iter')          
        
        self.loss = loss
        if val_loss is None:
            self.val_loss = loss
        else:
            self.val_loss = val_loss
    # END Encoder.__init__

    def compute_network_outputs( self, Xs):
        if type(Xs) is not list:
            Xs = [Xs]

        net_ins, net_outs = [], []
        for nn in range(len(self.networks)):
            if self.networks[nn].ffnets_in is None:
                # then getting external input
                #net_ins.append( Xs[self.networks[nn].xstim_n] )
                net_outs.append( self.networks[nn]( Xs[self.networks[nn].xstim_n] ) )
            else:
                # Concatenate the previous relevant network outputs
                in_nets = self.networks[nn].ffnets_in
                input_cat = net_outs[in_nets[0]]
                for mm in range(1, len(in_nets)):
                    input_cat = torch.cat( (input_cat, net_outs[in_nets[mm]]), 1 )

                #net_ins.append( input_cat )
                net_outs.append( self.networks[nn](input_cat) ) 
        return net_ins, net_outs
        #return net_outs


    def forward(self, Xs, shifter=None):
        """This applies the forwards of each network in sequential order.
        The tricky thing is concatenating multiple-input dimensions together correctly.
        Note that the external inputs is actually in principle a list of inputs"""

        net_ins, net_outs = self.compute_network_outputs( Xs )

        # For now assume its just one output, given by the first value of self.ffnet_out
        return net_outs[self.ffnet_out[0]]
    # END Encoder.forward

    #def training_step(self, batch, batch_idx):  # batch_indx not used, right?
    def training_step(self, batch):
        x = batch['stim']
        y = batch['robs']
        dfs = batch['dfs']

        #if self.readout.shifter is not None and batch['eyepos'] is not None and self.readout.shifter:
        #    y_hat = self(x, shifter=batch['eyepos'])
        #else:
        y_hat = self(x)

        loss = self.loss(y_hat, y, dfs)
        #regularizers = int(not self.detach_core) * self.core.regularizer() + self.readout.regularizer()
        regularizers = int(not self.detach_core) * self.network.regularizer()

        self.log('train_loss', loss + regularizers, on_step=False, on_epoch=True)
        #self.log('rloss', regularizers, on_step=False, on_epoch=True)
        return {'loss': loss + regularizers}
    # END Encoder.training_step

    #def validation_step(self, batch, batch_idx):
    def validation_step(self, batch):
        x = batch['stim']
        y = batch['robs']
        dfs = batch['dfs']
        #if "shifter" in dir(self.readout) and batch['eyepos'] is not None and self.readout.shifter:
        #if self.readout.shifter is not None and batch['eyepos'] is not None and self.readout.shifter:
        #    y_hat = self(x, shifter=batch['eyepos'])
        #else:

        y_hat = self(x)
        loss = self.val_loss(y_hat, y, dfs)
        reg_loss = int(not self.detach_core) * self.network.regularizer()
        self.log('val_loss', loss)
        self.log('reg_loss', reg_loss, on_step=False, on_epoch=True)
        return {'loss': loss}
    # END Encoder.validation_step

    def validation_epoch_end(self, validation_step_outputs):
        # logging
        if(self.current_epoch==1):
            self.logger.experiment.add_text('core', str(dict(self.core.hparams)))
            self.logger.experiment.add_text('readout', str(dict(self.readout.hparams)))

        #avg_val_loss = torch.tensor([x['loss'] for x in validation_step_outputs]).mean()
        #avg_reg_loss = torch.tensor([x['reg_loss'] for x in validation_step_outputs]).mean()
        #avg_reg_loss = torch.tensor(3.0)
        #tqdm_dict = {'val_loss': avg_val_loss}

        #return {
        #        'progress_bar': tqdm_dict,
        #        'log': {'val_loss': avg_val_loss, 'reg_lossX': avg_reg_loss}}

    def configure_optimizers(self):
        
        if self.hparams.optimizer=='LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(),
                lr=self.hparams.learning_rate,
                max_iter=10000) #, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100)
        elif self.hparams.optimizer=='AdamW':
            optimizer = torch.optim.AdamW(self.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                weight_decay=self.hparams.weight_decay,
                amsgrad=self.hparams.amsgrad)
        elif self.hparams.optimizer=='Adam':
            optimizer = torch.optim.Adam(self.parameters(),
            lr=self.hparams.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
    def on_save_checkpoint(self, checkpoint):
        # track the core, readout, shifter class and state_dicts
        checkpoint['core_type'] = type(self.core)
        checkpoint['core_hparams'] = self.core.hparams
        checkpoint['core_state_dict'] = self.core.state_dict()

        checkpoint['readout_type'] = type(self.readout)
        checkpoint['readout_hparams'] = self.readout.hparams
        checkpoint['readout_state_dict'] = self.readout.state_dict() # TODO: is this necessary or included in self state_dict?

        # checkpoint['shifter_type'] = type(self.shifter)
        # if checkpoint['shifter_type']!=type(None):
        #     checkpoint['shifter_hparams'] = self.shifter.hparams
        #     checkpoint['shifter_state_dict'] = self.shifter.state_dict() # TODO: is this necessary or included in model state_dict?

    def on_load_checkpoint(self, checkpoint):
        # properly handle core, readout, shifter state_dicts
        self.core = checkpoint['core_type'](**checkpoint['core_hparams'])
        self.readout = checkpoint['readout_type'](**checkpoint['readout_hparams'])
        # if checkpoint['shifter_type']!=type(None):
        #     self.shifter = checkpoint['shifter_type'](**checkpoint['shifter_hparams'])
        #     self.shifter.load_state_dict(checkpoint['shifter_state_dict'])
        self.core.load_state_dict(checkpoint['core_state_dict'])
        self.readout.load_state_dict(checkpoint['readout_state_dict'])


def get_trainer(dataset,
        version=1,
        save_dir='./checkpoints',
        name='jnkname',
        opt_params = None,
        #auto_lr=False,
        #batchsize=1000,
        #earlystopping=True,
        seed=None):
    """
    Returns a pytorch lightning trainer and splits the training set into "train" and "valid"
    """
    from torch.utils.data import Dataset, DataLoader, random_split
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TestTubeLogger
    from pathlib import Path

    save_dir = Path(save_dir)
    batchsize = opt_params['batch_size']
    #early_stopping = opt_params['early_stopping']

    n_val = np.floor(len(dataset)/5).astype(int)
    n_train = (len(dataset)-n_val).astype(int)

    gd_train, gd_val = random_split(dataset, lengths=[n_train, n_val])

    # build dataloaders
    train_dl = DataLoader(gd_train, batch_size=batchsize, num_workers=opt_params['num_workers'])
    valid_dl = DataLoader(gd_val, batch_size=batchsize, num_workers=opt_params['num_workers'])

    # Train
    early_stop_callback = EarlyStopping(
        monitor='val_loss', 
        min_delta=0.0,
        patience=opt_params['early_stopping_patience'])
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    logger = TestTubeLogger(
        save_dir=save_dir,
        name=name,
        version=version  # fixed to one to ensure checkpoint load
    )

    # ckpt_folder = save_dir / sessid / 'version_{}'.format(version) / 'checkpoints'
    if opt_params['early_stopping']:
        trainer = Trainer(gpus=opt_params['num_gpus'], 
            callbacks=[early_stop_callback],
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            deterministic=False,
            gradient_clip_val=0,
            accumulate_grad_batches=1,
            progress_bar_refresh_rate=opt_params['progress_bar_refresh'],
            max_epochs=opt_params['max_epochs'],
            auto_lr_find=opt_params['auto_lr'])
    else:
        trainer = Trainer(gpus=opt_params['num_gpus'],
            checkpoint_callback=checkpoint_callback,
            logger=logger,
            deterministic=False,
            gradient_clip_val=0,
            accumulate_grad_batches=1,
            progress_bar_refresh_rate=opt_params['progress_bar_refresh'],
            max_epochs=opt_params['max_epochs'],
            auto_lr_find=opt_params['auto_lr'])

    if seed:
        seed_everything(seed)

    return trainer, train_dl, valid_dl


def find_best_epoch(ckpt_folder):
    # from os import listdir
    # import glob
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: Integer of the highest epoch reached by the checkpoints.
    """
    try:
        # ckpt_files = listdir(ckpt_folder)  # list of strings
        ckpt_files = list(ckpt_folder.glob('*.ckpt'))
        #epochs = [int(str(filename)[str(filename).find('=')+1:-5]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
        # the above could not handle longer filenames
        epochs = []
        for filename in ckpt_files:
            i1 = int(str(filename).find('='))+1
            i2 = np.min([int(str(filename).find('-')), int(str(filename).find('.'))])
            epochs.append(int(str(filename)[i1:i2]))  # 'epoch={int}.ckpt' filename format
        out = max(epochs)
    except FileNotFoundError:
        out = None
    return out
