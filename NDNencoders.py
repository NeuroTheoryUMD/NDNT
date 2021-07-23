### NDNencoders.py
# This currently holds "Encoder" class from Jake's code, as well as 
# specific trainer information and checkpointing stuff: all specific
# to whether we use lightning and related organization

import torch
from torch import nn

# import regularizers
from NDNlayer import *
from FFnetworks import *


class Encoder(nn.Module):
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
        ffnet_out = [-1]):

        assert network_list is not None, 'Missing encoder' 
        assert loss is not None, 'Missing loss_function' 

        super().__init__()

        # Assemble ffnetworks
        self.networks = network_list
        self.ffnet_out = ffnet_out
        
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

    def training_step(self, batch, batch_idx=None):  # batch_indx not used, right?
        x = batch['stim'] # TODO: this will have to handle the multiple Xstims in the future
        y = batch['robs']
        dfs = batch['dfs']

        #if self.readout.shifter is not None and batch['eyepos'] is not None and self.readout.shifter:
        #    y_hat = self(x, shifter=batch['eyepos'])
        #else:
        y_hat = self(x)

        loss = self.loss(y_hat, y, dfs)

        regularizers = self.compute_reg_loss()

        return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}
    # END Encoder.training_step

    def validation_step(self, batch, batch_idx=None):
        x = batch['stim']
        y = batch['robs']
        dfs = batch['dfs']
        #if "shifter" in dir(self.readout) and batch['eyepos'] is not None and self.readout.shifter:
        #if self.readout.shifter is not None and batch['eyepos'] is not None and self.readout.shifter:
        #    y_hat = self(x, shifter=batch['eyepos'])
        #else:

        y_hat = self(x)
        loss = self.val_loss(y_hat, y, dfs)
        
        reg_loss = self.compute_reg_loss()
        
        return {'loss': loss, 'val_loss': loss, 'reg_loss': reg_loss}
    # END Encoder.validation_step

    
    # def on_save_checkpoint(self, checkpoint):
    #     # track the core, readout, shifter class and state_dicts
    #     # this is all temporary: to be replaced with new logging
    #     checkpoint['core_type'] = type(self.networks[0])
    #     checkpoint['core_hparams'] = self.networks[0].hparams
    #     checkpoint['core_state_dict'] = self.networks[0].state_dict()

    #     # checkpoint['shifter_type'] = type(self.shifter)
    #     # if checkpoint['shifter_type']!=type(None):
    #     #     checkpoint['shifter_hparams'] = self.shifter.hparams
    #     #     checkpoint['shifter_state_dict'] = self.shifter.state_dict() # TODO: is this necessary or included in model state_dict?

    def compute_reg_loss(self):
        rloss = 0
        for network in self.networks:
            rloss += network.compute_reg_loss()
        return rloss
### END Encoder class


def get_trainer(dataset, model,
        version=None,
        save_dir='./checkpoints',
        name='jnkname',
        opt_params = None):
    """
    Returns a pytorch lightning trainer and splits the training set into "train" and "valid"
    """
    from torch.utils.data import DataLoader, random_split
    from trainers import Trainer, EarlyStopping
    from pathlib import Path

    save_dir = Path(save_dir)
    batchsize = opt_params['batch_size']

    n_val = np.floor(len(dataset)/5).astype(int)
    n_train = (len(dataset)-n_val).astype(int)

    gd_train, gd_val = random_split(dataset, lengths=[n_train, n_val])

    # build dataloaders
    train_dl = DataLoader(gd_train, batch_size=batchsize, num_workers=opt_params['num_workers'])
    valid_dl = DataLoader(gd_val, batch_size=batchsize, num_workers=opt_params['num_workers'])

    # get optimizer: In theory this probably shouldn't happen here because it needs to know the model
    # but this was the easiest insertion point I could find for now
    if opt_params['optimizer']=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                lr=opt_params['learning_rate'],
                betas=opt_params['betas'],
                weight_decay=opt_params['weight_decay'],
                amsgrad=opt_params['amsgrad'])

    elif opt_params['optimizer']=='Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                lr=opt_params['learning_rate'],
                betas=opt_params['betas'])

    elif opt_params['optimizer']=='LBFGS':
        from LBFGS import LBFGS
        optimizer = LBFGS(model.parameters(), lr=opt_params['learning_rate'], history_size=10, line_search='Wolfe', debug=False)

    elif opt_params['optimizer']=='FullBatchLBFGS':
        from LBFGS import FullBatchLBFGS
        optimizer = FullBatchLBFGS(model.parameters(), lr=opt_params['learning_rate'], history_size=10, line_search='Wolfe', debug=False)

    else:
        raise ValueError('optimizer [%s] not supported' %opt_params['optimizer'])
        

    if opt_params['early_stopping']:
        if isinstance(opt_params['early_stopping'], EarlyStopping):
            earlystopping = opt_params['early_stopping']
        elif isinstance(opt_params['early_stopping'], dict):
            earlystopping = EarlyStopping(patience=opt_params['early_stopping']['patience'],delta=opt_params['early_stopping']['delta'])
        else:
            earlystopping = EarlyStopping(patience=opt_params['early_stopping_patience'],delta=0.0)
    else:
        earlystopping = None

    trainer = Trainer(model, optimizer, early_stopping=earlystopping,
            dirpath=save_dir,
            version=version) # TODO: how do we want to handle name? Variable name is currently unused

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