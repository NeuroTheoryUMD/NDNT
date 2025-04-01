Module NDNT.training.trainer
============================

Classes
-------

`LBFGSTrainer(full_batch=False, **kwargs)`
:   This class is for training with the LBFGS optimizer. It is a subclass of Trainer.
    
    Args:
        full_batch (bool): Whether to use full batch optimization.

    ### Ancestors (in MRO)

    * NDNT.training.trainer.Trainer

    ### Methods

    `fit_data_dict(self, model, train_data)`
    :   Fit data that is provided in a dictionary.
        
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            train_data (dict): Training data.
        
        Returns:
            None

`TemperatureCalibratedTrainer(**kwargs)`
:   This class is for training with temperature calibration. It is a subclass of Trainer.
    
    Args:
        None

    ### Ancestors (in MRO)

    * NDNT.training.trainer.Trainer

    ### Methods

    `validate_one_epoch(self, model, val_loader)`
    :   Validation step for one epoch.
        
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            val_loader (DataLoader): Validation data.
        
        Returns:
            dict: validation loss

`Trainer(optimizer=None, device=None, dirpath='./checkpoints', version=None, max_epochs=100, early_stopping=None, accumulate_grad_batches=1, verbose=1, save_epochs=False, optimize_graph=False, log_activations=False, scheduler=None, scheduler_after='batch', scheduler_metric=None, **kwargs)`
:   This is the most basic trainer. There are fancier things we could add (hooks, callbacks, etc.), but this is bare-bones variable tracking.
    
    Args:
        optimizer (torch.optim): Pytorch optimizer to use
        scheduler (torch.scheduler): learning rate scheduler (Default: None)
        device (torch.device): Device to train on (Default: None, in which case it will use CUDA if available)
        dirpath (str): Path to save checkpoints (Default: current directory)
        version (int): Version of model to use (Default: None, in which case will create next version)
        max_epochs (int): Maximum number of epochs to train (Default: 100)
        early_stopping (EarlyStopping): If not None, will use this as the early stopping callback (Default: None)
        accumulate_grad_batches (int): How many batches to accumulate before taking optimizer step (Default: 1)
        verbose: degree of feedback to screen (int): 0=None, 1=epoch-level, 2=batch-level, 3=add early stopping info (Default 1)
        save_epochs (bool): whether to save checkpointed model at the end of every epoch (Default: False)
        optimize_graph (bool): whether to optimize graph before training
        log_activations (bool): whether to log activations (Default: False)
        scheduler: Currently not used, along with next two (Default: None)
        scheduler_after: (Default: 'batch')
        scheduler_metric: (Default: None)

    ### Descendants

    * NDNT.training.dual_trainer.DualTrainer
    * NDNT.training.trainer.LBFGSTrainer
    * NDNT.training.trainer.TemperatureCalibratedTrainer

    ### Methods

    `checkpoint_model(self, model, epoch=None, is_best=False)`
    :   Checkpoint the model.
        
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            epoch (int): Current epoch.
            is_best (bool): Whether this is the best model.
        
        Returns:
            None

    `fit(self, model, train_loader, val_loader, seed=None)`
    :   Fit the model to the data.
        
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            seed (int): Random seed for reproducibility.
        
        Returns:
            None

    `fit_loop(self, model, epochs, train_loader, val_loader)`
    :   Main training loop. This is where the model is trained.
        
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            epochs (int): Number of epochs to train.
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
        
        Returns:
            None

    `graceful_exit(self, model)`
    :   Graceful exit. This is called at the end of training or it if gets interrupted.
        
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
        
        Returns:
            None

    `prepare_fit(self, model, seed=None)`
    :   This is called before fit_loop, and is used to set up the model and optimizer.
        
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            seed (int): Random seed for reproducibility.
        
        Returns:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.

    `train_one_epoch(self, model, train_loader, epoch=0)`
    :   Train for one epoch.
        
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            train_loader (DataLoader): Training data.
            epoch (int): Current epoch.
        
        Returns:
            dict: training loss

    `train_one_step(self, model, data, batch_idx=None)`
    :   Train for one step.
        
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            data (dict): Training data.
            batch_idx (int): Current batch index.
        
        Returns:
            dict: training loss

    `validate_one_epoch(self, model, val_loader)`
    :   validation step for one epoch
        
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            val_loader (DataLoader): Validation data.
        
        Returns:
            dict: validation loss