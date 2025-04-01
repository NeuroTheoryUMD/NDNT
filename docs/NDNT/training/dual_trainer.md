Module NDNT.training.dual_trainer
=================================

Classes
-------

`DualTrainer(optimizer2=None, **kwargs)`
:   This is the most basic trainer. There are fancier things we could add (hooks, callbacks, etc.), but I don't understand them well enough yet.
    
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

    ### Ancestors (in MRO)

    * NDNT.training.trainer.Trainer

    ### Methods

    `prepare_fit(self, model, seed=None)`
    :   This is called before fit_loop, and is used to set up the model and optimizer(s).
        
        Args:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.
            seed (int): Random seed for reproducibility.
        
        Returns:
            model (nn.Module): Pytorch Model. Needs training_step and validation_step defined.