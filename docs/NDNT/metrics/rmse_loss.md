Module NDNT.metrics.rmse_loss
=============================

Classes
-------

`RmseLoss(**kwargs)`
:   This is initialized with default behavior that requires no knowledge of the dataset:
    unit_weighting = False, meaning no external weighting of units, such as by firing rate)
        True corresponds to weighting being stored in buffer unit_weights        
    batch_weighting = 0: 'batch_size', 1: 'data_filter', 2: 'av_batch_size', -1: "unnormalized"  default 0
        Default requires no info from dataset: LL is normalized by time steps in batch, but otherwise
        could use average batch size (for consistency across epoch), or datafilter (to get correct per-neuron) 
    For example using batch size and unit_weights corresponding to reciprocal of probability of spike per bin will
    give standard LL/spk. 
    Note that default is using batch_size, and info must be oassed in using 'set_loss_weighting' to alter behavior
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, pred, target, data_filters=None) ‑> Callable[..., Any]`
    :   This is the forward function for the loss function. It calculates the loss based on the input and target data.
        The loss is calculated as the mean squared error between the prediction and target data.
        
        Args:
            pred (torch.tensor): predicted data
            target (torch.tensor): target data
            data_filters (torch.tensor): data filters for each unit
        
        Returns:
            loss (torch.tensor): mean squared error loss

    `set_loss_weighting(self, batch_weighting=None, unit_weighting=None, unit_weights=None, av_batch_size=None)`
    :   This changes default loss function weights to adjust for the dataset by setting two flags:
            unit_weighting: whether to weight neurons by different amounts in loss function (e.g., av spike rate)
            batch_weighting: how much to weight each batch, dividing by following quantity
                0 (default) 'batch
                1 'data_filters': weight each neuron individually by the amount of data
                2 'av_batch_size': weight by average batch size. Needs av_batch_size set/initialized from dataset info
                -1 'unnormalized': no weighing at all. this will implicitly increase with batch size
        
        Args:
            batch_weighting (int): 0, 1, 2, -1
            unit_weighting (bool): whether to weight units
            unit_weights (torch.tensor): weights for each unit
            av_batch_size (int): average batch size
        
        Returns:
            None

    `unit_loss(self, pred, target, data_filters=None, temporal_normalize=True)`
    :   This should be equivalent of forward, without sum over units
        Currently only true if batch_weighting = 'data_filter'.
        
        Args:
            pred (torch.tensor): predicted data
            target (torch.tensor): target data
            data_filters (torch.tensor): data filters for each unit
            temporal_normalize (bool): whether to normalize by time steps
        
        Returns:
            unitloss (torch.tensor): mean squared error loss