### NDNLosses.py
# Classes corresponding to each loss function 
# Note currently a Lightning module
# All classes should have 
#    name
#    forward: returns loss within optimizer
#    unit_loss: returns loss for each output unit (without reduction)
#    Also tried adding __repr__ but not sure how this works: a placeholder

#from os import POSIX_FADV_NOREUSE
import torch
from torch import nn


#### LOSS FUNCTIONS: could ideally be in separate file and imported directly
class PoissonLoss_datafilter(nn.Module):
    """
    This is initialized with default behavior that requires no knowledge of the dataset:
    unit_weighting = False, meaning no external weighting of units, such as by firing rate)
        True corresponds to weighting being stored in buffer unit_weights        
    batch_weighting = 0: 'batch_size', 1: 'data_filter', 2: 'av_batch_size', -1: "unnormalized"  default 0
        Default requires no info from dataset: LL is normalized by time steps in batch, but otherwise
        could use average batch size (for consistency across epoch), or datafilter (to get correct per-neuron) 
    For example using batch size and unit_weights corresponding to reciprocal of probability of spike per bin will
    give standard LL/spk. 
    Note that default is using batch_size, and info must be oassed in using 'set_loss_weighting' to alter behavior
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.loss_name = 'poisson'
        self.loss = nn.PoissonNLLLoss(log_input=False, reduction='mean')
        self.lossNR = nn.PoissonNLLLoss(log_input=False, reduction='none')
        self.unit_weighting = False
        self.batch_weighting = 0
        self.register_buffer('unit_weights', None)  
        self.register_buffer('av_batch_size', None) 
    # END PoissonLoss_datafilter.__init__

    # def __repr__(self):
    #     s = super().__repr__()
    #     s += 'poisson'
    #     return s 

    def set_loss_weighting( self, batch_weighting=None, unit_weighting=None, unit_weights=None, av_batch_size=None ):
        """
        This changes default loss function weights to adjust for the dataset by setting two flags:
            unit_weighting: whether to weight neurons by different amounts in loss function (e.g., av spike rate)
            batch_weighting: how much to weight each batch, dividing by following quantity
                0 (default) 'batch_size': weight by length of batch
                1 'data_filters': weight each neuron individually by the amount of data
                2 'av_batch_size': weight by average batch size. Needs av_batch_size set/initialized from dataset info
                -1 'unnormalized': no weighing at all. this will implicitly increase with batch size
        """
        import numpy as np  # added as zero-weighting kluge (for np.maximum)

        if batch_weighting is not None:
            self.batch_weighting = batch_weighting 
        if unit_weighting is not None:
            self.unit_weighting = unit_weighting 

        if unit_weights is not None:
            self.unit_weights = torch.tensor(unit_weights, dtype=torch.float32)

        assert self.batch_weighting in [-1, 0, 1, 2], "LOSS: Invalid batch_weighting"
        
        if av_batch_size is not None:
            self.av_batch_size = torch.tensor(av_batch_size, dtype=torch.float32)
    # END PoissonLoss_datafilter.set_loss_weighting

    def forward(self, pred, target, data_filters=None ):        
        
        unit_weights = torch.ones( pred.shape[1], device=pred.device)
        if self.batch_weighting == 0:  # batch_size
            unit_weights /= pred.shape[0]
        elif self.batch_weighting == 1: # data_filters
            assert data_filters is not None, "LOSS: batch_weighting requires data filters"
            unit_weights = torch.reciprocal( torch.sum(data_filters, axis=0).clamp(min=1) )
        elif self.batch_weighting == 2: # average_batch_size
            unit_weights /= self.av_batch_size
        # Note can leave as 1s if unnormalized

        if self.unit_weighting:
            unit_weights *= self.unit_weights

        if data_filters is None:
            # Currently this does not apply unit_norms
            loss = self.loss(pred, target)
        else:
            loss_full = self.lossNR(pred, target)
            # divide by number of valid time points
            
            loss = torch.sum(torch.mul(unit_weights, torch.mul(loss_full, data_filters))) / len(unit_weights)
        return loss
    # END PoissonLoss_datafilter.forward

    def unit_loss(self, pred, target, data_filters=None, temporal_normalize=True ):        
        """This should be equivalent of forward, without sum over units
        Currently only true if batch_weighting = 'data_filter'"""

        if data_filters is None:
            unitloss = torch.sum(
                self.lossNR(pred, target),
                axis=0)
        else:
            loss_full = self.lossNR(pred, target)

            unit_weighting = 1.0/torch.maximum(
                torch.sum(data_filters, axis=0),
                torch.tensor(1.0, device=data_filters.device) )

            if temporal_normalize:
                unitloss = torch.mul(unit_weighting, torch.sum( torch.mul(loss_full, data_filters), axis=0) )
            else:
                unitloss = torch.sum( torch.mul(loss_full, data_filters), axis=0 )
        return unitloss
        # END PoissonLoss_datafilter.unit_loss
