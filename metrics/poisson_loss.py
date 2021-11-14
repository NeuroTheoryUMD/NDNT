### NDNLosses.py
# Classes corresponding to each loss function 
# Note currently a Lightning module
# All classes should have 
#    name
#    forward: returns loss within optimizer
#    unit_loss: returns loss for each output unit (without reduction)
#    Also tried adding __repr__ but not sure how this works: a placeholder

from os import POSIX_FADV_NOREUSE
import torch
from torch import nn


#### LOSS FUNCTIONS: could ideally be in separate file and imported directly
class PoissonLoss_datafilter(nn.Module):
    """The standard way would be to weight a neurons contribution in a given batch by the number of spikes it happens
    to fire in that batch, relative to other neurons. But this means that the fit overall will be dominated by high-
    firing rate neurons. But also you cannot calc LL/spk within a batch because the number of spikes are too variable
    and lead to major stability problem. My solution in the NDN was to use a a *Poisson norm*. For now I'll do the
    standard thing though."""

    def __init__(self, **kwargs):
        super().__init__()

        self.loss_name = 'poisson'
        self.loss = nn.PoissonNLLLoss(log_input=False, reduction='mean')
        self.lossNR = nn.PoissonNLLLoss(log_input=False, reduction='none')
        self.unit_normalization = True
        self.register_buffer("unit_norms", None)  # this is explicitly the wrong size
    # END PoissonLoss_datafilter.__init__

    # def __repr__(self):
    #     s = super().__repr__()
    #     s += 'poisson'
    #     return s 

    def set_unit_normalization( self, cell_weighting ):
        """In this case (this loss function), it takes inputs of the average probability of spike per bin
        and will divide by this, unless zero. This can take other cell_weighting as well, but note that it
        will divide by the cell weighting"""
        self.unit_normalization = True
        self.unit_norms = torch.divide( torch.tensor(1.0), cell_weighting )
        # Note: this currently doesn't deal with neurons that have no spikes: will lead to problems 
    # END PoissonLoss_datafilter.set_unit_normalization

    def forward(self, pred, target, data_filters = None ):        
        
        if data_filters is None:
            # Currently this does not apply unit_norms
            loss = self.loss(pred, target)
        else:
            loss_full = self.lossNR(pred, target)
            # divide by number of valid time points
            unit_weighting = torch.reciprocal( torch.sum(data_filters, axis=0).clamp(min=1) )

            #unit_weighting = torch.tensor(1.0, device=data_filters.device) / \
            #    torch.maximum(
            #        torch.sum(data_filters, axis=0),
            #        torch.tensor(1.0, device=data_filters.device) )
            # in place of 1.0? torch.tensor(1.0, device=data_filters.device)
            
            if self.unit_normalization:
                unit_weighting *= self.unit_norms
            
            loss = torch.sum(torch.mul(unit_weighting, torch.mul(loss_full, data_filters))) / len(unit_weighting)
            
        return loss
    # END PoissonLoss_datafilter.forward

    def unit_loss(self, pred, target, data_filters=None, temporal_normalize=True ):        
        """This should be equivalent of forward, without sum over units"""

        if data_filters is None:
            unitloss = torch.sum(
                self.lossNR(pred, target),
                axis=0)
        else:
            loss_full = self.lossNR(pred, target)

            unit_weighting = 1.0/torch.maximum(
                torch.sum(data_filters, axis=0),
                torch.tensor(1.0, device=data_filters.device) )
            if self.unit_normalization:
                unit_weighting *= self.unit_norms

            if temporal_normalize:
                unitloss = torch.mul(unit_weighting, torch.sum( torch.mul(loss_full, data_filters), axis=0) )
            else:
                unitloss = torch.sum( torch.mul(loss_full, data_filters), axis=0 )

        return unitloss
