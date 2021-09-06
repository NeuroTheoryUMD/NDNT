### NDNLosses.py
# Classes corresponding to each loss function 
# Note currently a Lightning module
# All classes should have 
#    name
#    forward: returns loss within optimizer
#    unit_loss: returns loss for each output unit (without reduction)
#    Also tried adding __repr__ but not sure how this works: a placeholder

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
        self.unit_normalization = False
        self.unit_norms = None

        #print('loss is on', self.device)

    # def __repr__(self):
    #     s = super().__repr__()
    #     s += 'poisson'
    #     return s 

    def set_unit_normalization( self, cell_norms ):
        """This is a multiplier (weighting) of each neuron in the loss: for Poisson loss function,
        it will be the average firing rate over the full dataset"""
        self.unit_normalization = True
        self.unit_norms = cell_norms

    def forward(self, pred, target, data_filters = None ):        
        
        if data_filters is None:
            # Currently this does not apply unit_norms
            loss = self.Poisson_loss(pred, target)
        else:
            loss_full = self.lossNR(pred, target)
            # divide by number of valid time points
            unit_weighting = 1.0/torch.maximum(
                torch.sum(data_filters, axis=0),
                torch.tensor(1.0, device=data_filters.device) )
            # in place of 1.0? torch.tensor(1.0, device=data_filters.device)
            
            if self.unit_normalization:
                unit_weighting *= self.unit_norms
            
            loss = torch.sum(torch.mul(unit_weighting, torch.mul(loss_full, data_filters)))
            
        return loss

    def unit_loss(self, pred, target, data_filters = None ):        
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

            #unitloss = torch.div(
            #    torch.mul(loss_full, data_filters), 
            #    torch.maximum(
            #        torch.sum(data_filters, axis=0), torch.tensor(1.0, device=data_filters.device)) )
            unitloss = torch.mul(unit_weighting, torch.mul(loss_full, data_filters))

        return unitloss
