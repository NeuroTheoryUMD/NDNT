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
from pytorch_lightning import LightningModule


#### LOSS FUNCTIONS: could ideally be in separate file and imported directly
class PoissonLoss_datafilter(LightningModule):
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
        #print('loss is on', self.device)

    def __repr__(self):
        s = super().__repr__()
        s += 'poisson'
        return s 

    def forward(self, pred, target, data_filters = None ):        
        
        if data_filters is None:
            loss = self.Poisson_loss(pred, target)
        else:
            loss_full = self.Poisson_lossDF(pred, target)
            loss = torch.sum(
                torch.div(
                    torch.mul(loss_full, data_filters), 
                    torch.maximum(
                        torch.sum(data_filters, axis=0), torch.tensor(1.0)) ))
            
        return loss

    def unit_loss(self, pred, target, data_filters = None ):        
        """This should be equivalent of forward, without sum over units"""

        if data_filters is None:
            unitloss = torch.sum(
                self.lossNR(pred, target),
                axis=0)
        else:
            loss_full = self.lossNR(pred, target)
            unitloss = torch.div(
                torch.mul(loss_full, data_filters), 
                torch.maximum(
                    torch.sum(data_filters, axis=0), torch.tensor(1.0)) )
            
        return unitloss
