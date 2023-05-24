import torch


class ActivityRegularization:
    def __init__(self, reg_vals):
        self.reg_vals = reg_vals
    
    def activity(self, layer_output, alpha):
        """
        :param layer_output: torch.Tensor (output of a layer)
        :param alpha: float (regularization strength)
        :return: torch.Tensor (regularization loss)
        """
        # sum over the squared subunits, then average over the batch
        loss = alpha * torch.mean(torch.sum(layer_output**2, axis=1), axis=0)
        return loss


    def nonneg(self, layer_output, alpha):
        """
        :param layer_output: torch.Tensor (output of a layer)
        :param alpha: float (regularization strength)
        :return: torch.Tensor (regularization loss)
        """
        # sum over the relu of the negative subunits, then average over the batch
        return alpha * torch.mean(torch.sum(torch.relu(-layer_output), axis=1), axis=0)


    def regularize(self, layer_output):
        """
        :param layer_output: torch.Tensor (output of a layer)
        :return: torch.Tensor (regularization loss)
        """
        activity_loss = 0.0
        if 'activity' in self.reg_vals:
            activity_loss = self.activity(layer_output, alpha=self.reg_vals['activity'])

        nonneg_loss = 0.0
        if 'nonneg' in self.reg_vals:
            nonneg_loss = self.nonneg(layer_output, alpha=self.reg_vals['nonneg'])

        return activity_loss + nonneg_loss
