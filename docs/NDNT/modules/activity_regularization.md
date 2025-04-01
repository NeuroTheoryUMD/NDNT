Module NDNT.modules.activity_regularization
===========================================

Classes
-------

`ActivityRegularization(reg_vals)`
:   

    ### Methods

    `activity(self, layer_output, alpha)`
    :   Activity regularization loss function.
        
        Args:
            layer_output (torch.Tensor): output of a layer
            alpha (float): (regularization strength)
        
        Returns:
            torch.Tensor (regularization loss)

    `nonneg(self, layer_output, alpha)`
    :   Non-negative regularization loss function.
        
        Args:
            layer_output (torch.Tensor): output of a layer
            alpha (float): (regularization strength)
        
        Returns:
            torch.Tensor (regularization loss)

    `regularize(self, layer_output)`
    :   Regularization loss function.
        
        Args:
            layer_output (torch.Tensor): output of a layer
        
        Returns:
            torch.Tensor (regularization loss)