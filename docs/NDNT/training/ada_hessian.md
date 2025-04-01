Module NDNT.training.ada_hessian
================================

Classes
-------

`AdaHessian(params, lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, hessian_power=1.0, update_each=1, n_samples=1, average_conv_kernel=False)`
:   Implements the AdaHessian algorithm from "ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning".
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 0.1)
        betas ((float, float), optional): coefficients used for computing running averages of gradient and the squared hessian trace (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
        hessian_power (float, optional): exponent of the hessian trace (default: 1.0)
        update_each (int, optional): compute the hessian trace approximation only after *this* number of steps (to save time) (default: 1)
        n_samples (int, optional): how many times to sample `z` for the approximation of the hessian trace (default: 1)

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `get_params(self)`
    :   Gets all parameters in all param_groups with gradients.
        
        Returns:
            generator: a generator that produces all parameters with gradients

    `set_hessian(self)`
    :   Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.

    `step(self, closure=None)`
    :   Performs a single optimization step.
        
        Args:
            closure (callable, optional): a closure that reevaluates the model and returns the loss (default: None)

    `zero_hessian(self)`
    :   Zeros out the accumalated hessian traces.