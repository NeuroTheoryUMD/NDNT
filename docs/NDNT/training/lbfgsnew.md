Module NDNT.training.lbfgsnew
=============================

Classes
-------

`LBFGSNew(params, lr=1, max_iter=10, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=7, line_search_fn=False, batch_mode=False)`
:   Implements L-BFGS algorithm.
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.
    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.
    Args:
        lr (float): learning rate (fallback value when line search fails. not really needed) (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 10)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 7).
        line_search_fn: if True, use cubic interpolation to findstep size, if False: fixed step size
        batch_mode: True for stochastic version (default False)
        Example usage for full batch mode:
          optimizer = LBFGSNew(model.parameters(), history_size=7, max_iter=100, line_search_fn=True, batch_mode=False)
        Example usage for batch mode (stochastic):
          optimizer = LBFGSNew(net.parameters(), history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)
          Note: when using a closure(), only do backward() after checking the gradient is available,
          Eg: 
            def closure():
             optimizer.zero_grad()
             outputs=net(inputs)
             loss=criterion(outputs,labels)
             if loss.requires_grad:
               loss.backward()
             return loss

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure)`
    :   Performs a single optimization step.
        
        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.