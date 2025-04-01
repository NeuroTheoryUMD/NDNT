Module NDNT.training.lbfgs
==========================

Functions
---------

`is_legal(v)`
:   Checks that tensor is not NaN or Inf.
    
    Args:
        v (tensor): tensor to be checked

`polyinterp(points, x_min_bound=None, x_max_bound=None, plot=False)`
:   Gives the minimizer and minimum of the interpolating polynomial over given points
    based on function and derivative information. Defaults to bisection if no critical
    points are valid.
    Based on polyinterp.m Matlab function in minFunc by Mark Schmidt with some slight
    modifications.
    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 12/6/18.
    
    Args:
        points (nparray): two-dimensional array with each point of form [x f g]
        x_min_bound (float): minimum value that brackets minimum (default: minimum of points)
        x_max_bound (float): maximum value that brackets minimum (default: maximum of points)
        plot (bool): plot interpolating polynomial
    
    Returns:
        x_sol (float): minimizer of interpolating polynomial
        F_min (float): minimum of interpolating polynomial
    
    Note:
        Set f or g to np.nan if they are unknown

Classes
-------

`FullBatchLBFGS(params, lr=1, history_size=10, line_search='Wolfe', dtype=torch.float32, debug=False)`
:   Implements full-batch or deterministic L-BFGS algorithm. Compatible with
    Powell damping. Can be used when evaluating a deterministic function and
    gradient. Wraps the LBFGS optimizer. Performs the two-loop recursion,
    updating, and curvature updating in a single step.
    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 11/15/18.
    Warnings:
      . Does not support per-parameter options and parameter groups.
      . All parameters have to be on a single device.
      
    Args:
        lr (float): steplength or learning rate (default: 1)
        history_size (int): update history size (default: 10)
        line_search (str): designates line search to use (default: 'Wolfe')
            Options:
                'None': uses steplength designated in algorithm
                'Armijo': uses Armijo backtracking line search
                'Wolfe': uses Armijo-Wolfe bracketing line search
        dtype: data type (default: torch.float)
        debug (bool): debugging mode

    ### Ancestors (in MRO)

    * NDNT.training.lbfgs.LBFGS
    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, options=None)`
    :   Performs a single optimization step.
        Args:
            options (dict): contains options for performing line search (default: None)
            
        General Options:
            'eps' (float): constant for curvature pair rejection or damping (default: 1e-2)
            'damping' (bool): flag for using Powell damping (default: False)
        
        Options for Armijo backtracking line search:
            'closure' (callable): reevaluates model and returns function value
            'current_loss' (tensor): objective value at current iterate (default: F(x_k))
            'gtd' (tensor): inner product g_Ok'd in line search (default: g_Ok'd)
            'eta' (tensor): factor for decreasing steplength > 0 (default: 2)
            'c1' (tensor): sufficient decrease constant in (0, 1) (default: 1e-4)
            'max_ls' (int): maximum number of line search steps permitted (default: 10)
            'interpolate' (bool): flag for using interpolation (default: True)
            'inplace' (bool): flag for inplace operations (default: True)
            'ls_debug' (bool): debugging mode for line search
        
        Options for Wolfe line search:
            'closure' (callable): reevaluates model and returns function value
            'current_loss' (tensor): objective value at current iterate (default: F(x_k))
            'gtd' (tensor): inner product g_Ok'd in line search (default: g_Ok'd)
            'eta' (float): factor for extrapolation (default: 2)
            'c1' (float): sufficient decrease constant in (0, 1) (default: 1e-4)
            'c2' (float): curvature condition constant in (0, 1) (default: 0.9)
            'max_ls' (int): maximum number of line search steps permitted (default: 10)
            'interpolate' (bool): flag for using interpolation (default: True)
            'inplace' (bool): flag for inplace operations (default: True)
            'ls_debug' (bool): debugging mode for line search
        
        Outputs (depends on line search):
          . No line search:
                t (float): steplength
          . Armijo backtracking line search:
                F_new (tensor): loss function at new iterate
                t (tensor): final steplength
                ls_step (int): number of backtracks
                closure_eval (int): number of closure evaluations
                desc_dir (bool): descent direction flag
                    True: p_k is descent direction with respect to the line search
                    function
                    False: p_k is not a descent direction with respect to the line
                    search function
                fail (bool): failure flag
                    True: line search reached maximum number of iterations, failed
                    False: line search succeeded
          . Wolfe line search:
                F_new (tensor): loss function at new iterate
                g_new (tensor): gradient at new iterate
                t (float): final steplength
                ls_step (int): number of backtracks
                closure_eval (int): number of closure evaluations
                grad_eval (int): number of gradient evaluations
                desc_dir (bool): descent direction flag
                    True: p_k is descent direction with respect to the line search
                    function
                    False: p_k is not a descent direction with respect to the line
                    search function
                fail (bool): failure flag
                    True: line search reached maximum number of iterations, failed
                    False: line search succeeded
                    
        Notes:
          . If encountering line search failure in the deterministic setting, one
            should try increasing the maximum number of line search steps max_ls.

`LBFGS(params, lr=1.0, history_size=10, line_search='Wolfe', dtype=torch.float32, debug=False)`
:   Implements the L-BFGS algorithm. Compatible with multi-batch and full-overlap
    L-BFGS implementations and (stochastic) Powell damping. Partly based on the 
    original L-BFGS implementation in PyTorch, Mark Schmidt's minFunc MATLAB code, 
    and Michael Overton's weak Wolfe line search MATLAB code.
    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 10/20/20.
    
    Warnings:
      . Does not support per-parameter options and parameter groups.
      . All parameters have to be on a single device.
    
    Args:
        lr (float): steplength or learning rate (default: 1)
        history_size (int): update history size (default: 10)
        line_search (str): designates line search to use (default: 'Wolfe')
            Options:
                'None': uses steplength designated in algorithm
                'Armijo': uses Armijo backtracking line search
                'Wolfe': uses Armijo-Wolfe bracketing line search
        dtype: data type (default: torch.float)
        debug (bool): debugging mode
    
    References:
    [1] Berahas, Albert S., Jorge Nocedal, and Martin Takác. "A Multi-Batch L-BFGS 
        Method for Machine Learning." Advances in Neural Information Processing 
        Systems. 2016.
    [2] Bollapragada, Raghu, et al. "A Progressive Batching L-BFGS Method for Machine 
        Learning." International Conference on Machine Learning. 2018.
    [3] Lewis, Adrian S., and Michael L. Overton. "Nonsmooth Optimization via Quasi-Newton
        Methods." Mathematical Programming 141.1-2 (2013): 135-163.
    [4] Liu, Dong C., and Jorge Nocedal. "On the Limited Memory BFGS Method for 
        Large Scale Optimization." Mathematical Programming 45.1-3 (1989): 503-528.
    [5] Nocedal, Jorge. "Updating Quasi-Newton Matrices With Limited Storage." 
        Mathematics of Computation 35.151 (1980): 773-782.
    [6] Nocedal, Jorge, and Stephen J. Wright. "Numerical Optimization." Springer New York,
        2006.
    [7] Schmidt, Mark. "minFunc: Unconstrained Differentiable Multivariate Optimization 
        in Matlab." Software available at http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html 
        (2005).
    [8] Schraudolph, Nicol N., Jin Yu, and Simon Günter. "A Stochastic Quasi-Newton 
        Method for Online Convex Optimization." Artificial Intelligence and Statistics. 
        2007.
    [9] Wang, Xiao, et al. "Stochastic Quasi-Newton Methods for Nonconvex Stochastic 
        Optimization." SIAM Journal on Optimization 27.2 (2017): 927-956.

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Descendants

    * NDNT.training.lbfgs.FullBatchLBFGS

    ### Methods

    `curvature_update(self, flat_grad, eps=0.01, damping=False)`
    :   Performs curvature update.
        
        Args:
            flat_grad (tensor): 1-D tensor of flattened gradient for computing 
                gradient difference with previously stored gradient
            eps (float): constant for curvature pair rejection or damping (default: 1e-2)
            damping (bool): flag for using Powell damping (default: False)

    `line_search(self, line_search)`
    :   Switches line search option.
        
        Args:
            line_search (str): designates line search to use
                Options:
                    'None': uses steplength designated in algorithm
                    'Armijo': uses Armijo backtracking line search
                    'Wolfe': uses Armijo-Wolfe bracketing line search

    `step(self, p_k, g_Ok, g_Sk=None, options={})`
    :   Performs a single optimization step (parameter update).
        
        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        
        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.

    `two_loop_recursion(self, vec)`
    :   Performs two-loop recursion on given vector to obtain Hv.
        
        Args:
            vec (tensor): 1-D tensor to apply two-loop recursion to
        
        Output:
            r (tensor): matrix-vector product Hv