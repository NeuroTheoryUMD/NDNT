Module NDNT.utils.create_reg_matrices
=====================================

Functions
---------

`create_localpenalty_matrix(input_dims, separable=True, spatial_global=False)`
:   Usage: Tmat = create_maxpenalty_matrix(input_dims, reg_type)
    
    Creates a matrix specifying a an L2-regularization operator of the form
    ||T*k||^2, where T is a matrix and k is a vector of parameters. Currently
    only supports second derivative/Laplacian operations
    
    Args:
        input_dims (list of ints): dimensions associated with the target input,
            in the form [num_lags, num_x_pix, num_y_pix]
        reg_type (str): specify form of the regularization matrix
            'max' | 'max_filt' | 'max_space'
    
    Returns:
        numpy array: matrix specifying the desired Tikhonov operator
    
    Notes:
        Adapted from create_Tikhonov_matrix function above.

`create_maxpenalty_matrix(input_dims, reg_type)`
:   Usage: Tmat = create_maxpenalty_matrix(input_dims, reg_type)
    
    Creates a matrix specifying a an L2-regularization operator of the form
    ||T*k||^2, where T is a matrix and k is a vector of parameters. Currently 
    only supports second derivative/Laplacian operations
    
    Args:
        input_dims (list of ints): dimensions associated with the target input, 
            in the form [num_lags, num_x_pix, num_y_pix]
        reg_type (str): specify form of the regularization matrix
            'max' | 'max_filt' | 'max_space'
    
    Returns:
        numpy array: matrix specifying the desired Tikhonov operator
    
    Notes:
        Adapted from create_Tikhonov_matrix function above.

`create_tikhonov_matrix(stim_dims, reg_type, boundary_conditions=None)`
:   Usage: Tmat = create_Tikhonov_matrix(stim_dims, reg_type, boundary_cond)
    
    Creates a matrix specifying a an L2-regularization operator of the form
    ||T*k||^2, where T is a matrix and k is a vector of parameters. Currently 
    only supports second derivative/Laplacian operations
    
    Args:
        stim_dims (list of ints): dimensions associated with the target 
            stimulus, in the form [num_lags, num_x_pix, num_y_pix]
        reg_type (str): specify form of the regularization matrix
            'd2xt' | 'd2x' | 'd2t'
        boundary_conditions (None): is a list corresponding to all dimensions
            [i.e. [False,True,True]: will be free if false, otherwise true)
            [default is [True,True,True]
            would ideally be a dictionary with each reg
            type listed; currently unused
    
    Returns:
        scipy array: matrix specifying the desired Tikhonov operator
    
    Notes:
        The method of computing sparse differencing matrices used here is 
        adapted from Bryan C. Smith's and Andrew V. Knyazev's function 
        "laplacian", available here: 
        http://www.mathworks.com/matlabcentral/fileexchange/27279-laplacian-in-1d-2d-or-3d
        Written in Matlab by James McFarland, adapted into python by Dan Butts
    
        Currently, the no-boundary condition case for all but temporal dimension alone is untested and possibly wrong
        due to the fact that it *seems* that the indexing in python flips the first and second dimensions and a
        transpose is thus necessary at the early stage. Not a problem (it seems) because boundary conditions are
        currently applied by default, which makes the edges zero....