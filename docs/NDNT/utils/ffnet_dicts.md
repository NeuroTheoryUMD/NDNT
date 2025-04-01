Module NDNT.utils.ffnet_dicts
=============================

Functions
---------

`ffnet_dict_NIM(input_dims=None, layer_sizes=None, layer_types=None, act_funcs=None, ei_layers=None, conv_widths=None, norm_list=None, reg_list=None, xstim_n='stim', ffnet_n=None, ffnet_type='normal')`
:   This creates will make a list of layer dicts corresponding to a non-convolutional NIM].
    Note that input_dims can be set to none

`ffnet_dict_external(name='external', xstim_n='stim', ffnet_n=None, input_dims=None, input_dims_reshape=None, output_dims=None)`
:   The network information passed in must simply be:
    1. The name of the network in the network dictionary that is passed into the constructor.
    2. The source of network input (external stim or otherwise).
    3. If external input, its input dims must also be specified.
    4. If the network takes input that needs to be reshaped, pass in 'input_dims_reshaped' that adds the 
    batch dimension and then reshapes before passing into external network. Note this will be the dimensionality
    that the network takes as input, so does not need to be the NDN convention [CWHT]
    5. The output dims that are passed to the rest of the network (needs to be [CWHT]

`ffnet_dict_readout(ffnet_n=None, num_cells=0, act_func='softplus', bias=True, init_mu_range=0.1, init_sigma=1, batch_sample=True, align_corners=True, gauss_type='uncorrelated', pos_constraint=False, reg_list=None)`
:   This sets up dictionary parameters for readout ffnetwork, establishing all the relevant info. Note that the shifter
    is designated as the second of two ffnet_n inputs listed, so is not separately specified.

`ffnet_params_default(xstim_n=None, ffnet_n=None, input_dims=None)`
:   This creates a ffnetwork_params object that specifies details of ffnetwork
    ffnetwork dicts have these fields:
        ffnet_type: string specifying network type (default = 'normal')
        layer_list: defaults to None, to be set in internal function
        input_dims: defaults to None, but will be set when network made, if not sooner
        xstim_n: external input (or None). Note can only be from one source
        ffnet_n: list of internal network inputs (has to be a list, or None)
        conv: [boolean] whether ffnetwork is convolutional or not, defaults to False but to be set
        
    -- Note xstim_n and ffnet_n are created/formatted here. 
    -- If xstim_n is specified, it must specify input dimensions
    -- This should set all required fields as needed (even if none)

`layer_dict(input_dims=None, num_filters=1, NLtype='relu', norm_type=0, pos_constraint=False, num_inh=0, output_norm=None, conv=False, conv_width=None, stride=None, dilation=None)`
:   input dims are [num_filters, space1, space2, num_lags]

`list_complete(fauxlist, L=None, null_val=None)`
: