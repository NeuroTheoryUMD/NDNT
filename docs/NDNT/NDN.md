Module NDNT.NDN
===============

Classes
-------

`NDN(ffnet_list=None, layer_list=None, external_modules=None, loss_type='poisson', ffnet_out=None, optimizer_params=None, model_name=None, seed=None, working_dir='./checkpoints')`
:   Initializes an instance of the NDN (Neural Deep Network) class.
    
    Args:
        ffnet_list (list): A list of FFnetwork objects. If None, a single ffnet specified by layer_list will be used.
        layer_list (list): A list specifying the layers of the FFnetwork. Only used if ffnet_list is None.
        external_modules (list): A list of external modules to be used in the FFnetwork.
        loss_type (str): The type of loss function to be used. Default is 'poisson'.
        ffnet_out (int or list): The index(es) of the output layer(s) of the FFnetwork(s). If None, the last layer will be used.
        optimizer_params (dict): Parameters for the optimizer. If None, default parameters will be used.
        model_name (str): The name of the model. If None, a default name will be assigned.
        seed (int): The random seed to be used for reproducibility.
        working_dir (str): The directory to save checkpoints and other files.
    
    Returns:
        None
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `load_model(path)`
    :   Load the model from an a zip file (with extension .ndn) containing a json file with the model parameters
        and a .ckpt file with the state_dict.
        
        Args:
            path: The name of the zip file to load.
        
        Returns:
            NDN: The loaded model.

    `load_model_ckpt(checkpoint_path=None, model_name=None, version=None, filename=None)`
    :   Load a model from disk. Note that if model not named, the full path must be put into checkpoint path.
        If filename 
        
        Args:
            checkpoint_path: The path to the directory containing model checkpoints
            model_name: The name of the model (from model.model_name)
            version: The checkpoint version (default: best)
            filename: Enter if want to override 'best model' and load specific file in specified path
        
        Returns:
            model: The loaded model

    `load_model_pkl(filename=None, pt=False)`
    :   Load a pickled model from disk.
        
        Args:
            filename: The path and filename.
            pt: Whether to use torch.load instead of dill.
        
        Returns:
            model: The loaded model on CPU.

    ### Instance variables

    `block_sample`
    :

    `device`
    :

    ### Methods

    `assemble_ffnetworks(self, ffnet_list, external_nets=None)`
    :   This function takes a list of ffnetworks and puts them together 
        in order. This has to do two steps for each ffnetwork: 
        
        1. Plug in the inputs to each ffnetwork as specified
        2. Builds the ff-network with the input
        
        This returns the 'network', which is (currently) a module with a 
        'forward' and 'reg_loss' function specified.
        
        When multiple ffnet inputs are concatenated, it will always happen in the first
        (filter) dimension, so all other dimensions must match
        
        Args:
            ffnet_list (list): A list of ffnetworks to be assembled.
            external_nets (optional): External networks to be passed into the 'external' type ffnetworks.
        
        Returns:
            networks (nn.ModuleList): A module list containing the assembled ff-networks.
        
        Raises:
            AssertionError: If ffnet_list is not a list.

    `change_loss(self, new_loss_type, dataset=None)`
    :   Change the loss function for the model.
        
        Swaps in new loss module for model.
        Include dataset if wish to initialize loss (which happens during fit).
        
        Args:
            new_loss_type: The new loss type.
            dataset: The dataset to use for initializing the loss.
        
        Returns:
            None
        
        Notes:
            This method swaps in a new loss module for the model.
            If a dataset is provided, the loss will be initialized during the fit.

    `compute_average_responses(self, dataset, data_inds=None)`
    :   Computes the average responses for the specified dataset.
        
        Args:
            dataset: The dataset to use for computing the average responses.
            data_inds: The indices of the data in the dataset.
        
        Returns:
            ndarray: The average responses.

    `compute_network_outputs(self, Xs)`
    :   Computes the network outputs for the given input data.
        
        Args:
            Xs (list): The input data.
        
        Returns:
            tuple: A tuple containing the network inputs and network outputs.
        
        Raises:
            AssertionError: If no networks are defined in this NDN.
        
        Note:
            This method currently saves only the network outputs and does not return the network inputs.

    `compute_reg_loss(self)`
    :   Computes the regularization loss for the NDNT model.
        
        Args:
            None
        
        Returns:
            The total regularization loss.

    `eval_models(self, data, data_inds=None, bits=False, null_adjusted=False, speckledXV=False, train_val=1, batch_size=1000, num_workers=0, device=None, **kwargs)`
    :   Evaluate the neural network models on the given data.
        
        get null-adjusted log likelihood (if null_adjusted = True)
        bits=True will return in units of bits/spike
        
        Note that data will be assumed to be a dataset, and data_inds will have to be specified batches
        from dataset.__get_item__()
        
        Args:
            data (dict or Dataset): The input data to evaluate the models on. If a dictionary is provided, it is assumed to be a single sample. If a Dataset object is provided, it is assumed to contain multiple samples.
            data_inds (list, optional): The indices of the data samples to evaluate. Only applicable if `data` is a Dataset object. Defaults to None.
            bits (bool, optional): If True, the result will be returned in units of bits/spike. Defaults to False.
            null_adjusted (bool, optional): If True, the null-adjusted log likelihood will be calculated. Defaults to False.
            speckledXV (bool, optional): If True, speckled cross-validation will be used. Defaults to False.
            train_val (int, optional): The train/validation split ratio. Only applicable if `speckledXV` is True. Defaults to 1.
            batch_size (int, optional): The batch size for data loading. Defaults to 1000.
            num_workers (int, optional): The number of worker processes for data loading. Defaults to 0.
            device (torch.device, optional): The device to perform the evaluation on. If None, the device of the model will be used. Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            numpy.ndarray: The evaluated log likelihood values for each neuron.
        
        Note:
            If `data` is a dictionary, it is assumed to contain the following keys:
                'robs': The observed responses.
                'Mval': The validation mask.
                'dfs': The data filters.
            If `data` is a Dataset object, it is assumed to have the following attributes:
                'robs': The observed responses.
                'dfs': The data filters.
        
        Raises:
            AssertionError: If `data_inds` is not None when `data` is a dictionary.

    `fit(self, dataset, train_inds=None, val_inds=None, speckledXV=False, verbose=2, seed=None, save_dir: str = None, version: int = None, optimizer=None, scheduler=None, batch_size=None, force_dict_training=False, block_sample=None, reuse_trainer=False, device=None, save_epochs=False, **kwargs)`
    :   Trains the model using the specified dataset.
        
        Args:
            dataset (Dataset): The dataset to use for training and validation.
            train_inds (list): The indices of the training data.
            val_inds (list): The indices of the validation data.
            speckledXV (bool): Whether to use speckled cross-validation. Default is False.
            verbose (int): level of text info while running (0=None, 1=epoch level, 2=batch and extra info): default 2
            seed (int): The seed to use for reproducibility.
            save_dir (str): The directory to save the model checkpoints.
            version (int): The version of the trainer.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to use for adjusting the learning rate during training.
            batch_size (int): The batch size to use.
            force_dict_training (bool): Whether to force dictionary-based training. Default is False.
            block_sample (bool): Whether to use block sampling. Default is None.
            reuse_trainer (bool): Whether to reuse the trainer. Default is False.
            device (str or torch.device): The device to use for training. Default is None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None
        
        Raises:
            AssertionError: If batch_size is not specified.
        
        Notes:
            The fit method is the main training loop for the model.
            Steps:
            1. Get a trainer and dataloaders
            2. Prepare regularizers
            3. Run the main fit loop from the trainer, checkpoint, and save model

    `fit_dl(self, train_ds, val_ds, seed=None, save_dir: str = None, version: int = None, name: str = None, optimizer=None, scheduler=None, batch_size=None, num_workers=0, force_dict_training=False, device=None, save_epochs=False, **kwargs)`
    :   Fits the model using deep learning training.
        
        Args:
            train_ds (Dataset): The training dataset.
            val_ds (Dataset): The validation dataset.
            seed (int, optional): The seed for data and optimization. Defaults to None.
            save_dir (str, optional): The directory to save the model. Defaults to None.
            version (int, optional): The version of the model. Defaults to None.
            name (str, optional): The name of the model. Defaults to None.
            optimizer (Optimizer, optional): The optimizer for training. Defaults to None.
            scheduler (Scheduler, optional): The scheduler for training. Defaults to None.
            batch_size (int, optional): The batch size for training. Defaults to None.
            num_workers (int, optional): The number of workers for data loading. Defaults to 0.
            force_dict_training (bool, optional): Whether to force dictionary-based training instead of using data loaders for LBFGS. Defaults to False.
            device (str, optional): The device to use for training. Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None
        
        Raises:
            AssertionError: If batch_size is not provided.
        
        Notes:
            This is the main training loop.
            Steps:
            1. Get a trainer and data loaders.
            2. Prepare regularizers.
            3. Run the main fit loop from the trainer, checkpoint, and save the model.

    `forward(self, Xs) ‑> Callable[..., Any]`
    :   Applies the forward pass of each network in sequential order.
        
        Args:
            Xs (list): List of input data.
        
        Returns:
            ndarray: Output of the forward pass.
        
        Notes:
            The tricky thing is concatenating multiple-input dimensions together correctly.
            Note that the external inputs are actually in principle a list of inputs.

    `get_activations(self, sample, ffnet_target=0, layer_target=0, NL=False)`
    :   Returns the inputs and outputs of a specified ffnet and layer
        
        Args:
            sample: dictionary of sample data from a dataset
            ffnet_target: which network to target (default: 0)
            layer_target: which layer to target (default: 0)
            NL: get activations using the nonlinearity as the module target
        
        Returns:
            activations dict
            with keys:
                'input' : input to layer
                'output' : output of layer

    `get_dataloaders(self, dataset, train_inds=None, val_inds=None, batch_size=10, num_workers=0, is_multiexp=False, full_batch=False, pin_memory=False, data_seed=None, verbose=0, **kwargs)`
    :   Creates dataloaders for training and validation.
        
        Args:
            dataset (Dataset): The dataset to use for training and validation.
            train_inds (list): The indices of the training data.
            val_inds (list): The indices of the validation data.
            batch_size (int): The batch size to use.
            num_workers (int): The number of workers to use for data loading.
            is_multiexp (bool): Whether the dataset is a multi-experiment dataset.
            full_batch (bool): Whether to use the full batch for training.
            pin_memory (bool): Whether to pin memory for faster data loading.
            data_seed (int): The seed to use for the data loader.
            verbose (int): whether to output information during fit (above 1) <== not currently used
            **kwargs: Additional keyword arguments.
        
        Returns:
            train_dl (DataLoader): The training data loader.
            valid_dl (DataLoader): The validation data loader.

    `get_network_info(self, abbrev=False)`
    :   Prints out a decreiption of the model structure.

    `get_null_adjusted_ll(self, sample, bits=False)`
    :   Get the null-adjusted log likelihood.
        
        Args:
            sample: The sample data from a dataset.
            bits: Whether to return the result in units of bits/spike.
        
        Returns:
            float: The null-adjusted log likelihood.

    `get_optimizer(self, optimizer_type='AdamW', learning_rate=0.001, weight_decay=0.01, amsgrad=False, betas=(0.9, 0.999), momentum=0.9, max_iter=10, history_size=4, tolerance_change=0.001, tolerance_grad=0.0001, line_search_fn=None, **kwargs)`
    :   Returns an optimizer object.
        
        Args:
            optimizer_type (str): The type of optimizer to use. Default is 'AdamW'.
            learning_rate (float): The learning rate to use. Default is 0.001.
            weight_decay (float): The weight decay to use. Default is 0.01.
            amsgrad (bool): Whether to use the AMSGrad variant of Adam. Default is False.
            betas (tuple): The beta values to use for the optimizer. Default is (0.9, 0.999).
            max_iter (int): The maximum number of iterations for the LBFGS optimizer. Default is 10.
            history_size (int): The history size to use for the LBFGS optimizer. Default is 4.
        
        Returns:
            optimizer (torch.optim.Optimizer): The optimizer object.

    `get_readout_locations(self)`
    :   Get the readout locations and sigmas.
        
        Returns:
            list: The readout locations and sigmas.
        
        Notes:
            This method currently returns a list of readout locations and sigmas set in the readout network.

    `get_trainer(self, version=None, save_dir='./checkpoints', name='jnkname', optimizer=None, scheduler=None, device=None, optimizer_type='AdamW', early_stopping=False, early_stopping_patience=5, early_stopping_delta=0.0, optimize_graph=False, save_epochs=False, verbose=1, **kwargs)`
    :   Returns a trainer object.
        
        Args:
            version (str): The version of the trainer.
            save_dir (str): The directory to save the trainer checkpoints.
            name (str): The name of the trainer.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to use for adjusting the learning rate during training.
            device (str or torch.device): The device to use for training. If not specified, it will default to 'cuda:0' if available, otherwise 'cpu'.
            optimizer_type (str): The type of optimizer to use. Default is 'AdamW'.
            early_stopping (bool): Whether to use early stopping during training. Default is False.
            early_stopping_patience (int): The number of epochs to wait for improvement before stopping early. Default is 5.
            early_stopping_delta (float): The minimum change in the monitored metric to be considered as improvement. Default is 0.0.
            optimize_graph (bool): Whether to optimize the computation graph during training. Default is False.
            verbose (bool): whether trainer should output messages, passed from fit options (default 0)
            **kwargs: Additional keyword arguments to be passed to the trainer.
        
        Returns:
            trainer (Trainer): The trainer object.

    `get_weights(self, ffnet_target=0, **kwargs)`
    :   Get the weights for the specified feedforward network.
        
        Passed down to layer call, with optional arguments conveyed.
        
        Args:
            ffnet_target: The target feedforward network.
            **kwargs: Additional keyword arguments.
        
        Returns:
            list: The weights for the specified feedforward network.
        
        Notes:
            This method is passed down to the layer call with optional arguments conveyed.

    `info(self, expand=False)`
    :

    `initialize_biases(self, dataset, data_inds=None, ffnet_target=-1, layer_target=-1)`
    :   Initializes the biases for the specified dataset.
        
        Args:
            dataset: The dataset to use for initializing the biases.
            data_inds: The indices of the data in the dataset.
            ffnet_target: The target feedforward network.
            layer_target: The target layer.
        
        Returns:
            None
        
        Notes:
            This method initializes the biases for the specified feedforward network and layer using the average responses.

    `initialize_loss(self, dataset=None, batch_size=None, data_inds=None, batch_weighting=None, unit_weighting=None)`
    :   Interacts with loss_module to set loss flags and/or pass in dataset information.
        
        Args:
            dataset (optional): The dataset to be used for computing average batch size and unit weights.
            batch_size (optional): The batch size to be used for computing average batch size.
            data_inds (optional): The indices of the data in the dataset to be used.
            batch_weighting (optional): The batch weighting value to be set. Must be one of [-1, 0, 1, 2].
            unit_weighting (optional): The unit weighting value to be set.
        
        Returns:
            None
        
        Notes:
            Without a dataset, this method will only set the flags 'batch_weighting' and 'unit_weighting'.
            With a dataset, this method will set 'av_batch_size' and 'unit_weights' based on the average rate.

    `list_parameters(self, ffnet_target=None, layer_target=None)`
    :   List the parameters for the specified feedforward network and layer.
        
        Args:
            ffnet_target: The target feedforward network.
            layer_target: The target layer.
        
        Returns:
            None

    `model_string(self)`
    :   Automated way to name model, based on NDN network structure, num outputs, and number of layers.
        Format is NDN101_S3_R1_N1_A1 would be an NDN with 101 outputs, 3-layer scaffold followed by readout, 
        normal (drift) layer, and add (comb) layer
        
        Returns:
            str: The model name.

    `passive_readout(self)`
    :   Applies the passive readout function to the readout layer.
        
        Returns:
            None
        
        Notes:
            This method finds the readout layer and applies its passive_readout function.

    `plot_filters(self, ffnet_target=0, **kwargs)`
    :   Plot the filters for the specified feedforward network.
        
        Args:
            ffnet_target: The target feedforward network.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None

    `predictions(self, data, data_inds=None, block_inds=None, batch_size=None, num_lags=0, ffnet_target=None, device=None)`
    :   Generate predictions for the model for a dataset. Note that will need to send to device if needed, and enter 
        batch_size information. 
        
        Args:
            data: The dataset.
            data_inds: The indices to use from the dataset.
            block_inds: The blocks to use from the dataset (if block_sample).
            batch_size: The batch size to use for processing. Reduce if running out of memory.
            num_lags: The number of lags to use for prediction.
            ffnet_target: The index of the feedforward network to use for prediction. Defaults to None.
            device: The device to use for prediction
        
        Returns:
            torch.Tensor: The predictions array (detached, on cpu)

    `prepare_regularization(self)`
    :   Prepares the regularization for the model.
        
        Returns:
            None

    `save_model(self, path, ffnet_list=None, ffnet_out=None, loss_type='poisson', model_name=None, working_dir='./checkpoints')`
    :   Save the model as a zip file (with extension .ndn) containing a json file with the model parameters
        and a .ckpt file with the state_dict.
        
        Args:
            path: The name of the zip file to save.
            ffnet_list: The list of feedforward networks. (if this is set, it uses the ffnet_list, ffnet_out, loss_type, model_name, and working_dir arguments)
            ffnet_out: The output layer of the feedforward network.
            loss_type: The loss type.
            model_name: The name of the model.
            working_dir: The working directory.
        
        Returns:
            None

    `save_model_chk(self, filename=None, alt_dirname=None)`
    :   Models will be saved using dill/pickle in the directory above the version
        directories, which happen to be under the model-name itself. This assumes the
        current save-directory (notebook specific) and the model name
        
        Args:
            filename: The name of the file to save.
            alt_dirname: The alternate directory to save the file.
        
        Returns:
            None

    `save_model_pkl(self, filename=None, pt=False)`
    :   Models will be saved using dill/pickle in as the filename, which can contain
        the directory information. Will be put in the CPU first
        
        Args:
            filename: The name of the file to save.
            pt: Whether to use torch.save instead of dill.
        
        Returns:
            None

    `set_parameters(self, ffnet_target=None, layer_target=None, name=None, val=None)`
    :   Set the parameters for the specified feedforward network and layer.
        
        Args:
            ffnet_target: The target feedforward network.
            layer_target: The target layer.
            name: The name of the parameter.
            val: The value to set the parameter to.
        
        Returns:
            None
        
        Notes:
            This method sets the parameters for the specified feedforward network and layer.

    `set_reg_val(self, reg_type=None, reg_val=None, ffnet_target=None, layer_target=None)`
    :   Set reg_values for listed network and layer targets (default 0,0).
        
        Args:
            reg_type: The regularization type.
            reg_val: The regularization value.
            ffnet_target: The target feedforward network.
            layer_target: The target layer.
        
        Returns:
            None
        
        Notes:
            This method sets the regularization values for the specified feedforward network and layer.

    `training_step(self, batch, batch_idx=None)`
    :   Performs a single training step.
        
        Args:
            batch (dict): A dictionary containing the input data batch.
            batch_idx (int, optional): The index of the current batch. Defaults to None.
        
        Returns:
            dict: A dictionary containing the loss values for training, total loss, and regularization loss.

    `update_ffnet_list(self)`
    :   the ffnet_list builds the NDN, but ideally holds a record of the regularization (a dictionary)
        since dictionaries are not saveable in a checkpoint. So, this function pulls the dictionaries
        from each reg_module back into the ffnet_list
        
        Args:
            None
        
        Returns:
            None, but updates self.ffnet_list

    `validation_step(self, batch, batch_idx=None)`
    :   Performs a validation step for the model.
        
        Args:
            batch (dict): A dictionary containing the input batch data.
            batch_idx (int, optional): The index of the current batch. Defaults to None.
        
        Returns:
            dict: A dictionary containing the computed losses for the validation step.