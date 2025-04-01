Module NDNT.modules.experiment_sampler
======================================

Functions
---------

`construct_exp_to_time(dataset, indices=None) ‑> dict`
:   Construct a dictionary that maps experiment indices to timepoints.
    
    Args:
        dataset (Dataset): dataset to sample from
        indices (list): indices to filter by

Classes
-------

`ExperimentBatchGenerator(timepoints: list, batch_size: int, shuffle: bool, random_seed: int = None)`
:   Generator that returns batches of timepoints for a single experiment.
    
    Args:
        timepoints (list): list of timepoints to sample from
        batch_size (int): size of the batch to sample
        shuffle (bool): whether to shuffle the batches
        random_seed (int): random seed to use for shuffling

    ### Methods

    `next(self) ‑> list`
    :   Next method of the generator.
        
        Returns:
            the next batch

`ExperimentBatchIterator(exp_to_time, exp_batch_sizes, shuffle: bool, random_seed: int = None)`
:   Iterator that returns batches of timepoints across experiments.
    
    Args:
        exp_to_time (dict): map from experiment indices to timepoints
        exp_batch_sizes (list): list of batch sizes for each experiment
        shuffle (bool): whether to shuffle the batches
        random_seed (int): random seed to use for shuffling

    ### Ancestors (in MRO)

    * collections.abc.Iterator
    * collections.abc.Iterable
    * typing.Generic

`ExperimentSampler(dataset: torch.utils.data.dataset.Dataset, batch_size: int, indices: list = None, shuffle: bool = True, random_seed: int = None, verbose=False)`
:   Samples elements across experiments.
    
    Args:
        dataset (Dataset): dataset to sample from
        batch_size (int): size of the batch to sample
        indices (list): indices to sample from
        shuffle (bool): whether to shuffle the indices
        random_seed (int): random seed to use for shuffling
        verbose (bool): whether to print out information about the sampler

    ### Ancestors (in MRO)

    * torch.utils.data.sampler.Sampler
    * typing.Generic