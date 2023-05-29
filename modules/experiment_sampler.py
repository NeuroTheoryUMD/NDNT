import numpy as np

from random import Random
from typing import Iterator
from torch.utils.data import Dataset
from torch.utils.data import Sampler

class ExperimentBatchGenerator:
    """
    Generator that returns batches of timepoints for a single experiment.
    
    Args:
        timepoints (list): list of timepoints to sample from
        batch_size (int): size of the batch to sample
        shuffle (bool): whether to shuffle the batches
        random_seed (int): random seed to use for shuffling 
    """
    
    def __init__(self, timepoints:list, batch_size:int, shuffle:bool, random_seed:int=None) -> None:
        # break the timepoints into a set of batches that can be shuffled
        self.batches = []
        self.current_batch_index = 0
        for i in range(0, len(timepoints), batch_size):
            # if the batch is larger than what is left, don't use the remaining timepoints
            if i+batch_size > len(timepoints):
                continue
            self.batches.append(timepoints[i:i+batch_size])
        
        if shuffle: # shuffle the batches
            if random_seed is not None: # use the seed if provided
                Random(random_seed).shuffle(self.batches)
            else:
                Random().shuffle(self.batches)
    
    def next(self) -> list:
        if self.current_batch_index < len(self.batches):
            batch = self.batches[self.current_batch_index]
            self.current_batch_index += 1
            return batch
        else:
            # there are no more batches to serve 
            raise StopIteration()


class ExperimentBatchIterator(Iterator):
    """
    Iterator that returns batches of timepoints across experiments.
    
    Args:
        exp_to_time (dict): map from experiment indices to timepoints
        exp_batch_sizes (list): list of batch sizes for each experiment
        shuffle (bool): whether to shuffle the batches
        random_seed (int): random seed to use for shuffling
    """
    def __init__(self, exp_to_time, exp_batch_sizes, shuffle:bool, random_seed:int=None) -> None:
        self.exp_batch_generators = []

        # create the batch generators for each experiment
        for exp_index, timepoints in exp_to_time.items():
            exp_batch_size = exp_batch_sizes[exp_index]
            self.exp_batch_generators.append(ExperimentBatchGenerator(timepoints, exp_batch_size, shuffle, random_seed))
    
    def __next__(self) -> list:
        # get the next batch for each experiment and accumulate
        batch = []
        for exp_batch_generator in self.exp_batch_generators:
            batch.extend(exp_batch_generator.next())
        return batch


def construct_exp_to_time(dataset, indices=None) -> dict:
    """
    Construct a dictionary that maps experiment indices to timepoints.
    
    Args:
        dataset (Dataset): dataset to sample from
        indices (list): indices to filter by
    """
    
    # diff the dataset.file_index
    # we need to expand out the file_index by block_inds before we can work with them
    # file_index --> block_inds --> timepoints
    exp_to_time = {}
    for fi, file_index in enumerate(dataset.file_index):
        # use try/except trick to speed up dictionary key checking
        # equivalent to - if file_index in exp_to_time:
        exp_indices = dataset.block_inds[fi].tolist()
        try:
            exp_to_time[file_index].extend(exp_indices)
        except:
            exp_to_time[file_index] = exp_indices

    # filter by indices if they are provided
    if indices is not None:
        for ei in exp_to_time.keys():
            # get the indices for the experiment
            if type(indices) is not np.array: # convert to an array for filtering purposes
                indices = np.array(indices)
            exp_indices = np.array(exp_to_time[ei])
            # subset indices to the range of the exp_indices
            min_index = np.min(exp_indices)
            max_index = np.max(exp_indices)
            subsetted_indices = indices[np.where((indices >= min_index) & (indices <= max_index))]
            # filter by value (not index)
            exp_to_time[ei] = [exp_ind for exp_ind in exp_indices.tolist() if exp_ind in subsetted_indices] # convert back to list
            
    return exp_to_time

class ExperimentSampler(Sampler):
    """Samples elements across experiments.

    Args:
        dataset (Dataset): dataset to sample from
        batch_size (int): size of the batch to sample
        indices (list): indices to sample from
        shuffle (bool): whether to shuffle the indices
        random_seed (int): random seed to use for shuffling
        verbose (bool): whether to print out information about the sampler 
    """
    
    def __init__(self, dataset: Dataset, batch_size:int, indices:list=None, shuffle:bool=True, random_seed:int=None, verbose=False) -> None:
        # don't need to call super() here
        if verbose:
            print('USING EXPERIMENT SAMPLER')

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.num_exps = len(np.unique(dataset.file_index))

        # we need to expand out the file_index by block_inds before we can work with them
        # file_index --> block_inds --> timepoints
        self.exp_to_time = construct_exp_to_time(dataset, indices)

        self.exp_lengths = [len(time) for time in self.exp_to_time.values()]
        self.exp_batch_sizes = []
        for exp_index, exp_time in self.exp_to_time.items():
            exp_length = len(exp_time)
            # TODO: maybe don't use round, b/c it could leave some experiment batches empty in some cases
            if indices is not None:
                exp_batch_size = round(exp_length / (dataset.NT - len(indices)) * batch_size)
            else:
                exp_batch_size = round(exp_length / dataset.NT * batch_size)
            if verbose:
                print(exp_index, '-->', 'L', exp_length, 'B', exp_batch_size)
            self.exp_batch_sizes.append(exp_batch_size)

        # TODO: use the floor of the smallest experiment for now, and consider using the largest one later
        if verbose:
            print(np.array(self.exp_lengths) / np.array(self.exp_batch_sizes))
        self.num_batches = int(np.min(np.array(self.exp_lengths) / np.array(self.exp_batch_sizes)))

    def __iter__(self) -> Iterator[int]:
        return ExperimentBatchIterator(self.exp_to_time, self.exp_batch_sizes, self.shuffle, self.random_seed)

    def __len__(self) -> int:
        return self.num_batches
