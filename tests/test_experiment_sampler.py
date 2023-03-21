import sys
sys.path.append('../../') # to have access to NDNT

from NDNT.samplers.experiment_sampler import *


class TestDataset_withThreeSameExperiments(Dataset):
    def __init__(self):
        self.stim = [[0,0,1,1]]*27  # just some sample stim and robs
        self.robs = [0]*10+[1]*7+[0]*10
        self.NT = 27
        # file_index --> block_inds --> timepoints
        self.file_index = [0,0,0, 1,1,1, 2,2,2]
        self.block_inds = [np.array([0,1,2]),   np.array([3,4,5]),   np.array([6,7,8]),
                           np.array([9,10,11]), np.array([12,13,14]),np.array([15,16,17]),
                           np.array([18,19,20]),np.array([21,22,23]),np.array([24,25,26])]

    def __getitem__(self, idx):
        return self.stim[idx]

    def __len__(self):
        return len(self.robs)


class TestDataset_withDifferentExperiments(Dataset):
    def __init__(self):
        self.stim = [[0,0,1,1]]*53 # just some sample stim and robs
        self.robs = [0]*25+[1]*25+[0]*3
        self.NT = 53
        # file_index --> block_inds --> timepoints
        self.file_index = [0,0,0,0,0, 1,1,1,1, 2,2,2,2,2,2,2]
        self.block_inds = [np.array([0,1,2]), np.array([3,4,5,6,7,8,9]), np.array([10]), np.array([11,12]), np.array([13,14,15,16,17,18,19,20]),
                           np.array([21,22,23]), np.array([24,25,26,27,28,29,30]), np.array([31,32]), np.array([33]),
                           np.array([34,35]), np.array([36,37,38,39]), np.array([40]), np.array([41,42,43,44,45]), np.array([46,47,48]), np.array([49]), np.array([50,51,52])]

    def __getitem__(self, idx):
        return self.stim[idx]

    def __len__(self):
        return len(self.robs)


def test_construct_exp_to_time_diff_exps():
    expected_exp_to_time = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                            1: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                            2: [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]}
    
    dataset = TestDataset_withDifferentExperiments()
    actual_exp_to_time = construct_exp_to_time(dataset)
    assert(expected_exp_to_time == actual_exp_to_time)


def test_construct_exp_to_time_diff_exps_with_indices():
    expected_exp_to_time = {0: [0, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20],
                            1: [25, 26, 27, 28, 33],
                            2: [34, 35, 36, 40, 45, 46, 47, 48]}
    
    indices = [0, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 25, 26, 27, 28, 33, 34, 35, 36, 40, 45, 46, 47, 48]

    dataset = TestDataset_withDifferentExperiments()
    actual_exp_to_time = construct_exp_to_time(dataset, indices=indices)
    assert(expected_exp_to_time == actual_exp_to_time)


def test_experiment_sampler_same_exps_and_small_batch_size():
    expected_batches = [[0, 9, 18],
                        [1, 10, 19],
                        [2, 11, 20],
                        [3, 12, 21],
                        [4, 13, 22],
                        [5, 14, 23],
                        [6, 15, 24],
                        [7, 16, 25],
                        [8, 17, 26]]

    # make a sample dataset
    dataset = TestDataset_withThreeSameExperiments()

    # create the sampler
    exp_sampler = ExperimentSampler(dataset, batch_size=3, shuffle=False)

    # test the sequential sampler
    assert(len(list(exp_sampler)) == 9)
    assert(len(exp_sampler) == 9)
    for expected_batch, actual_batch in zip(expected_batches, exp_sampler):
        assert(expected_batch == actual_batch)


def test_experiment_sampler_diff_exps_and_small_batch_size():
    expected_batches = [[0, 21, 34],
                        [1, 22, 35],
                        [2, 23, 36],
                        [3, 24, 37],
                        [4, 25, 38],
                        [5, 26, 39],
                        [6, 27, 40],
                        [7, 28, 41],
                        [8, 29, 42],
                        [9, 30, 43],
                        [10, 31, 44],
                        [11, 32, 45]]

    # make a sample dataset
    dataset = TestDataset_withDifferentExperiments()

    # create the sampler
    exp_sampler = ExperimentSampler(dataset, batch_size=3, shuffle=False)

    # test the sequential sampler
    assert(len(exp_sampler) == 13)
    assert(len(list(exp_sampler)) == 13)
    for expected_batch, actual_batch in zip(expected_batches, exp_sampler):
        assert(expected_batch == actual_batch)


def test_experiment_sampler_diff_exps_and_medium_batch_size():
    expected_batches = [[0, 1, 2, 21, 22, 34, 35, 36],
                        [3, 4, 5, 23, 24, 37, 38, 39],
                        [6, 7, 8, 25, 26, 40, 41, 42],
                        [9, 10, 11, 27, 28, 43, 44, 45],
                        [12, 13, 14, 29, 30, 46, 47, 48],
                        [15, 16, 17, 31, 32, 49, 50, 51]] 

    # make a sample dataset
    dataset = TestDataset_withDifferentExperiments()

    # create the sampler
    exp_sampler = ExperimentSampler(dataset, batch_size=7, shuffle=False)

    # test the sequential sampler
    assert(len(list(exp_sampler)) == 6)
    assert(len(exp_sampler) == 6)
    for expected_batch, actual_batch in zip(expected_batches, exp_sampler):
        assert(expected_batch == actual_batch)
    
    
def test_experiment_sampler_diff_exps_and_large_batch_size():
    expected_batches = [[0, 1, 2, 3, 4, 5, 6, 7, 21, 22, 23, 24, 25, 34, 35, 36, 37, 38, 39, 40],
                        [8, 9, 10, 11, 12, 13, 14, 15, 26, 27, 28, 29, 30, 41, 42, 43, 44, 45, 46, 47]]

    # make a sample dataset
    dataset = TestDataset_withDifferentExperiments()

    # create the sampler
    exp_sampler = ExperimentSampler(dataset, batch_size=20, shuffle=False)

    # test the sequential sampler
    assert(len(list(exp_sampler)) == 2)
    assert(len(exp_sampler) == 2)
    for expected_batch, actual_batch in zip(expected_batches, exp_sampler):
        assert(expected_batch == actual_batch)


def test_experiment_sampler_diff_exps_with_indices():
    expected_batches = [[0, 6, 7, 25, 34, 35],
                        [8, 9, 10, 26, 36, 40],
                        [11, 12, 16, 27, 45, 46],
                        [17, 18, 19, 28, 47, 48]]

    # make a sample dataset
    dataset = TestDataset_withDifferentExperiments()
    
    # create the sampler
    indices = [0, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 25, 26, 27, 28, 33, 34, 35, 36, 40, 45, 46, 47, 48]
    exp_sampler = ExperimentSampler(dataset, batch_size=7, indices=indices, shuffle=False)

    # test the sequential sampler
    assert(len(list(exp_sampler)) == 4)
    assert(len(exp_sampler) == 4)
    for expected_batch, actual_batch in zip(expected_batches, exp_sampler):
        assert(expected_batch == actual_batch)


def test_experiment_sampler_same_exps_with_shuffle():
    expected_batches = [[8, 17, 26],
                        [3, 12, 21],
                        [2, 11, 20],
                        [5, 14, 23],
                        [4, 13, 22],
                        [6, 15, 24],
                        [0, 9, 18],
                        [1, 10, 19],
                        [7, 16, 25]]

    # make a sample dataset
    dataset = TestDataset_withThreeSameExperiments()

    # create the sampler
    exp_sampler = ExperimentSampler(dataset, batch_size=3, shuffle=True, random_seed=1234)

    # test the sequential sampler
    assert(len(list(exp_sampler)) == 9)
    assert(len(exp_sampler) == 9)
    for expected_batch, actual_batch in zip(expected_batches, exp_sampler):
        assert(expected_batch == actual_batch)
    

def test_experiment_sampler_diff_exps_with_shuffle():
    expected_batches = [[3, 4, 5, 23, 24, 37, 38, 39],
                        [6, 7, 8, 25, 26, 40, 41, 42],
                        [15, 16, 17, 31, 32, 49, 50, 51],
                        [12, 13, 14, 29, 30, 46, 47, 48],
                        [0, 1, 2, 21, 22, 34, 35, 36],
                        [9, 10, 11, 27, 28, 43, 44, 45]]

    # make a sample dataset
    dataset = TestDataset_withDifferentExperiments()

    # create the sampler
    exp_sampler = ExperimentSampler(dataset, batch_size=7, shuffle=True, random_seed=1234)
    
    # test the sequential sampler
    assert(len(list(exp_sampler)) == 6)
    assert(len(exp_sampler) == 6)
    for expected_batch, actual_batch in zip(expected_batches, exp_sampler):
        assert(expected_batch == actual_batch)
