import unittest

import numpy as np

import dataprocess.DataProcess as dp
from reducer.util import get_data_from_dataset_index


class TestSomething(unittest.TestCase):
    def test_load_dataset(self):
        dataset = dp.load_dataset("LamostDataset-000")
        data = get_data_from_dataset_index("LamostDataset-000")
        pass

    @unittest.skip("skip")
    def test_load_reducedata(self):
        data1 = np.load(
            "data/reduced_data/LamostDataset-000/PCA-n_components-2.npy",
            allow_pickle=True,
        )
        data2 = np.load(
            "data/reduced_data/LamostDataset-000/PCA-n_components-50.npy",
            allow_pickle=True,
        )
        data3 = np.load(
            "data/reduced_data/LamostDataset-000/TSNE-n_components-2-perplexity-30-learning_rate-200-n_iter-250.npy",
            allow_pickle=True,
        )
        data4 = np.load(
            "data/reduced_data/LamostDataset-000/UMAP-n_components-20-n_neighbors-5-metric-euclidean-learning_rate-1.0-min_dist-0.1.npy",
            allow_pickle=True,
        )
        pass
