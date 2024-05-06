import unittest

import numpy as np

from reducer.TSNEReducer import TSNEReducer


class TestTSNEReducer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_tsne_reduce(self):
        reducer = TSNEReducer(dimension=2, perplexity=30, learning_rate=200, n_iter=250)
        reduce_data = reducer.reduce("LamostDataset-000")
        pass
