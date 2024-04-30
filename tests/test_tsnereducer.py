import unittest

import numpy as np

from reducer.TSNEReducer import TSNEReducer


class TestTSNEReducer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_reduce(self):
        reducer = TSNEReducer(
            dimension=2, perplexity=30, learning_rate=200, n_iter=1000
        )
        data = np.random.rand(100, 100)
        reduce_data = reducer.reduce(data)
        self.assertEqual(reduce_data.shape, (100, 2))
