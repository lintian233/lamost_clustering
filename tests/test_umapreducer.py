import unittest

import numpy as np

from reducer.UMAPReducer import UMAPReducer

class TestUMAPReducer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_reduce(self):
        reducer = UMAPReducer(dimension=2, n_neighbors=5, metric='euclidean', learning_rate=1.0, min_dist=0.1)
        data = np.random.rand(100, 100)
        reduce_data = reducer.reduce(data)
        self.assertEqual(reduce_data.shape, (100, 2))