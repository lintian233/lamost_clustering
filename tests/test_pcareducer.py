import unittest


from reducer.PCAReducer import PCAReducer
import numpy as np


class TestPCAReducer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_reduce(self):
        reducer = PCAReducer(dimension=2)
        data = np.random.rand(100, 100)
        reduce_data = reducer.reduce(data)
        self.assertEqual(reduce_data.shape, (100, 2))
