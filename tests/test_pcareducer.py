import unittest


from reducer.PCAReducer import PCAReducer
import numpy as np


class TestPCAReducer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_pca_reduce(self):
        reducer = PCAReducer(dimension=50)
        result = reducer.reduce("LamostDataset-000")
        pass
