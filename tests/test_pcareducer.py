import unittest

import numpy as np

from reducer.PCAReducer import PCAReducer
from reducer.TSNEReducer import TSNEReducer
from reducer.UMAPReducer import UMAPReducer
from dataprocess.DataProcess import DataProcess


class TestPCAReducer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_pca_reduce(self):
        dataset = DataProcess.load_dataset("LamostDataset-000")
        dataset = DataProcess.preprocessing(dataset)
        reducer = PCAReducer(dimension=50)
        result = reducer.reduce(dataset)
        pass
