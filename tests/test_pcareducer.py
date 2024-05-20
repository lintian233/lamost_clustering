import unittest

import numpy as np

from reducer.PCAReducer import PCAReducer
from dataprocess.DataProcess import DataProcess



class TestPCAReducer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_pca_reduce(self):
        reducer = PCAReducer(dimension=50)
        dataset = DataProcess.load_dataset("LamostDataset-008")
        dataset = DataProcess.preprocessing(dataset)
        result = reducer.reduce(dataset)
        pass
