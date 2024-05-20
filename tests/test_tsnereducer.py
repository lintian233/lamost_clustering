import unittest

import numpy as np

from reducer.TSNEReducer import TSNEReducer
from dataprocess.DataProcess import DataProcess


class TestTSNEReducer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_tsne_reduce(self):
        reducer = TSNEReducer(dimension=2, perplexity=30)
        dataset = DataProcess.load_dataset("LamostDataset-008")
        dataset = DataProcess.preprocessing(dataset)
        result = reducer.reduce(dataset)
        pass
