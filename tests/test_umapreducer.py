import unittest

import numpy as np

from reducer.UMAPReducer import UMAPReducer


class TestUMAPReducer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_umap_reduce(self):
        reducer = UMAPReducer(
            dimension=20,
            n_neighbors=5,
            metric="euclidean",
            learning_rate=1.0,
            min_dist=0.1,
        )
        reducer.reduce("LamostDataset-000")
        pass
