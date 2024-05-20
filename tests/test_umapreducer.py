import unittest
import time

import numpy as np

from reducer.UMAPReducer import UMAPReducer
from dataprocess.DataProcess import DataProcess


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
        dataset = DataProcess.load_dataset("LamostDataset-008")
        dataset = DataProcess.preprocessing(dataset)
        result = reducer.reduce(dataset)
        pass

    @unittest.skip("Time-consuming test")
    def test_dataprocess_time(self):
        NUM = 100
        LODER = 5000  # 加载数据集的次数
        start_time = time.time()

        for i in range(LODER):
            reducer = UMAPReducer(
                dimension=20,
                n_neighbors=5,
                metric="euclidean",
                learning_rate=1.0,
                min_dist=0.01 * i,
            )
            reducer.reduce("LamostDataset-000")

        end_time = time.time()
        print(f"Load dataset time: {end_time - start_time} seconds")
