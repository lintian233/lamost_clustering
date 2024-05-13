import unittest
import numpy as np
import time

from dataprocess.DataProcess import DataProcess
from dataprocess.SDSSDataset import SDSSDataset
from dataprocess.SpectralData import SDSSSpectraData
from util import show_spectraldata


class TestSDSSDataset(unittest.TestCase):

    def test_add_dataset_sdss(self):
        dataset = SDSSDataset()
        dataset.add_dataset(r"origin_data\SDSS\STAR100")
        self.assertIsInstance(dataset, SDSSDataset)
        self.assertIsInstance(dataset[0], SDSSSpectraData)

    def test_load_dataset_sdss(self):
        start_time = time.time()
        dataset = DataProcess.load_dataset("SDSSDataset-000")
        print(f"Time: {time.time() - start_time}")
        self.assertIsInstance(dataset, SDSSDataset)
        self.assertIsInstance(dataset[0], SDSSSpectraData)

        for data in dataset:
            show_spectraldata(data)
            break

    def test_add_dataset_sdss_parallel(self):

        dataset = SDSSDataset()
        start_time = time.time()
        dataset.add_dataset_parallel(r"origin_data\SDSS\STAR100")
        print(f"Parallel time: {time.time() - start_time}")
        self.assertIsInstance(dataset, SDSSDataset)
        self.assertIsInstance(dataset[0], SDSSSpectraData)

        # start_time = time.time()
        # dataset.add_dataset(r"origin_data\SDSS\STAR100")
        # print(f"Sequential time: {time.time() - start_time}")
        # self.assertIsInstance(dataset, SDSSDataset)
        # self.assertIsInstance(dataset[0], SDSSSpectraData)
