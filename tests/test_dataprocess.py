import glob
import os

import unittest
import numpy as np
from pandas import DataFrame

from dataprocess.DataProcess import DataProcess
from dataprocess.Dataset import Dataset
from dataprocess.SpectralData import SpectralData, SpectralDataType
from dataprocess.LamostDataset import LamostDataset


class TestDataProcess(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_list_datasets(self):
        result = DataProcess.list_datasets()
        self.assertIsInstance(result, DataFrame)

    def test_load_dataset(self):
        raw_data_path = glob.glob(r"data/LamostDataset/LamostDataset-000*.npy")[0]
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(
                f"File {raw_data_path} not found, please check the dataset exists."
            )

        result = DataProcess.load_dataset("LamostDataset-000")
        spectral_data = result[0]
        raw_data = np.load(raw_data_path, allow_pickle=True)

        self.assertEqual(spectral_data.raw_data.dtype, SpectralDataType)
        self.assertIsInstance(spectral_data, SpectralData)
        self.assertIsInstance(result, Dataset)
        self.assertEqual(len(result), len(raw_data))

    def test_get_class_dataset(self):
        dataset = DataProcess.load_dataset("LamostDataset-000")
        result = DataProcess.get_class_dataset(dataset, "QSO")

        self.assertIsInstance(result, LamostDataset)
        for data in result:
            self.assertEqual(data.header["CLASS"], "QSO")
