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
        # print(result)
        self.assertIsInstance(result, DataFrame)

    def test_load_dataset(self):
        result = DataProcess.load_dataset("LamostDataset-001")
        
        self.assertEqual(len(result), 100)
        spectral_data = result[0]
        self.assertEqual(spectral_data.data.dtype, SpectralDataType)
        self.assertIsInstance(spectral_data, SpectralData)
        self.assertIsInstance(result, Dataset)

        numpy_data = np.load("data\LamostDataset\LamostDataset-001-SN100-STAR0-QSO100-GALAXY0.npy", allow_pickle=True)
        self.assertTrue(np.array_equal(result.to_numpy(), numpy_data))

    def test_get_class_dataset(self):
        dataset = DataProcess.load_dataset("LamostDataset-001")
        result = DataProcess.get_class_dataset(dataset, "QSO")
        self.assertEqual(len(result), 100)
        self.assertIsInstance(result, LamostDataset)

        for data in result:
            self.assertEqual(data["class"], "QSO")
        
