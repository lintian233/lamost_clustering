import glob
import os
import time

import unittest
import numpy as np
from pandas import DataFrame

from dataprocess.DataProcess import DataProcess
from dataprocess.Dataset import Dataset
from dataprocess.SpectralData import SpectralData, SpectralDataType
from dataprocess.LamostDataset import LamostDataset
from util import show_spectraldata


class TestDataProcess(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_list_datasets(self):
        result = DataProcess.list_datasets()
        self.assertIsInstance(result, DataFrame)

    def test_load_dataset(self):
        raw_data_path = glob.glob(r"data/LamostDataset/LamostDataset-000*.fits")
        if not raw_data_path:
            raise FileNotFoundError("File not found, please check the dataset exists.")
        raw_data_path = raw_data_path[0]
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(
                f"File {raw_data_path} not found, please check the dataset exists."
            )

        result = DataProcess.load_dataset("LamostDataset-000")
        spectral_data = result[0]

        try:
            bad_result = DataProcess.load_dataset("NonsenDataset-000")

        except ValueError as e:
            self.assertEqual(str(e), "DatsetType NonsenDataset not found")

        self.assertIsInstance(spectral_data, SpectralData)
        self.assertIsInstance(result, Dataset)

    def test_get_class_dataset(self):
        dataset = DataProcess.load_dataset("LamostDataset-000")
        result = DataProcess.get_class_dataset(dataset, "QSO")

        self.assertIsInstance(result, LamostDataset)
        for data in result:
            self.assertEqual(data.header["CLASS"], "QSO")

    @unittest.skip("Skip this test. it will take a long time to run.")
    def test_dataprocess_time(self):
        NUM = 1000000
        LODER = 100  # 加载数据集的次数
        start_time = time.time()

        for i in range(LODER):
            dataset = DataProcess.load_dataset("LamostDataset-000")

        end_time = time.time()
        print(f"Load dataset time: {end_time - start_time} seconds")

        dataset = DataProcess.load_dataset("LamostDataset-000")
        numpy_dataset = np.array(dataset.dataset)

        # 测量遍历Python列表的时间
        start_time = time.time()
        for i in range(NUM):
            for item in dataset:
                pass
        end_time = time.time()
        print(f"Python list iteration time: {end_time - start_time} seconds")

        # 测量遍历NumPy数组的时间
        start_time = time.time()
        for i in range(NUM):
            for item in numpy_dataset:
                pass
        end_time = time.time()
        print(f"NumPy array iteration time: {end_time - start_time} seconds")

    def test_lamost_preprocess(self):
        raw_lamost = DataProcess.load_dataset("LamostDataset-000")
        DataProcess.preprocessing("LamostDataset-000")
        pre_lamost = DataProcess.load_dataset("StdDataset-000")

        for i in range(len(raw_lamost.dataset)):
            if pre_lamost.dataset[i].header["USEFUL"]:
                show_spectraldata(raw_lamost[i])
                show_spectraldata(pre_lamost[i])
                break
        pass

    def test_sdss_preprocess(self):
        raw_sdss = DataProcess.load_dataset("SDSSDataset-000")
        DataProcess.preprocessing("SDSSDataset-000")
        pre_sdss = DataProcess.load_dataset("StdDataset-000")

        dataset = []
        for i in pre_sdss.dataset:
            if i.header["USEFUL"]:
                dataset.append(i)
        pre_sdss.dataset = dataset

        print(len(pre_sdss.dataset))
        pass
