import os
import unittest
import glob

import numpy as np

from dataprocess.util import *
from dataprocess.SpectralData import SpectralDataType
from dataprocess.DataProcess import DataProcess


def clear():
    dirpath = r"./tests/file/"
    files = glob.glob(dirpath + "LamostDataset-000*.npy")
    for file in files:
        os.remove(file)


class TestDataprocessutil(unittest.TestCase):
    def setUp(self) -> None:
        # clear()
        dirpath = r"./tests/file/"

        # 如果没有这个文件夹创建一个
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.dirpath = dirpath

    def test_genetrate_new_index(self):
        data = np.zeros(10, dtype=SpectralDataType)
        index = generate_new_index(self.dirpath)
        self.assertEqual(index, "001")

    def test_check_dataset_index(self):
        self.assertTrue(check_dataset_index("LamostDataset-001"))
        self.assertTrue(check_dataset_index("SDSSDataset-002"))
        self.assertFalse(check_dataset_index("NONSENDataset-003"))
        self.assertFalse(check_dataset_index("LamostDataset-00a"))

    def test_get_useful(self):
        sdss_dataset = DataProcess.load_dataset("SDSSDataset-000")
        result = get_useful(sdss_dataset.dataset[0].ORMASK)
        pass
    