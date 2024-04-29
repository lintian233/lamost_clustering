from dataprocess.LamostDataset import LamostDataset
from dataprocess.SpectralData import SpectralDataType

import unittest
import os
import glob


def clear():
    dirpath = r"./tests/file/"
    files = glob.glob(dirpath + "LamostDataset-*.npy")
    for file in files:
        os.remove(file)


class TestLamostDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.lamost_dataset = LamostDataset()
        self.lamost_dataset.change_dir_base_path(r"./tests/file/")

    def test_add_dataset(self):
        dirpath = r"./origin_data/Lamost/"
        self.lamost_dataset.add_dataset(dirpath)

        self.assertEqual(len(self.lamost_dataset), 100)

        # to_numpy()
        data = self.lamost_dataset.to_numpy()
        self.assertEqual(data.shape, (100,))

        # get_item
        data = self.lamost_dataset[0]
        self.assertEqual(data.raw_data.dtype, SpectralDataType)
