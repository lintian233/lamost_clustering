from dataprocess.LamostDataset import LamostDataset
from dataprocess.SpectralData import SpectralDataType
from astropy.io.fits.header import Header

import unittest
import os
import glob


class TestLamostDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.lamost_dataset = LamostDataset()

    def test_add_dataset(self):
        dirpath = r"./origin_data/Lamost/"
        self.lamost_dataset.add_dataset(dirpath)

        self.assertIsInstance(self.lamost_dataset, LamostDataset)
        self.assertIsInstance(self.lamost_dataset[0].header, Header)
