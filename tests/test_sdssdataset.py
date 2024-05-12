import unittest

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
        dataset = DataProcess.load_dataset("SDSSDataset-000")
        self.assertIsInstance(dataset, SDSSDataset)
        self.assertIsInstance(dataset[0], SDSSSpectraData)

        show_spectraldata(dataset[0])
