import unittest
import numpy as np

from dataprocess.SpectralData import SpectralData
from astropy.io.fits.header import Header
from astropy.io.fits.fitsrec import FITS_rec


from dataprocess.DataProcess import DataProcess
from dataprocess.LamostDataset import LamostDataset
from dataprocess.SDSSDataset import SDSSDataset

from util import show_spectraldata, show_reduce_data
from reducer import ReduceData
from reducer.UMAPReducer import UMAPReducer
from reducer.Reducer import Reducer
from reducer.ReduceManager import ReduceManager


class TestUtil(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_show_spectraldata(self):
        dataset = DataProcess.load_dataset("StdDataset-007")
        for i in dataset:
            show_spectraldata(i)

        self.assertTrue(True)

    def test_show_reducedata(self):
        # dataset = SDSSDataset()
        # dataset.add_dataset_parallel(r"origin_data\SDSS\QSG")
        # reducer = UMAPReducer(
        #     dimension=20,
        #     n_neighbors=5,
        #     metric="euclidean",
        #     learning_rate=1.0,
        #     min_dist=0.1,
        # )
        # reducer.reduce("SDSSDataset-002")

        ReduceManager.info_result()
        data = ReduceManager.get_result(4)
        # separate
        show_reduce_data(data, mode="overlay", label="subclass")
