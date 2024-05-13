import unittest
import numpy as np

from dataprocess.SpectralData import SpectralData
from astropy.io.fits.header import Header
from astropy.io.fits.fitsrec import FITS_rec


from dataprocess.DataProcess import DataProcess
from util import show_spectraldata, show_reduce_data
from reducer import ReduceData
from reducer.PCAReducer import PCAReducer
from reducer.Reducer import Reducer


class TestUtil(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_show_spectraldata(self):
        dataset = DataProcess.load_dataset("LamostDataset-000")

        data = dataset[0]
        show_spectraldata(data)

        self.assertTrue(True)

    def test_show_reducedata(self):
        reducer = PCAReducer(5)
        reducer.info_result()
        reduce_data = reducer.get_result(3)
        show_reduce_data(reduce_data, mode="separate", label="subclass")
