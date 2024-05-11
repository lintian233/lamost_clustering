import unittest
import numpy as np

from dataprocess.SpectralData import SpectralData, SpectralDataType, datatype
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.header import Header


class TestSpectralData(unittest.TestCase):
    data: SpectralData

    def setUp(self) -> None:
        # self.data = SpectralData()
        pass
