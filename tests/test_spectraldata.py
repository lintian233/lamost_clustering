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

    def test_from_numpy(self):
        data = np.zeros(1, dtype=SpectralDataType)[0]
        data[1] = FITS_rec(np.zeros(1, dtype=datatype))

        spectral_data = SpectralData.from_numpy(data)

        self.assertEqual(spectral_data.raw_data.dtype, data.dtype)
        self.assertIsInstance(spectral_data.header, Header)
        self.assertIsInstance(spectral_data, SpectralData)
        self.assertIsInstance(spectral_data.data, FITS_rec)
