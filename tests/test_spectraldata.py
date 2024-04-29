import unittest
import numpy as np

from dataprocess.SpectralData import SpectralData, SpectralDataType
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.header import Header

class TestSpectralData(unittest.TestCase):
    data: SpectralData

    def setUp(self) -> None:
        # self.data = SpectralData()
        pass

    def test_init(self):
        # test_data = np.zeros(1, dtype=SpectralDataType)[0]
        # test_data["name"] = "test"
        # test_data["flux"] = np.random.rand(6000)
        # test_data["wavelength"] = np.random.rand(6000)
        # test_data["class"] = "test"
        # test_data["subclass"] = "test"
        # self.data = SpectralData.from_numpy(test_data)
        # save_path = r"./tests/file/test_spectral_data.npy"

        # np.save(save_path, self.data.data)
        # self.assertTrue(np.array_equal(self.data.data, test_data))

        # new_data = np.load(save_path, allow_pickle=True)
        # self.assertTrue(np.array_equal(new_data, test_data))
        pass

    def test_from_numpy(self):
        data = np.zeros(1, dtype=SpectralDataType)[0]
        
        spectral_data = SpectralData.from_numpy(data)
        self.assertIsInstance(spectral_data, SpectralData)
        self.assertEqual(spectral_data.raw_data.dtype, data.dtype)
        self.assertIsInstance(spectral_data.header, Header)
        self.assertIsInstance(spectral_data.data, FITS_rec)
