
import unittest
import numpy as np

from dataprocess import SpectralData, SpectralDataType



class TestSpectralData(unittest.TestCase):
    data: SpectralData
    def setUp(self) -> None:
        # self.data = SpectralData()
        pass


    def test_init(self):
        test_data = np.zeros(1, dtype=SpectralDataType)
        test_data["name"] = "test"
        test_data["flux"][0] = np.random.rand(6000)
        test_data["wavelength"][0] = np.random.rand(6000)
        test_data["class"] = "test"
        test_data["subclass"] = "test"
        self.data = SpectralData.from_spectral_data(test_data)
        save_path = r"./tests/file/test_spectral_data.npy"
        
        np.save(save_path, self.data.data)
        self.assertTrue(np.array_equal(self.data.data, test_data))

        new_data = np.load(save_path, allow_pickle=True)
        self.assertTrue(np.array_equal(new_data, test_data))



        
