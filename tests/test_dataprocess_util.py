
import os
from dataprocess.util import *
from dataprocess.SpectralData import SpectralDataType
import unittest
import glob

def clear():
    dirpath = r"./tests/file/"
    files = glob.glob(dirpath + "LamostDataset-*.npy")
    for file in files:
        os.remove(file)

class TestDataprocessutil(unittest.TestCase):
    
    def setUp(self) -> None:
        clear()
    def test_genetrate_new_index(self):
        dirpath = r"./tests/file/"
        data = np.zeros(10, dtype=SpectralDataType)
        np.save(dirpath + "LamostDataset-000-SN100-STAR0-QSO100-GALAXY0.npy", data)
        index = generate_new_index(dirpath)
        self.assertEqual(index, "001")





