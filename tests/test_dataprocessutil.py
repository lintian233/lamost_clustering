

from dataprocess.util import *
import unittest



class TestDataprocessutil(unittest.TestCase):
    
    def setUp(self) -> None:
        pass

    def test_init(self):

        dirpath = r"./tests/file/"
        data = np.zeros(10, dtype=SpectralDataType)
        np.save(dirpath + "001-SN10-STAR5-YSO3-GALAXY2.npy", data)
        index = generate_new_index(dirpath)
        self.assertEqual(index, "002")





