import unittest
from pandas import DataFrame


from dataprocess.DataProcess import DataProcess


class TestDataProcess(unittest.TestCase):
    def test_list_datasets(self):
        result = DataProcess.list_datasets()
        # print(result)
        self.assertIsInstance(result, DataFrame)
        
