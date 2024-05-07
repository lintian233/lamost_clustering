import unittest

from reducer.PCAReducer import PCAReducer


class TestReduce(unittest.TestCase):
    def test_info_result(self):
        print('\n')
        reducer = PCAReducer(dimension=50)
        reducer.info_result
        pass
