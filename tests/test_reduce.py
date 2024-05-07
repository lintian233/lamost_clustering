import unittest

from reducer.PCAReducer import PCAReducer


class TestReduce(unittest.TestCase):
    def test_info_result(self):
        reducer = PCAReducer(dimension=50)
        data = reducer.get_result(4)
        pass
