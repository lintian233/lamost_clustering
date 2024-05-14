import unittest

from reducer.ReduceManager import ReduceManager


class TestReduceManager(unittest.TestCase):

    def test_info_result(self):
        ReduceManager.info_result()
        pass

    def test_get_result(self):
        data = ReduceManager.get_result(0)
        pass
