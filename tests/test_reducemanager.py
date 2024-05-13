import unittest

from reducer.ReduceManager import ReduceManager


class TestReduceManager(unittest.TestCase):
    def test_info_result(self):
        manager = ReduceManager()
        print(manager.info_result())
        pass

    def test_get_result(self):
        manager = ReduceManager()
        result = manager.get_result(1)
        pass
