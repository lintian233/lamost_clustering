import unittest

import dataprocess.DataProcess as dp


class TestSomething(unittest.TestCase):
    def test_load_dataset(self):
        dataset = dp.load_dataset("LamostDataset-000")
        pass
