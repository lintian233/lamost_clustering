
from dataprocess import LamostDataset

import unittest


class TestLamostDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.lamost_dataset = LamostDataset()

    def test_add_dataset(self):
        dirpath = r"./origin_data/Lamost/"
        self.lamost_dataset.add_dataset(dirpath)
        self.assertEqual(len(self.lamost_dataset), 100)