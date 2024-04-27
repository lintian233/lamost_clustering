import unittest

from cluster import SpectralCluster

from typing import Any


class TestSpectralCluster(unittest.TestCase):
    def setUp(self) -> None:
        self.spectral_cluster = SpectralCluster()

    @unittest.skip("Not implemented")
    def test_cluster(self) -> None:
        raise NotImplementedError("test_cluster method not implemented")

    @unittest.skip("Not implemented")
    def test_info_result(self) -> None:
        raise NotImplementedError("test_info_result method not implemented")
