import unittest

from cluster import SpectralCluster

from typing import Any

class TestSpectralCluster(unittest.TestCase):
    def setUp(self) -> None:
        self.spectral_cluster = SpectralCluster()
    
    
    def test_cluster(self) -> None:
        raise NotImplementedError("test_cluster method not implemented")
