from cluster.SpectralCluster import SpectralCluster
from cluster.HDBSCANCluster import HDBSCANCluster
from cluster.Cluster import Cluster
from reducer.ReduceManager import ReduceManager
import unittest


class TestCluster(unittest.TestCase):

    def test_spectral_cluster(self):
        data = ReduceManager.get_result(0)
        cluster = SpectralCluster(
            n_clusters=3,
            assign_labels="discretize",
            n_components=20,
            n_neighbors=5,
        )
        cluster.fit(data)

        pass

    def test_hdbscan_cluster(self):
        data = ReduceManager.get_result(0)
        cluster = HDBSCANCluster(
            min_cluster_size=20,
            min_samples=20,
            cluster_selection_epsilon=0.5,
        )
        cdata = cluster.fit(data)
        pass
