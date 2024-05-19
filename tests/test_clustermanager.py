from cluster.ClusterManager import ClusterManager

import unittest


class TestClusterManager(unittest.TestCase):

    def test_info(self):
        df = ClusterManager.info()
        print(df)

        pass

    def test_get_cluster_data(self):
        data = ClusterManager.get_cluster_data(0)

        pass
