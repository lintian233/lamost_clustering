import hdbscan
import numpy as np
from typing import Any
import os

from .ClusterData import ClusterData
from .Cluster import Cluster

from reducer.ReduceData import ReduceData
from reducer.util import get_save_name


class HDBSCANCluster(Cluster):

    def __init__(self, min_cluster_size: int, min_samples: int, **args) -> None:
        super().__init__()
        self.hyperparameters = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            **args,
        }
        self.cluster = hdbscan.HDBSCAN(
            core_dist_n_jobs=-1,
            **self.hyperparameters,
        )

    def fit(self, reduce_data: ReduceData) -> ClusterData:

        save_name = get_save_name("HDBSCAN", self.hyperparameters)

        save_path = self.cluster_dir + reduce_data.info[0] + "/" + save_name + ".npy"

        if os.path.exists(save_path):
            cluster_np = np.load(save_path, allow_pickle=True)
            result = ClusterData.from_numpy(*cluster_np)
            return result

        data2d = reduce_data.data2d
        datand = reduce_data.datand
        classes = reduce_data.classes
        subclasses = reduce_data.subclasses
        obsid = reduce_data.obsid
        labels = self.cluster.fit(datand).labels_
        info = np.array([reduce_data.info[0], save_name], dtype=object)

        result_numpy = np.zeros(7, dtype=object)
        result_numpy[0] = data2d
        result_numpy[1] = datand
        result_numpy[2] = classes
        result_numpy[3] = subclasses
        result_numpy[4] = labels
        result_numpy[5] = obsid
        result_numpy[6] = info

        result = ClusterData(data2d, datand, classes, subclasses, labels, obsid, info)
        dataset_index = reduce_data.info[0]

        if not os.path.exists(self.cluster_dir + dataset_index):
            os.makedirs(self.cluster_dir + dataset_index)

        np.save(save_path, result_numpy, allow_pickle=True)

        return result
