import numpy as np
from numpy.typing import NDArray
import os
from sklearn.cluster import SpectralClustering

from reducer.ReduceData import ReduceData
from reducer.ReduceManager import ReduceManager
from .Cluster import Cluster
from .ClusterData import ClusterData

from reducer.util import get_save_name

from typing import Any


class SpectralCluster(Cluster):
    def __init__(self, n_clusters: int, assign_labels: str, **args) -> None:
        super().__init__()
        self.hyperparameters = {
            "n_clusters": n_clusters,
            "assign_labels": assign_labels,
            **args,
        }
        self.cluster = SpectralClustering(
            n_jobs=-1,
            **self.hyperparameters,
        )

    def fit(self, reduce_data: ReduceData) -> str:

        save_name = get_save_name("Spectral", self.hyperparameters)
        dataset_index = reduce_data.info[0]
        reduce_info = reduce_data.info[1]
        save_path = f"{self.cluster_dir}{dataset_index}/{reduce_info}/{save_name}.npy"

        if os.path.exists(save_path):
            cluster_np = np.load(save_path, allow_pickle=True)
            result = ClusterData.from_numpy(*cluster_np)
            return result

        data2d = reduce_data.data2d
        datand = reduce_data.datand
        classes = reduce_data.classes
        subclasses = reduce_data.subclasses
        obsid = reduce_data.obsid
        labels = self.cluster.fit_predict(datand)
        info = np.array(
            [reduce_data.info[0], reduce_data.info[1], save_name], dtype=object
        )

        result_numpy = np.zeros(7, dtype=object)
        result_numpy[0] = data2d
        result_numpy[1] = datand
        result_numpy[2] = classes
        result_numpy[3] = subclasses
        result_numpy[4] = labels
        result_numpy[5] = obsid
        result_numpy[6] = info

        result = ClusterData(data2d, datand, classes, subclasses, labels, obsid, info)

        save_dir = f"{self.cluster_dir}{dataset_index}/{reduce_info}/"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(save_path, result_numpy, allow_pickle=True)

        return result
