import numpy as np
from dataclasses import dataclass

from numpy.typing import NDArray
from typing import Any

"""
因为ReduceData的二维和n维数组非常的大，对于data2d,datand,labels,用npy储存,超参数列表用json储存
存储在一个文件夹内，文件夹名字为目录结构如下。
Data/
    -ClusterData/
                -SpectralClustering/
                                   -XXX-SNxxxx-Qxxx-Gxxx/
                                                        -data2d.npy
                                                        -labels.npy
                                                        -hyperparameters.json
"""
@dataclass
class ClusterData():
    data2d: NDArray[np.float64]
    labels: NDArray[np.float64]
    hyperparameters: dict[str, Any]

    def __init__(self, data2d: NDArray[np.float64], labels: NDArray[np.float64], hyperparameters: dict[str, Any]):
        self.data2d = data2d
        self.labels = labels
        self.hyperparameters = hyperparameters

    @staticmethod
    def from_cluster_data(cluster_data: "ClusterData") -> "ClusterData":
        return ClusterData(cluster_data.data2d, cluster_data.labels, cluster_data.hyperparameters)


