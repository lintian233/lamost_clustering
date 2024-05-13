from abc import ABC, abstractmethod
from typing import Any

from .ClusterData import ClusterData
from reducer.ReduceData import ReduceData
from config.config import CLUSTER_DATA_PATH


class Cluster(ABC):
    cluster: Any
    result_dir: str
    hyperparameters: dict[str, Any]

    def __init__(self) -> None:
        self.result_dir = CLUSTER_DATA_PATH
        self.hyperparameters = {}

    @abstractmethod
    def cluster(self, data: ReduceData) -> str:
        """
        实现自有的聚类方法，将聚类结果保存在result_dir中，
        返回文件路径
        """
        pass

    @staticmethod
    def info_result() -> None:
        """
        返回一个所有聚类结果的表，
        给出了结果目录下的所有当前方法(PCA/UMAP)->(当前类名)的一个超参数所对应的数据表：
        exp:
        INDEX METHOD DATASET N_COMPONENTS N_NEIGHBORS ..
        001 SPECTRA LamostDataset-000
        002 HDBSCAN LamostDataset-001
        """
        raise NotImplementedError("info_result method not implemented")

    @staticmethod
    def get_result(self, index: str) -> ClusterData:
        """
        给定名称，在结果目录文件夹内检索并返回ClusterData
        """
        raise NotImplementedError("get_result method not implemented")
