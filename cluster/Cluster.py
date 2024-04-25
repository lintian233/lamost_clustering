from abc import ABC, abstractmethod
from typing import Any

from .ClusterData import ClusterData

class Cluster(ABC):
    cluster: Any
    result_dir: str
    hyperparameters: dict[str, Any]


    @abstractmethod
    def cluster(self, data) -> str:
        """
        实现自有的聚类方法，将聚类结果保存在result_dir中，
        返回文件路径
        """
        pass


    def info_result(self) -> None:
        """
        返回一个所有聚类结果的表，
        给出了结果目录下的所有当前方法(PCA/UMAP)->(当前类名)的一个超参数所对应的数据表：
        exp:
        index method n_cluster random_stat
        1           SpectralClustering         50             42

        """
        raise NotImplementedError("info_result method not implemented")
    
    
    def get_result(self, index: str) -> ClusterData:
        """
        给定名称，在结果目录文件夹内检索并返回ClusterData
        """
        raise NotImplementedError("get_result method not implemented")