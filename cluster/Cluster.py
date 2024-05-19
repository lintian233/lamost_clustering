import os
from abc import ABC, abstractmethod
from typing import Any

from reducer.ReduceData import ReduceData
from config.config import CLUSTER_DATA_PATH


class Cluster(ABC):
    cluster: Any
    cluster_dir: str
    hyperparameters: dict[str, Any]

    def __init__(self) -> None:
        self.cluster_dir = CLUSTER_DATA_PATH
        if not os.path.exists(self.cluster_dir):
            os.makedirs(self.cluster_dir)
        if self.cluster_dir[-1] != "/":
            self.cluster_dir += "/"

    @abstractmethod
    def fit(self, data: ReduceData) -> str:
        """
        实现自有的聚类方法，将聚类结果保存在result_dir中，
        返回文件路径
        """
        pass
