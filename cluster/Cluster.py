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
        """
        pass


    def info_result(self) -> None:
        """
        """
        raise NotImplementedError("info_result method not implemented")
    
    
    def get_result(self, index: int) -> ClusterData:
        """
        """
        raise NotImplementedError("get_result method not implemented")