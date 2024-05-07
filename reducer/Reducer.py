from abc import ABC, abstractmethod
from typing import Any
import os

from config.config import REDUCEDATAPATH
from reducer.ReduceData import ReduceData


"""

"""


class Reducer(ABC):
    index: str
    reducer: Any
    result_dir: str
    hyperparameters: dict[str, Any]
    dimension: int

    def __init__(self) -> None:
        self.result_dir = REDUCEDATAPATH
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    @abstractmethod
    def reduce(self, dataset_index: str) -> str:
        """
        TODO: 将当前数据集以当前的超参数降维至所需维度，
        然后将结果保存在result_dir中。
        完成后返回文件路径
        """
        pass

    def info_result(self):
        """
        TODO :返回一个所有降维结果的表，
        给出了结果目录下的所有当前方法(PCA/UMAP)->(当前类名)的一个超参数所对应的数据表：
        exp:
        index method n_cluster random_stat
        1           PCA         50             42
        """

        raise NotImplementedError("info_result method not implemented")

    def get_result(self, index) -> ReduceData:
        """
        TODO :给定名称，在结果目录文件夹内检索并返回ReduceData
        """
        raise NotImplementedError("get_result method not implemented")
