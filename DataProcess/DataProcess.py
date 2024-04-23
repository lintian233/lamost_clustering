from typing import int, Any, List, Tuple

from numpy.typing import ArrayLike

from SpectralData import SpectralData
from Dataset import Dataset

class DataProcess:
    
    @staticmethod
    def get_subclass_dataset(dataset:Dataset, subclass: str) -> Dataset:
        """
        TODO :从数据集中获取子类数据集
        参数：
        dataset: Dataset, 数据集
        subclass: str, 子类名称
        返回：
        Dataset, 子类数据集
        """
        raise NotImplementedError("get_subclass_dataset method not implemented")
    
    @staticmethod
    def get_class_data(dataset:Dataset, _class: str) -> Dataset:
        """
        TODO :从数据集中获取类数据集
        参数：
        dataset: Dataset, 数据集
        class: str, 类名称
        返回：
        Dataset, 类数据集
        """
        raise NotImplementedError("get_class_data method not implemented")
    
    
