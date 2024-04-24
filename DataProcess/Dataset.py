from abc import ABC, abstractmethod
from typing import int, Any

from numpy.typing import ArrayLike

from .SpectralData import SpectralData
from ..config import DATASETBASEPATH

class Dataset(ABC):
    __dataset: ArrayLike[SpectralData]
    __dir_base_path = DATASETBASEPATH

    def __init__(self, dirpath: str):
        """
        TODO :目录结构有待确定尝试从_dir_data_path中读取数据集.
        使用np.load()函数加载数据集。
        对于每一个光谱初始化成SpectralData对象。    
        """

    def __getitem__(self, idx: int):
        """Return the item at the given index"""
        return self.dataset[idx]
    
    
    def __len__(self):
        """Return the number of items in the dataset"""
        return len(self.dataset)    


    def __iter__(self):
        """Return an iterator over the dataset"""
        return iter(self.dataset)
    

    def info(self) -> str:
        """
        TODO: 返回输出有多少条光谱，每条光谱的大小，每个光谱有多少类。
        FORMAT:
        CLASS, SUBCLASS, FLUX_SHAPE, WAVELENGTH_SHAPE
        """
        raise NotImplementedError("info method not implemented")


    def __str__(self) -> str:
        return self.info()
    

    def parse_dataset(self, dirpath: str) -> None:
        """
        TODO : 解析数据集文件夹，使用read_data函数， 读取数据集中的所有数据。
        以ArrayLike[SpectralData]的形式存储在self.__dataset中。并将这个数据集序列化到__dir_data_path中。
        参数：
        dirpath: str, 数据集的路径
        返回：
        无

        序列化的数据集文件格式：
        1. 数据集的路径
        __dir_data_path: str
        2. 数据集格式：ArrayLike[SpectralDataType](定义在SpectralData中)
        """
        raise NotImplementedError("parse_dataset method not implemented")
    

    def to_numpy(self) -> ArrayLike[Any]:
        """
        将数据集转化为numpy数组。
        """
        raise NotImplementedError("to_numpy method not implemented")
    
    
    @abstractmethod
    def read_data(self, path: str) -> SpectralData:
        """
        TODO : 派生类需要实现的方法，根据特定数据格式读取一条光谱数据。
        参数：
        path: str, 数据集的路径
        返回：
        一个SpectralData对象。
        """
        pass

