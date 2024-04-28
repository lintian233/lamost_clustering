from abc import ABC, abstractmethod
import os
from typing import Any, List

import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ThreadPoolExecutor

from .SpectralData import SpectralData, SpectralDataType
from .util import parser_fits_path, generate_dataset_name
from config.config import DATASETBASEPATH


class Dataset(ABC):
    dataset: List[SpectralData]
    dir_base_path = DATASETBASEPATH
    name: str

    # fortest
    def change_dir_base_path(self, path: str) -> None:
        """
        更改数据集的基础路径，用于测试
        """
        self.dir_base_path = path

    def __init__(self) -> None:
        """
        TODO: 初始化Dataset类，初始化一个空的数据集。
        讲_dir_data_path设置为DATA_PATH+dataset_name/
        {dataset_name} 是一个派生类类名
        """
        if not os.path.exists(self.dir_base_path):
            os.makedirs(self.dir_base_path)

        class_name = self.__class__.__name__
        self.dir_base_path = self.dir_base_path + class_name + "/"
        if not os.path.exists(self.dir_base_path):
            os.makedirs(self.dir_base_path)

        self.dataset = []
        self.name = None

    def __getitem__(self, key: Any) -> SpectralData: 
        if isinstance(key, int):
            return self.dataset[key]

        raise TypeError("Invalid argument type")


    def __len__(self) -> int:
        """Return the number of items in the dataset"""
        return len(self.dataset)

    def __iter__(self) -> Any:
        """Return an iterator over the dataset"""
        return iter(self.dataset)

    def add_dataset(self, dirpath: str) -> NDArray:
        """
        TODO : 解析数据集文件夹，使用read_data函数， 读取数据集中的所有数据。
        以ArrayLike[SpectralData]的形式存储在self.dataset中。并将这个数据集序列化到__dir_data_path中。
        参数：
        dirpath: str, 数据集的路径
        dirpath: 下面是*.fits
        返回：
        无

        序列化的数据集文件格式：
        1. 数据集的路径
        __dir_data_path: str
        2. 数据集格式：NDArray[SpectralDataType](定义在SpectralData中)
        """
        fits_path = parser_fits_path(dirpath)

        with ThreadPoolExecutor() as executor:
            results = executor.map(self.read_data, fits_path)

        self.dataset = list(results)

        data_numpy = self.to_numpy()
        dataset_name = generate_dataset_name(
            self.__class__.__name__, self.dir_base_path, data_numpy
        )

        self.name = dataset_name
        save_path = self.dir_base_path + dataset_name + ".npy"

        np.save(save_path, data_numpy, allow_pickle=True)

        return save_path

    def to_numpy(self) -> NDArray[Any]:
        """
        NDArray[SpectralDataType] : 返回一个numpy数组，Any是SpectralDataType类型。
        将数据集转化为numpy数组。

        """
        if self.name is not None:
            datapath = self.dir_base_path + self.name + ".npy"
            if os.path.exists(datapath):
                return np.load(datapath, allow_pickle=True)

        data_numpy = np.zeros(len(self.dataset), dtype=SpectralDataType)
        for i, data in enumerate(self.dataset):
            data_numpy[i] = data.data

        return data_numpy
    

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
