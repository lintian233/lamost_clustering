from abc import ABC, abstractmethod
from typing import Any, List
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from astropy.io import fits

from .SpectralData import SpectralData
from config.config import DATASETBASEPATH



class Dataset(ABC):
    __dataset: List[SpectralData]
    __dir_base_path = DATASETBASEPATH

    def __init__(self):
        """
        TODO: 初始化Dataset类，初始化一个空的数据集。
        讲_dir_data_path设置为DATA_PATH+dataset_name/
        {dataset_name} 是一个派生类类名
        """

    def __getitem__(self, idx: int) -> SpectralData:
        """Return the item at the given index"""
        return self.dataset[idx]
    
    
    def __len__(self) -> int:
        """Return the number of items in the dataset"""
        return len(self.dataset)    


    def __iter__(self) -> Any:
        """Return an iterator over the dataset"""
        return iter(self.dataset)
    

    def info(self) -> DataFrame:
        """
        TODO: 返回输出有多少条光谱，每条光谱的大小，每个光谱有多少类。
        FORMAT:
        CLASS, SUBCLASS, FLUX_SHAPE, WAVELENGTH_SHAPE
        """
        raise NotImplementedError("info method not implemented")


    def __str__(self) -> DataFrame:
        return self.info()

    def add_dataset(self, dirpath: str) -> NDArray:
        """
        TODO : 解析数据集文件夹，使用read_data函数， 读取数据集中的所有数据。
        以ArrayLike[SpectralData]的形式存储在self.__dataset中。并将这个数据集序列化到__dir_data_path中。
        参数：
        dirpath: str, 数据集的路径
        dirpath: 下面是*.fits.gz
        返回：
        无

        序列化的数据集文件格式：
        1. 数据集的路径
        __dir_data_path: str
        2. 数据集格式：NDArray[SpectralDataType](定义在SpectralData中)
        """
        # filenames = os.listdir(dirpath)
        # spectral_datas = np.empty(len(filenames), dtype=object)
        # for i, file_name in enumerate(filenames):
        #     with fits.open(dirpath + file_name) as hdulist:
        #         spectral_datas[i] = SpectralData(hdulist)
        # return spectral_datas
        filenames = os.listdir(dirpath)
    
        def load_fits(file_name):
            with fits.open(os.path.join(dirpath, file_name)) as hdulist:
                return SpectralData(hdulist)
    
        spectral_datas = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(load_fits, filenames)
            spectral_datas = list(results)
    
        return np.array(spectral_datas, dtype=object)
    

    def to_numpy(self) -> NDArray[Any]:
        """
        NDArray[SpectralDataType] : 返回一个numpy数组，Any是SpectralDataType类型。
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

