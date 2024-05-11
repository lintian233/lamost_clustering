from abc import ABC, abstractmethod
import os
from typing import Any, List

import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ThreadPoolExecutor
from astropy.io import fits

from .SpectralData import SpectralData, SpectralDataType
from .util import parser_fits_path, generate_dataset_name
from config.config import DATASETBASEPATH


class Dataset(ABC):
    dataset: List[SpectralData]
    dir_base_path = DATASETBASEPATH
    name: str

    def __init__(self) -> None:
        if not os.path.exists(self.dir_base_path):
            os.makedirs(self.dir_base_path)

        if self.dir_base_path[-1] != "/":
            self.dir_base_path += "/"

        # dir_base_path = DATASETBASEPATH/ClassName/
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

    def add_dataset(self, dirpath: str) -> str:
        """
        从给定目录中读取数据集，并将其添加到数据集中。

        参数：
        dirpath: str, 数据集的路径

        返回：
        str, 数据集的保存路径

        示例：
        Data/Fits/中有多个FITS文件
        >>> add_dataset("Data/Fits/")
        'DATSETBASEPATH/XXXDataset/XXXDataset-XXX-SNXXX-STARXXX-QSOXXX-GALAXYXXX.fits'
        """
        fits_path = parser_fits_path(dirpath)

        with ThreadPoolExecutor() as executor:
            results = executor.map(self.read_data, fits_path)

        self.dataset = list(results)

        labels_list = np.array([i.CLASS for i in self.dataset])

        dataset_name = generate_dataset_name(
            self.__class__.__name__, self.dir_base_path, labels_list
        )

        self.name = dataset_name
        save_path = self.dir_base_path + dataset_name + ".fits"

        with fits.HDUList() as hdulist:
            for data in self.dataset:
                hdulist.append(data.hdul[0])
                hdulist.append(data.hdul[1])

            hdulist.writeto(save_path, overwrite=True, output_verify="ignore")

        return save_path

    @abstractmethod
    def read_data(self, path: str) -> SpectralData:
        """
        派生类需要实现的方法,根据FITS文件路径读取一条光谱数据。
        参数：
        path: str, 数据集的路径
        返回：
        一个SpectralData对象。
        """
        pass
