import os
import time

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np

from astropy.io import fits
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler

from .SpectralData import SpectralData
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
        if isinstance(key, np.ndarray) or isinstance(key, list):
            return [self.dataset[i] for i in key]

        raise TypeError(f"Invalid argument type:{key.__class__.__name__}")

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
                for hdu in data.hdul:
                    hdulist.append(hdu)

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

    def add_dataset_parallel(self, dirpath: str) -> str:
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
        # print(f"Total files: {len(fits_path)}")
        # start_time = time.time()

        # 定义一个包装函数，用于读取数据并更新进度条
        def read_data(path):
            try:
                data = self.read_data(path)
                return data
            except Exception as e:
                print(f"read data Error: {e}, fits: {path.split('/')[-1]}")
                return None

        # 初始化进度条
        set_loky_pickler("dill")
        parallel = Parallel(n_jobs=-1, backend="loky")
        results = parallel(delayed(read_data)(path) for path in tqdm(fits_path))

        if None in results:
            results = [i for i in results if i is not None]

        self.dataset = list(results)
        # print(f"load data time: {time.time() - start_time}")

        labels_list = np.array([i.CLASS for i in self.dataset])

        dataset_name = generate_dataset_name(
            self.__class__.__name__, self.dir_base_path, labels_list
        )

        self.name = dataset_name
        save_path = self.dir_base_path + dataset_name + ".fits"

        # print("saving...")
        # start_time = time.time()
        with fits.HDUList() as hdulist:
            for data in self.dataset:
                for hdu in data.hdul:
                    hdulist.append(hdu)
                # hdulist.extend([data.hdul[i] for i in range(len(data.hdul))])

            hdulist.writeto(save_path, overwrite=True, output_verify="ignore")

        # print(f"Save time: {time.time() - start_time}")

        return save_path
