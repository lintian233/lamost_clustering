import os
import glob
import re


import numpy as np
from pandas import DataFrame
from astropy.io import fits


from config.config import DATASETBASEPATH
from .Dataset import Dataset
from .SpectralData import SpectralData, LamostSpectraData, SDSSSpectraData
from .LamostDataset import LamostDataset
from .SDSSDataset import SDSSDataset
from .StdDataset import StdDataset
from .util import find_dataset_path, generate_dataset_name
from .util import init_lamost_dataset, init_sdss_dataset

"""
/Data/ 
    - LamostDataset/
                    -LamostDataset-XXX-SN10000-Q1111-G10000-S9000.npy
    - SDSSDataset/
                    -SDSSDataset-XXX-SN10000-Q1111-G10000-S9000.npy
"""

DATASET_DICT = {
    "LamostDataset": LamostDataset,
    "SDSSDataset": SDSSDataset,
    "StdDataset": StdDataset,
}


class DataProcess:
    @staticmethod
    def get_subclass_dataset(dataset: Dataset, subclass: str) -> Dataset:
        """
        从数据集中获取子类数据集
        参数：
        dataset: Dataset, 数据集
        subclass: str, 子类名称
        返回：
        Dataset, 子类数据集
        """
        subdataset = dataset.__class__()
        class_array = np.array([item.SUBCLASS for item in dataset])
        class_name_set = np.unique(class_array)
        if subclass not in class_name_set:
            raise ValueError(f"Class {subclass} not found in dataset")

        index = np.where(class_array == subclass)[0]
        sublist = dataset[index]

        subdataset.dataset = sublist
        labels = np.array([subclass] * len(sublist))
        subdataset.name = generate_dataset_name(
            subdataset.__class__.__name__,
            subdataset.dir_base_path,
            labels,
        )
        return subdataset

    @staticmethod
    def get_class_dataset(dataset: Dataset, class_name: str) -> Dataset:
        """
        参数：
        dataset: Dataset, 数据集
        class: str, 类名称
        返回：
        Dataset, 类数据集
        """
        subdataset = dataset.__class__()
        class_array = np.array([item.CLASS for item in dataset])
        class_name_set = np.unique(class_array)
        if class_name not in class_name_set:
            raise ValueError(f"Class {class_name} not found in dataset")

        index = np.where(class_array == class_name)[0]

        sublist = dataset[index]
        subdataset.dataset = sublist
        labels = np.array([class_name] * len(sublist))
        subdataset.name = generate_dataset_name(
            subdataset.__class__.__name__,
            subdataset.dir_base_path,
            labels,
        )
        return subdataset

    @staticmethod
    def info_dataset(dataset_index: str = None) -> DataFrame:
        """
        看一下本地目录文件有没有dataset，有把基本信息读出来。
        INDEX, DATASET_NAME, NUM_SPECTRA, QSO, GALAXY, STAR
        """
        raise NotImplementedError("info_dataset method not implemented")

    @staticmethod
    def load_dataset(dataset_index: str) -> Dataset:
        """
        根据dataset_index加载数据集。

        参数：
        dataset_index: str, 数据集的索引

        返回：
        Dataset, 数据集

        示例：
        >>> load_dataset("LamostDataset-000")
        <LamostDataset>
        >>> load_dataset("SDSSDataset-000")
        <SDSSDataset>
        >>> load_dataset("NonsenDataset-000")
        ValueError: 'DatsetType NonsenDataset not found'
        """
        telescope = dataset_index.split("-")[0]
        if telescope not in DATASET_DICT.keys():
            raise ValueError(f"DatsetType {telescope} not found")

        dataset: Dataset = DATASET_DICT.get(telescope)()

        dataset_path = find_dataset_path(dataset_index)
        spectrum_data = []

        hdulist = fits.open(dataset_path)

        match telescope:
            case "LamostDataset":
                spectrum_data = init_lamost_dataset(hdulist)
            case "SDSSDataset":
                spectrum_data = init_sdss_dataset(hdulist)
            case "StdDataset":
                raise NotImplementedError("StdDataset not implemented")
                # init_std_dataset(hdulist)

        dataset.dataset = spectrum_data
        dataset.name = dataset_path.split("\\")[-1].split(".")[0]
        return dataset

    @staticmethod
    def list_datasets() -> DataFrame:
        """
        返回所有数据集的列表

        返回：
        DataFrame, 数据集列表
        """
        base_path = DATASETBASEPATH
        if base_path[-1] != "/":
            base_path += "/"

        dataset = []
        dataset_dirs = glob.glob(base_path + "*Dataset/")
        for item in dataset_dirs:
            current = glob.glob(item + "*.fits")
            dataset.append(current)

        pattern = r"\\([A-Za-z]+-\d+)-SN(\d+)-STAR(\d+)-QSO(\d+)-GALAXY(\d+)"

        datasets_info = []
        for item in dataset:
            for i in item:
                info = []
                match = re.search(pattern, i)
                if match:
                    info.append(match.group(1))
                    info.append(match.group(2))
                    info.append(match.group(3))
                    info.append(match.group(4))
                    info.append(match.group(5))
                datasets_info.append(info)

        INFO = DataFrame(
            datasets_info,
            columns=["DATASET_NAME", "NUM_SPECTRA", "STAR", "QSO", "GALAXY"],
        )

        return INFO

    @staticmethod
    def Preprocessing(dataset: Dataset) -> StdDataset:
        """
        数据预处理
        """
        raise NotImplementedError("Preprocessing method not implemented")
