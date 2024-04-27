import os
import glob
import re

from pandas import DataFrame


from .Dataset import Dataset
from config.config import DATASETBASEPATH

"""
/Data/
     - Dataset/
            - Lamost/
                    - XXX-SN10000-Q1111-G10000-S9000.npy
            - SDSS/
"""


class DataProcess:
    @staticmethod
    def get_subclass_dataset(dataset: Dataset, subclass: str) -> Dataset:
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
    def get_class_data(dataset: Dataset, _class: str) -> Dataset:
        """
        TODO :从数据集中获取类数据集
        参数：
        dataset: Dataset, 数据集
        class: str, 类名称
        返回：
        Dataset, 类数据集
        """
        raise NotImplementedError("get_class_data method not implemented")

    @staticmethod
    def info_dataset(dataset_name: str = None) -> DataFrame:
        """
        看一下本地目录文件有没有dataset，有把基本信息读出来。
        INDEX, DATASET_NAME, NUM_SPECTRA, QSO, GALAXY, STAR
        """
        raise NotImplementedError("info_dataset method not implemented")

    @staticmethod
    def load_dataset(dataset_index: str) -> Dataset:
        """
        加载数据集
        参数：
        dataset_index: str, 数据集的索引
        返回：
        Dataset, 数据集
        """
        raise NotImplementedError("load_dataset method not implemented")
    
    @staticmethod
    def list_datasets() -> DataFrame:
        """
        返回所有数据集的列表
        """
        base_path = DATASETBASEPATH
        if base_path[-1] != "/":
            base_path += "/"

        dataset = []
        dataset_dirs = glob.glob(base_path + "*Dataset/")
        for item in dataset_dirs:
            current = glob.glob(item + "*.npy")
            dataset.append(current)
        
        pattern = r'\\([A-Za-z]+-\d+)-SN(\d+)-STAR(\d+)-QSO(\d+)-GALAXY(\d+)'

        datasets_info = []
        for item in dataset:
            info = []
            for i in item:
                match = re.search(pattern, i)
                if match:
                    info.append(match.group(1))
                    info.append(match.group(2))
                    info.append(match.group(3))
                    info.append(match.group(4))
                    info.append(match.group(5))
            datasets_info.append(info)
        
        INFO =  DataFrame(datasets_info, columns=["DATASET_NAME", "NUM_SPECTRA", "STAR", "QSO", "GALAXY"])
        
        return INFO
        
        




