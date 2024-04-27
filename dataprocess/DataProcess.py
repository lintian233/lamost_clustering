from pandas import DataFrame

from .Dataset import Dataset

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
