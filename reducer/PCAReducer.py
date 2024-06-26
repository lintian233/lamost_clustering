import os
from typing import Any

from sklearn.decomposition import PCA
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .Reducer import Reducer
import dataprocess.DataProcess as dp
from .util import get_data_from_dataset_index, get_data_from_dataset
from .util import get_data2d
from .util import get_save_name
from reducer.ReduceData import ReduceData
from dataprocess.Dataset import Dataset
from dataprocess.LoadedDatasetManager import LoadedDatasetManager
from dataprocess.util import print_bule, print_red, print_green


class PCAReducer(Reducer):
    def __init__(self, dimension: int, **args) -> None:
        super().__init__()
        self.dimension = dimension
        self.reducer = PCA
        self.hyperparameters = {**args}

    def reduce(self, dataset: Dataset) -> ReduceData:
        """
        实现PCA降维
        将降维结果保存在result_dir中
        返回ReduceData对象
        """

        save_name = get_save_name(
            "PCA", {"n_components": self.dimension, **self.hyperparameters}
        )

        dataset_nl = dataset.name.split("-")
        dataset_index = dataset_nl[0] + "-" + dataset_nl[1]

        if os.path.exists(self.result_dir + dataset_index + "/" + save_name + ".npy"):
            print_green("PCA result exists.Load from cache.")
            result = np.load(
                self.result_dir + dataset_index + "/" + save_name + ".npy",
                allow_pickle=True,
            )
            reduce_data = ReduceData.from_numpy(*result)
            reduce_data.info = [dataset_index, save_name]
            return reduce_data

        data, classes, subclasses, obsid = get_data_from_dataset(dataset)

        print_bule("PCA reduce datand")
        reduce_data = self.reducer(
            n_components=self.dimension, **self.hyperparameters
        ).fit_transform(data)

        print_bule("PCA reduce data2d")
        data2d = get_data2d(dataset)

        result = np.zeros(5, dtype=object)
        result[0] = data2d
        result[1] = reduce_data
        result[2] = classes
        result[3] = subclasses
        result[4] = obsid

        if not os.path.exists(self.result_dir + dataset_index):
            os.makedirs(self.result_dir + dataset_index)

        np.save(self.result_dir + dataset_index + "/" + save_name, result)

        result = ReduceData.from_numpy(*result)
        result.info = [dataset_index, save_name]

        print_green(f"PCA reduce {dataset_index} finished.")
        return result
