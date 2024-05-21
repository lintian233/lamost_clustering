import os

from openTSNE import TSNE
import numpy as np

from .Reducer import Reducer
import dataprocess.DataProcess as dp
from .util import get_data_from_dataset_index, get_data_from_dataset
from .util import get_save_name
from .util import get_data2d
from reducer.ReduceData import ReduceData
from dataprocess.Dataset import Dataset
from dataprocess.LoadedDatasetManager import LoadedDatasetManager
from dataprocess.util import print_bule, print_red, print_green


class TSNEReducer(Reducer):
    def __init__(self, dimension: int, **args) -> None:
        super().__init__()
        self.dimension = dimension
        self.hyperparameters = {**args}
        self.reducer = TSNE(
            n_components=dimension,
            **self.hyperparameters,
        )

    def reduce(self, dataset: Dataset) -> ReduceData:
        """
        实现TSNE降维
        将降维结果保存在result_dir中
        返回ReduceData对象
        """

        save_name = get_save_name(
            "TSNE",
            {"n_components": self.dimension, **self.hyperparameters},
        )
        dataset_index = dataset.name.split("-")[0] + "-" + dataset.name.split("-")[1]

        if os.path.exists(self.result_dir + dataset_index + "/" + save_name + ".npy"):
            print_green("TSNE result exists.Load from cache.")

            result = np.load(
                self.result_dir + dataset_index + "/" + save_name + ".npy",
                allow_pickle=True,
            )
            reduce_data = ReduceData.from_numpy(*result)
            reduce_data.info = [dataset_index, save_name]
            return reduce_data

        data, classes, subclasses, obsid = get_data_from_dataset(dataset)

        print_bule("TSNE reduce datand")
        reduce_data = self.reducer.fit(data)
        print_bule("TSNE reduce data2d")
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
        print_green("TSNE reduce done.")
        return result
