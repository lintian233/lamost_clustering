import os
from typing import Any

from sklearn.decomposition import PCA
import numpy as np
from numpy.typing import NDArray

from .Reducer import Reducer
import dataprocess.DataProcess as dp
from .util import get_data_from_dataset_index
from .util import get_data2d
from .util import get_save_name
from reducer.ReduceData import ReduceData


class PCAReducer(Reducer):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.reducer = PCA

    def reduce(self, dataset_index: str) -> ReduceData:
        """
        实现PCA降维
        将降维结果保存在result_dir中
        返回ReduceData对象
        """

        save_name = get_save_name("PCA", {"n_components": self.dimension})

        if os.path.exists(self.result_dir + dataset_index + "/" + save_name + ".npy"):

            result = np.load(
                self.result_dir + dataset_index + "/" + save_name + ".npy",
                allow_pickle=True,
            )
            ReduceData.from_numpy(*result)

            return ReduceData

        data, classes, subclasses, obsid = get_data_from_dataset_index(dataset_index)

        reduce_data = self.reducer(n_components=self.dimension).fit_transform(data)

        data2d = get_data2d(dataset_index)

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

        return result
