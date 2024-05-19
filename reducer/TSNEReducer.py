import os

from openTSNE import TSNE
import numpy as np

from .Reducer import Reducer
import dataprocess.DataProcess as dp
from .util import get_data_from_dataset_index
from .util import get_save_name
from .util import get_data2d
from reducer.ReduceData import ReduceData


class TSNEReducer(Reducer):
    def __init__(
        self, dimension: int, perplexity: int, learning_rate: int, n_iter: int
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.hyperparameters = {
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "n_iter": n_iter,
        }
        self.reducer = TSNE(
            n_components=dimension,
            n_jobs=-1,
            **self.hyperparameters,
        )

    def reduce(self, dataset_index: str) -> ReduceData:
        """
        实现TSNE降维
        将降维结果保存在result_dir中
        返回ReduceData对象
        """

        save_name = get_save_name(
            "TSNE",
            {
                "n_components": self.dimension,
                "perplexity": self.hyperparameters["perplexity"],
                "learning_rate": self.hyperparameters["learning_rate"],
                "n_iter": self.hyperparameters["n_iter"],
            },
        )

        if os.path.exists(self.result_dir + dataset_index + "/" + save_name + ".npy"):

            result = np.load(
                self.result_dir + dataset_index + "/" + save_name + ".npy",
                allow_pickle=True,
            )
            reduce_data = ReduceData.from_numpy(*result)
            reduce_data.info = [dataset_index, save_name]
            return reduce_data

        data, classes, subclasses, obsid = get_data_from_dataset_index(dataset_index)

        reduce_data = self.reducer.fit(data)

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
        result.info = [dataset_index, save_name]

        return result
