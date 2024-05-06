from sklearn.decomposition import PCA

import numpy as np
from numpy.typing import NDArray

from .Reducer import Reducer
import dataprocess.DataProcess as dp


class PCAReducer(Reducer):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.reducer = PCA

    def reduce(self, dataset_index: str) -> str:
        """
        实现PCA降维，将降维结果保存在result_dir中
        """
        dataset = dp.load_dataset(dataset_index)
        data = np.zeros((len(dataset), 3000))

        for i in range(len(dataset)):
            data[i] = dataset[i].data[0][0][:3000]

        reduce_data = self.reducer(n_components=self.dimension).fit_transform(data)

        return reduce_data
