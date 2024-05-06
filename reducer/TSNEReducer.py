from sklearn.manifold import TSNE

from .Reducer import Reducer
import dataprocess.DataProcess as dp
import numpy as np


class TSNEReducer(Reducer):
    def __init__(
        self, dimension: int, perplexity: int, learning_rate: int, n_iter: int
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.reducer = TSNE
        self.hyperparameters = {
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "n_iter": n_iter,
        }

    def reduce(self, dataset_index: str) -> str:
        """
        实现TSNE降维，将降维结果保存在result_dir中
        """
        dataset = dp.load_dataset(dataset_index)
        data = np.zeros((len(dataset), 3000))
        for i in range(len(dataset)):
            data[i] = dataset[i].data[0][0][:3000]
        reduce_data = self.reducer(
            n_components=self.dimension,
            perplexity=self.hyperparameters["perplexity"],
            learning_rate=self.hyperparameters["learning_rate"],
            n_iter=self.hyperparameters["n_iter"],
        ).fit_transform(data)
        return reduce_data
