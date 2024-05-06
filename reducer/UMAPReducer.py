import umap
import numpy as np

from .Reducer import Reducer
import dataprocess.DataProcess as dp


class UMAPReducer(Reducer):
    def __init__(
        self,
        dimension: int,
        n_neighbors: int,
        metric: str,
        learning_rate: float,
        min_dist: float,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.reducer = umap.UMAP
        self.hyperparameters = {
            "n_neighbors": n_neighbors,
            "metric": metric,
            "learning_rate": learning_rate,
            "min_dist": min_dist,
        }

    def reduce(self, dataset_index: str) -> str:
        """
        实现UMAP降维，将降维结果保存在result_dir中
        """
        dataset = dp.load_dataset(dataset_index)
        data = np.zeros((len(dataset), 3000))
        for i in range(len(dataset)):
            data[i] = dataset[i].data[0][0][:3000]
        reduce_data = self.reducer(
            n_components=self.dimension,
            n_neighbors=self.hyperparameters["n_neighbors"],
            metric=self.hyperparameters["metric"],
            learning_rate=self.hyperparameters["learning_rate"],
            min_dist=self.hyperparameters["min_dist"],
        ).fit_transform(data)
        return reduce_data
