from abc import ABC, abstractmethod
from typing import Any
import os

import pandas as pd
import numpy as np

from config.config import REDUCEDATAPATH
from reducer.ReduceData import ReduceData
from .util import get_reduce_data


"""

"""


class Reducer(ABC):
    index: str
    reducer: Any
    result_dir: str
    hyperparameters: dict[str, Any]
    dimension: int

    def __init__(self) -> None:
        self.result_dir = REDUCEDATAPATH
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.all_result = self.info_result()

    @abstractmethod
    def reduce(self, dataset_index: str) -> str:
        """
        TODO: 将当前数据集以当前的超参数降维至所需维度，
        然后将结果保存在result_dir中。
        完成后返回文件路径
        """
        pass

    def info_result(self):
        """
        TODO :返回一个所有降维结果的表，
        给出了结果目录下的所有当前方法(PCA/UMAP)->(当前类名)的一个超参数所对应的数据表：
        """
        PCA_data = []
        PCA_files = []
        TSNE_data = []
        TSNE_files = []
        UMAP_data = []
        UMAP_files = []
        index = 0
        for dir in os.listdir(self.result_dir):
            for file in os.listdir(self.result_dir + dir):
                if file.startswith("PCA"):
                    splits = file[:-4].split("-")
                    new_splits = []
                    for i in range(0, len(splits)):
                        if i % 2 == 0:
                            new_splits.append(splits[i])
                    PCA_data.append([index, dir] + new_splits)
                    PCA_files.append([index, self.result_dir + dir + "/" + file])
                elif file.startswith("TSNE"):
                    splits = file[:-4].split("-")
                    new_splits = []
                    for i in range(0, len(splits)):
                        if i % 2 == 0:
                            new_splits.append(splits[i])
                    TSNE_data.append([index, dir] + new_splits)
                    TSNE_files.append([index, self.result_dir + dir + "/" + file])
                elif file.startswith("UMAP"):
                    splits = file[:-4].split("-")
                    new_splits = []
                    for i in range(0, len(splits)):
                        if i % 2 == 0:
                            new_splits.append(splits[i])
                    UMAP_data.append([index, dir] + new_splits)
                    UMAP_files.append([index, self.result_dir + dir + "/" + file])
                index += 1

        PCA_df = pd.DataFrame(
            PCA_data, columns=["index", "Dataset", "method", "n_components"]
        )
        TSNE_df = pd.DataFrame(
            TSNE_data,
            columns=[
                "index",
                "Dataset",
                "method",
                "n_components",
                "perplexity",
                "learning_rate",
                "n_iter",
            ],
        )
        UMAP_df = pd.DataFrame(
            UMAP_data,
            columns=[
                "index",
                "Dataset",
                "method",
                "n_components",
                "n_neighbors",
                "metric",
                "learning_rate",
                "min_dist",
            ],
        )
        print("PCA redeuce data:")
        print(PCA_df.to_string(index=False))
        print("TSNE reduce data:")
        print(TSNE_df.to_string(index=False))
        print("UMAP reduce data:")
        print(UMAP_df.to_string(index=False))
        return PCA_files + TSNE_files + UMAP_files

    def get_result(self, index) -> ReduceData:
        """
        TODO :给定名称，在结果目录文件夹内检索并返回ReduceData
        """
        for item in self.all_result:
            if item[0] == index:
                return get_reduce_data(item[1])
