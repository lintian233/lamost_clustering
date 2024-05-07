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
        self.all_result = ''

    @abstractmethod
    def reduce(self, dataset_index: str) -> ReduceData:
        """
        根据数据集的索引进行降维
        将降维结果保存在result_dir中
        返回ReduceData对象
        此方法需要在子类中实现
        """

        pass

    def info_result(self):
        """
        读写result_dir下所有的降维结果
        将结果打印并返回

        返回：
        List[List[int, str]]
        每个元素是一个列表
        列表中包含了一个降维结果的索引和对应文件名
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

        combineed_df = pd.concat([PCA_df, TSNE_df, UMAP_df], ignore_index=True)

        print(combineed_df.to_string(index=False))

        self.all_result = PCA_files + TSNE_files + UMAP_files

        return combineed_df

    def get_result(self, index) -> ReduceData:
        """
        根据索引获取降维结果
        返回对应降维结果的ReduceData对象
        """

        for item in self.all_result:
            if item[0] == index:
                return get_reduce_data(item[1])
