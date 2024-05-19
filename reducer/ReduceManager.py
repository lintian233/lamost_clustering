import os
import pandas as pd
from typing import Tuple

from reducer.ReduceData import ReduceData
from reducer.util import get_reduce_data
from config.config import REDUCEDATAPATH


class ReduceManager:
    @staticmethod
    def info_result():
        """
        读写result_dir下所有的降维结果
        将结果打印并返回

        返回：
        List[List[int, str]]
        每个元素是一个列表
        列表中包含了一个降维结果的索引和对应文件名
        """
        result_dir = REDUCEDATAPATH
        all_result = []
        PCA_data = []
        PCA_files = []
        TSNE_data = []
        TSNE_files = []
        UMAP_data = []
        UMAP_files = []
        index = 0
        for dir in os.listdir(result_dir):
            for file in os.listdir(result_dir + dir):
                if file.startswith("PCA"):
                    splits = file[:-4].split("-")
                    new_splits = [splits[i] for i in range(0, len(splits), 2)]
                    PCA_data.append([index, dir] + new_splits)
                    PCA_files.append([index, result_dir + dir + "/" + file])
                elif file.startswith("TSNE"):
                    splits = file[:-4].split("-")
                    new_splits = [splits[i] for i in range(0, len(splits), 2)]
                    TSNE_data.append([index, dir] + new_splits)
                    TSNE_files.append([index, result_dir + dir + "/" + file])
                elif file.startswith("UMAP"):
                    splits = file[:-4].split("-")
                    new_splits = [splits[i] for i in range(0, len(splits), 2)]
                    UMAP_data.append([index, dir] + new_splits)
                    UMAP_files.append([index, result_dir + dir + "/" + file])
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

        combined_df = pd.concat([PCA_df, TSNE_df, UMAP_df], ignore_index=True)

        print(combined_df.to_string(index=False))

        all_result = PCA_files + TSNE_files + UMAP_files

        return combined_df

    @staticmethod
    def get_result(index) -> Tuple[ReduceData, str]:
        """
        根据索引获取降维结果
        返回对应降维结果的ReduceData对象
        """
        result_dir = REDUCEDATAPATH
        all_result = []
        i = 0
        for dir in os.listdir(result_dir):
            for file in os.listdir(result_dir + dir):
                if (
                    file.startswith("PCA")
                    or file.startswith("TSNE")
                    or file.startswith("UMAP")
                ):
                    all_result.append([i, result_dir + dir + "/" + file])
                    i += 1
        for item in all_result:
            if item[0] == index:
                file_name = item[1].split("/")[-1].split(".")[0]
                return get_reduce_data(item[1]), file_name
