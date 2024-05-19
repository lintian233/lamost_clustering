import os
import pandas as pd
from pandas import DataFrame
import numpy as np
from .ClusterData import ClusterData
from config.config import CLUSTER_DATA_PATH


class ClusterManager:

    @staticmethod
    def info() -> DataFrame:
        result_dir = CLUSTER_DATA_PATH
        index = 0

        def process_file(root, file, index):
            if file.startswith(("Spectral", "HDBSCAN")):
                method = file.split("-")[0]
                splits = file[:-4].split("-")
                key = [splits[i] for i in range(1, len(splits), 2)]
                value = [splits[i] for i in range(2, len(splits), 2)]
                return {
                    "index": index,
                    "Dataset": os.path.basename(root),
                    "method": method,
                    **dict(zip(key, value)),
                }
            return None

        files_to_process = [
            (root, file, index)
            for root, _, files in os.walk(result_dir)
            for file in files
        ]

        all_result = []
        for root, file, idx in files_to_process:
            result = process_file(root, file, index)
            all_result.append(result) if result is not None else None
            index += 1

        return pd.DataFrame(all_result)

    @staticmethod
    def get_cluster_data(index: int) -> ClusterData:
        result_dir = CLUSTER_DATA_PATH

        files_to_process = [
            (root, file)
            for root, _, files in os.walk(result_dir)
            for file in files
            if file.startswith(("Spectral", "HDBSCAN"))
        ]

        for root, file in files_to_process:
            if index == 0:
                file_path = os.path.join(root, file)
                return ClusterData.from_numpy(*np.load(file_path, allow_pickle=True))
            index -= 1

        return None
