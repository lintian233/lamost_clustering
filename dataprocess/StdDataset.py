from typing import List

from .SpectralData import StdSpectralData
from .Dataset import Dataset


class StdDataset(Dataset):
    dataset: List[StdSpectralData]

    def add_dataset(self, dirpath: str) -> str:
        raise NotImplementedError("add_dataset method not implemented")

    def read_data(self, path: str) -> StdSpectralData:
        raise NotImplementedError("read_data method not implemented")

    def add_dataset_parallel(self, dirpath: str) -> str:
        raise NotImplementedError("add_dataset_parallel method not implemented")
