from typing import List

from .SpectralData import StdSpectraData
from .Dataset import Dataset


class StdDataset(Dataset):
    dataset: List[StdSpectraData]

    def add_dataset(self, dirpath: str) -> str:
        raise NotImplementedError("add_dataset method not implemented")

    def read_data(self, path: str) -> StdSpectraData:
        raise NotImplementedError("read_data method not implemented")

    def add_dataset_parallel(self, dirpath: str) -> str:
        raise NotImplementedError("add_dataset_parallel method not implemented")
