from typing import Dict
from .Dataset import Dataset


class LoadedDatasetManager:
    loaded_datasets: Dict[str, Dataset]
    _instance = None

    def __init__(self) -> None:
        self.loaded_datasets = {}

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add(self, dataset: Dataset) -> None:
        dataset_index = dataset.name.split("-")[0] + "-" + dataset.name.split("-")[1]
        self.loaded_datasets[dataset_index] = dataset

    def get(self, dataset_index: str) -> Dataset:
        if dataset_index not in self.loaded_datasets:
            return None
        return self.loaded_datasets[dataset_index]
