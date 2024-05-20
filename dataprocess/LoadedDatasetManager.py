from typing import Dict
from .Dataset import Dataset


class LoadedDatasetManager:
    cache: Dict[str, Dataset]
    _instance = None

    def __init__(self) -> None:
        self.cache = {}

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add(self, index: str, dataset: Dataset) -> None:
        self.cache[index] = dataset

    def get(self, dataset_index: str) -> Dataset:
        if dataset_index not in self.cache:
            return None
        return self.cache[dataset_index]

    def get_index(self, dataset: Dataset) -> str:
        for index, data in self.cache.items():
            if data == dataset:
                return index
        raise ValueError("Dataset not found in cache")
    