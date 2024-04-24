
from .SpectralData import SpectralData
from .Dataset import Dataset

class LamostDataset(Dataset):

    def __init__(self, dirpath: str) -> None:
        super().__init__(dirpath)

    def read_data(self, path: str) -> SpectralData:
        raise NotImplementedError("read_data method not implemented")
     