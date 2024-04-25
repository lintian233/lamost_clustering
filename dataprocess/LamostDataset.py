import os

from astropy.io import fits

from .SpectralData import SpectralData
from .Dataset import Dataset

class LamostDataset(Dataset):


    def read_data(self, path: str) -> SpectralData:

        raise NotImplementedError("read_data method not implemented")
    