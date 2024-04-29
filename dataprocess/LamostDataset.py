import os
from astropy.io import fits

from .SpectralData import SpectralData
from .Dataset import Dataset


class LamostDataset(Dataset):
    def read_data(self, path: str) -> SpectralData:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")

        with fits.open(path) as hdulist:
            header = hdulist[0].header
            fits_data = hdulist[1].data
            
        return SpectralData(header, fits_data)
