import os

from astropy.io import fits

from .SpectralData import SpectralData, LamostSpectraData
from .Dataset import Dataset


class LamostDataset(Dataset):
    def read_data(self, path: str) -> SpectralData:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")

        with fits.open(path) as hdulist:
            new_hdulist = fits.HDUList([hdulist[0].copy(), hdulist[1].copy()])
            return LamostSpectraData(new_hdulist)
