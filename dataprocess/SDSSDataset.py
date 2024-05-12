import os

from astropy.io import fits
from .SpectralData import SpectralData, SDSSSpectraData
from .Dataset import Dataset


class SDSSDataset(Dataset):
    def read_data(self, path: str) -> SpectralData:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")

        with fits.open(path) as hdulist:
            new_hdulist = fits.HDUList([hdu.copy() for hdu in hdulist[:4]])
            return SDSSSpectraData(new_hdulist)
