import os
import numpy as np
from astropy.io import fits

from .SpectralData import SpectralData
from .Dataset import Dataset


class LamostDataset(Dataset):
    def read_data(self, path: str) -> SpectralData:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")

        with fits.open(path) as hdulist:
            name = hdulist[0].header["OBSID"]
            class_name = hdulist[0].header["CLASS"]
            subclass = hdulist[0].header["SUBCLASS"]
            f = hdulist[1].data[0][0]
            wave = hdulist[1].data[0][2]

        return SpectralData(name, f, wave, class_name, subclass)
