import os
import numpy as np
from astropy.io import fits

from .SpectralData import SpectralData
from .Dataset import Dataset

class LamostDataset(Dataset):


    def read_data(self, path: str) -> SpectralData:
        with fits.open(path) as hdulist:
            f = hdulist[1].data[0][0][:3900]
            f = (f - np.min(f)) / (np.max(f) - np.min(f))
            wave = hdulist[0].data[0][2][:3900]    
            
            subclass = hdulist[0].header['SUBCLASS'][0]
            scls = 'M' if subclass in ['d', 'g', 's'] else subclass

        #f is the flux of the spectrum np array
    
