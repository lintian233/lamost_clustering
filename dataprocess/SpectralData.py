import numpy as np

from numpy.typing import NDArray
from numpy import dtype

from typing import Any, TypeVar, Generic
from dataclasses import dataclass

SpectralDataType = dtype([
    ('name', "U10", (1,)),
    ('flux', np.float64, (6000,)),
    ('wavelength', np.float64, (6000,)),
    ('class', "U10", (1,)),
    ('subclass', "U10", (1,))
])

@dataclass
class SpectralData:
    data: NDArray[Any]
    name: str

    def __init__(self, name, flux, wav, cls, subcls):
        self.data = np.zeros(1, dtype=SpectralDataType)
        self.data["name"] = name
        
        if len(flux) > 6000:
            raise ValueError("Flux data too long")
        if len(wav) > 6000:
            raise ValueError("Wavelength data too long")
        self.data["flux"][0][:] = -1
        self.data["flux"][0][:len(flux)] = flux
        self.data["wavelength"][0][:] = -1
        self.data["wavelength"][0][:len(wav)] = wav

        self.data["class"] = cls
        self.data["subclass"] = subcls
        
        self.name = name

    @classmethod
    def from_spectral_data(cls, data: NDArray[Any]) -> 'SpectralData':
        return cls(data["name"], data["flux"][0], data["wavelength"][0], data["class"], data["subclass"]) 
           



