import numpy as np

from numpy.typing import NDArray
from numpy import dtype

from typing import Any, TypeVar, Generic
from dataclasses import dataclass

SpectralDataType = dtype(
    [
        ("name", "U10", (1,)),
        ("flux", np.float64, (6000,)),
        ("wavelength", np.float64, (6000,)),
        ("class", "U10", (1,)),
        ("subclass", "U10", (1,)),
    ]
)


@dataclass
class SpectralData:
    data: NDArray[Any] 
    name: str
    class_name: str
    subclass: str

    def __init__(self, name, flux, wav, cls, subcls):
        self.data = np.zeros(1, dtype=SpectralDataType)[0]
        self.data["name"] = name

        if len(flux) > 6000:
            raise ValueError("Flux data too long")
        if len(wav) > 6000:
            raise ValueError("Wavelength data too long")
        self.data["flux"][:] = -1
        self.data["flux"][: len(flux)] = flux
        self.data["wavelength"][:] = -1
        self.data["wavelength"][: len(wav)] = wav

        self.data["class"] = cls
        self.data["subclass"] = subcls

        self.name = name
        self.class_name = cls
        self.subclass = subcls

    @classmethod
    def from_numpy(cls, data: NDArray[Any]) -> "SpectralData":
        return cls(
            data["name"],
            data["flux"],
            data["wavelength"],
            data["class"],
            data["subclass"],
        )
    
    def __getitem__(self, key):
        key_list = self.data.dtype.names
        if key not in key_list:
            raise KeyError(f"{key} not found in {key_list}")
        return self.data[key]