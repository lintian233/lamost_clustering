import numpy as np

from typing import Any
from dataclasses import dataclass

SpectralDataType = [
    ('name', "U10", (1,)),
    ('flux', np.float64, (3700,)),
    ('wavelength', np.float64, (3700,))
    ('class', "U10", (1,)),
    ('subclass', "U10", (1,))
]

@dataclass
class SpectralData:
    data: NDArray[Any]
    name: str

    def __init__(self,data: NDArray[Any] = None):
        if data is None:
            self.data = np.zeros(1, dtype=SpectralDataType)
        else:
            # Check if data is a numpy array
            if not isinstance(data, np.ndarray):
                raise TypeError("data must be a numpy array")
            # Check if data has the correct dtype
            if data.dtype != SpectralDataType:
                raise TypeError("data must have the correct dtype")
            self.data = data






