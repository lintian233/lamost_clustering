import numpy as np

from numpy.typing import NDArray
from numpy import dtype

from typing import Any, TypeVar, Generic
from dataclasses import dataclass

SpectralDataType = dtype([
    ('name', "U10", (1,)),
    ('flux', np.float64, (3700,)),
    ('wavelength', np.float64, (3700,)),
    ('class', "U10", (1,)),
    ('subclass', "U10", (1,))
])

@dataclass
class SpectralData:
    data: NDArray[Any]
    name: str
    '''
    TODO: 这里要改一下，fits文件读取后是hdulist类型，并且无法转换成ndarray
    '''
    def __init__(self,data: NDArray[Any] = None) -> None:
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





