import numpy as np

from numpy.typing import NDArray
from numpy import dtype
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.header import Header
from typing import Any, TypeVar, Generic
from dataclasses import dataclass

# SpectralDataType = dtype(
#     [
#         ("name", "U10", (1,)),
#         ("flux", np.float64, (6000,)),
#         ("wavelength", np.float64, (6000,)),
#         ("class", "U10", (1,)),
#         ("subclass", "U10", (1,)),
#     ]
# )

# LAMOST_DATA_TYPE
datatype = dtype(
    [
        ("FLUX", ">f4", (3909,)),  # 大端序32位浮点数
        ("IVAR", ">f4", (3909,)),
        ("WAVELENGTH", ">f4", (3909,)),
        ("ANDMASK", ">f4", (3909,)),
        ("ORMASK", ">f4", (3909,)),
        ("NORMALIZATION", ">f4", (3909,)),
    ]
)

SpectralDataType = dtype(
    [
        ("header", "U11520", ()),
        ("data", FITS_rec, ()),
    ]
)


@dataclass
class SpectralData:
    raw_data: NDArray[Any]
    header: Header
    data: FITS_rec

    def __init__(self, header: Header, data: FITS_rec):
        self.header = header
        self.data = data
        header_str = header.tostring()
        self.raw_data = np.array([(header_str, data)], dtype=SpectralDataType)[0]

    @classmethod
    def from_numpy(cls, data: NDArray[Any]) -> "SpectralData":
        header = Header.fromstring(data[0])
        fits_data = FITS_rec(data[1])
        return cls(header, fits_data)
