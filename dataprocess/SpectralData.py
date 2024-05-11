import numpy as np
from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray
from numpy import dtype
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.header import Header
from astropy.io.fits import HDUList, PrimaryHDU
from astropy.io.fits.hdu.table import BinTableHDU


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
    hdul: HDUList
    header: Header
    data: FITS_rec

    def __init__(self, hdul: HDUList):
        self.hdul = hdul
        self.header = hdul[0].header
        self.data = hdul[1].data

    @classmethod
    def from_numpy(cls, data: NDArray[Any]) -> "SpectralData":
        header = Header.fromstring(data[0])
        fits_data = FITS_rec(data[1])
        hdu_header = PrimaryHDU(header=header)
        hdu_fitsrec = BinTableHDU(fits_data)
        hdul = HDUList([hdu_header, hdu_fitsrec])
        return cls(hdul)

    def __getitem__(self, key: str) -> Any:
        return self.header[key]


@dataclass
class LamostSpectraData(SpectralData):
    def __init__(self, hdul: HDUList):
        super().__init__(hdul)

    @property
    def FLUX(self) -> NDArray:
        return self.data.FLUX[0]

    @property
    def SUBCLASS(self) -> str:
        return self.header["SUBCLASS"]

    @property
    def CLASS(self) -> str:
        return self.header["CLASS"]

    @property
    def WAVELENGTH(self) -> NDArray:
        return self.data.WAVELENGTH[0]
