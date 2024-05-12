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
    def WAVELENGTH(self) -> NDArray:
        return self.data.WAVELENGTH[0]

    @property
    def SUBCLASS(self) -> str:
        return self.header["SUBCLASS"]

    @property
    def CLASS(self) -> str:
        return self.header["CLASS"]

    @property
    def OBSID(self) -> str:
        return self.header["OBSID"]


@dataclass
class SDSSSpectraData(SpectralData):
    overall_header: Header
    spectra: BinTableHDU
    objinfo: BinTableHDU
    spzline: BinTableHDU

    def __init__(self, hdul: HDUList):
        # super: SpectralData
        self.hdul = hdul
        self.overall_header = hdul[0].header
        self.spectra = hdul[1]
        self.objinfo = hdul[2]
        self.spzline = hdul[3]

    @property
    def FLUX(self) -> NDArray:
        return self.spectra.data.flux

    @property
    def WAVELENGTH(self) -> NDArray:
        # 转换为线性波长,两位小数
        return 10**self.spectra.data.loglam

    @property
    def CLASS(self) -> str:
        return self.objinfo.data.CLASS[0]

    @property
    def OBSID(self) -> str:
        return self.objinfo.data.SPECOBJID[0]

    @property
    def SUBCLASS(self) -> str:
        return self.objinfo.data.SUBCLASS[0]
