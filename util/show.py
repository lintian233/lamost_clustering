import numpy as np
import matplotlib.pyplot as plt


from astropy.io.fits.header import Header
from astropy.io.fits.fitsrec import FITS_rec


from dataprocess import SpectralData
from reducer import ReduceData
from cluster import ClusterData


def show_spectraldata(data: SpectralData) -> None:
    header: Header = data.header
    data: FITS_rec = data.data

    flux = data.FLUX[0]
    wavelength = data.WAVELENGTH[0]

    name = header["OBSID"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(wavelength[0], wavelength[-1])

    ax.plot(wavelength, flux, color="black", linewidth=0.41)
    ax.set_xlabel("Wavelength(Å)")
    ax.set_ylabel("Flux")
    ax.set_title(f"spectral data - {name} ")
    # ax2.xaxis.set_tick_params(labelbottom=False)  # Hide x-axis values
    # ax2.legend(loc="upper right")
    # Adjust subplots to fit the figure a
    plt.tight_layout()
    plt.show()


def show_reduce_data(data: ReduceData) -> None:
    raise NotImplementedError("Show_ReduceData method not implemented")


def show_cluster_data(data: ClusterData) -> None:
    raise NotImplementedError("Show_ClusterData method not implemented")
