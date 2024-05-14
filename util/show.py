import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from astropy.io.fits.header import Header
from astropy.io.fits.fitsrec import FITS_rec


from dataprocess import SpectralData
from dataprocess.SpectralData import LamostSpectraData, SDSSSpectraData
from reducer import ReduceData
from cluster import ClusterData


def show_spectraldata(data: SpectralData) -> None:
    telescope = (
        "LAMOST"
        if isinstance(data, LamostSpectraData)
        else "SDSS" if isinstance(data, SDSSSpectraData) else "Other"
    )
    flux = data.FLUX
    wavelength = data.WAVELENGTH
    name = data.OBSID
    class_name = data.CLASS
    sub_class_name = data.SUBCLASS

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(wavelength[0], wavelength[-1])

    ax.plot(wavelength, flux, color="black", linewidth=0.41)
    ax.set_xlabel("Wavelength(Å)")
    ax.set_ylabel("Flux")
    ax.set_title(f"{telescope}-{name}-{class_name}-{sub_class_name}")
    # ax2.xaxis.set_tick_params(labelbottom=False)  # Hide x-axis values
    # ax2.legend(loc="upper right")
    # Adjust subplots to fit the figure a
    plt.tight_layout()
    plt.show()


def show_reduce_data(reduce_data: ReduceData, mode="overlay", label="class") -> None:
    match mode:
        case "overlay":
            show_reduce_data_overlay(reduce_data, label)
        case "separate":
            show_reduce_data_separate(reduce_data, label)


def show_reduce_data_overlay(reduce_data: ReduceData, label) -> None:
    node_size = 10
    data = reduce_data.data2d
    match label:
        case "class":
            class_name = reduce_data.classes
        case "subclass":
            class_name = reduce_data.classes
            sub_class_name = reduce_data.subclasses
            # class_name-sub_class_name
            class_name = np.array(
                [f"{class_name[i]}-{sub_class_name[i]}" for i in range(len(class_name))]
            )

    num = np.unique(class_name)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=400)
    # fig.patch.set_facecolor("black")
    # ax.set_facecolor("black")

    squares = []
    for i in range(len(num)):
        index = np.where(class_name == num[i])
        colors = sns.color_palette("Spectral", as_cmap=True)((i) / len(num))
        ax.scatter(
            data[index, 0], data[index, 1], s=node_size, label=num[i], color=colors
        )

    ax.legend(handles=squares, labels=num)
    sns.despine(left=True, right=True, top=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")

    plt.savefig(f"overlay.png", bbox_inches="tight", dpi=400)
    # plt.show()


def show_reduce_data_separate(reduce_data: ReduceData, label) -> None:
    node = 10
    data2d = reduce_data.data2d
    match label:
        case "class":
            class_name = reduce_data.classes
        case "subclass":
            class_name = reduce_data.classes
            sub_class_name = reduce_data.subclasses
            # class_name-sub_class_name
            class_name = np.array(
                [f"{class_name[i]}-{sub_class_name[i]}" for i in range(len(class_name))]
            )

    class_name_set = np.unique(class_name)

    default_row = 4
    length = len(class_name_set)

    if default_row > length:
        default_row = length

    col = length // default_row + (length % default_row > 0)

    fig, axes = plt.subplots(default_row, col, figsize=(15, 10), dpi=400)
    fig.patch.set_facecolor("white")

    for i, class_name_item in enumerate(class_name_set):
        row, j = divmod(i, col)

        all_data = pd.DataFrame(data2d, columns=["x", "y"])
        spec_data = pd.DataFrame(
            data2d[class_name == class_name_item], columns=["x", "y"]
        )
        # check if axes is a 1D array
        if col == 1 and default_row == 1:
            ax = axes
        elif col == 1:
            ax = axes[row]
        else:
            ax = axes[row, j]

        sns.scatterplot(
            x="x",
            y="y",
            data=all_data,
            s=node,
            color="grey",
            ax=ax,
            legend=False,
        )
        sns.scatterplot(
            x="x",
            y="y",
            data=spec_data,
            s=node,
            ax=ax,
            c="cyan",
            legend=False,
        )

        sns.despine(left=True, right=True, top=True, bottom=True, ax=ax)
        ax.set_title(class_name_item, color="black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide empty subplots
    for i in range(length, default_row * col):
        row, j = divmod(i, col)
        # axes[row, j].axis("off")
        ax.axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.tight_layout()
    plt.savefig(f"separate.png", bbox_inches="tight", dpi=400)
    # plt.show()


def show_cluster_data(data: ClusterData) -> None:
    raise NotImplementedError("Show_ClusterData method not implemented")
