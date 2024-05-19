import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


from astropy.io.fits.header import Header
from astropy.io.fits.fitsrec import FITS_rec
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from matplotlib.colors import ListedColormap, BoundaryNorm

from dataprocess import SpectralData
from dataprocess.SpectralData import LamostSpectraData, SDSSSpectraData, StdSpectraData
from reducer.ReduceData import ReduceData
from cluster.ClusterData import ClusterData
from config.config import VISUALIZATION_REDUCE_PATH, VISUALIZATION_CLUSTER_PATH

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#393b79",
    "#5254a3",
    "#6b6ecf",
    "#9c9ede",
    "#637939",
    "#8ca252",
    "#b5cf6b",
    "#cedb9c",
    "#8c6d31",
    "#bd9e39",
    "#e7ba52",
    "#e7cb94",
    "#843c39",
    "#ad494a",
    "#d6616b",
    "#e7969c",
    "#7b4173",
    "#a55194",
    "#ce6dbd",
    "#de9ed6",
    "#3182bd",
    "#6baed6",
    "#9ecae1",
    "#c6dbef",
    "#e6550d",
    "#fd8d3c",
    "#fdae6b",
    "#fdd0a2",
    "#31a354",
    "#74c476",
    "#a1d99b",
    "#c7e9c0",
    "#756bb1",
    "#9e9ac8",
    "#bcbddc",
    "#dadaeb",
    "#636363",
    "#969696",
    "#bdbdbd",
    "#d9d9d9",
]


def show_spectra_data(data: SpectralData) -> None:
    telescope = ""
    if isinstance(data, LamostSpectraData):
        telescope = "LAMOST"
    elif isinstance(data, SDSSSpectraData):
        telescope = "SDSS"
    elif isinstance(data, StdSpectraData):
        telescope = "STD-" + data.ORIGIN
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
    # plt.show()


def show_data(data) -> None:
    if isinstance(data, ReduceData):
        info = data.info
        save_dir = f"{VISUALIZATION_REDUCE_PATH}/{info[0]}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = f"{save_dir}{info[1]}"

        set_loky_pickler("dill")
        mode = ["overlay", "separate", "separate", "overlay"]
        label = ["class", "class", "subclass", "subclass"]
        Parallel(n_jobs=4)(
            delayed(_show_reduce_data)(
                data, mode=mode[i], label=label[i], save_path=save_path
            )
            for i in range(4)
        )

        print(f"Save {info[0]}-{info[1]} to {save_dir}")

    elif isinstance(data, ClusterData):
        info = data.info
        save_dir = f"{VISUALIZATION_CLUSTER_PATH}/{info[0]}/{info[1]}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = f"{save_dir}{info[2]}"
        set_loky_pickler("dill")
        mode = ["overlay", "separate"]
        Parallel(n_jobs=2)(
            delayed(_show_cluster_data)(data, mode=mode[i], save_path=save_path)
            for i in range(2)
        )

        print(f"Save {info[0]}-{info[1]} to {save_dir}")


def _show_cluster_data(
    cluster_data,
    save_path: str,
    mode="overlay",
) -> None:
    data2d = cluster_data.data2d
    labels = cluster_data.labels
    num = np.unique(labels)
    save_path = f"{save_path}-cluster-{len(num)}"

    match mode:
        case "overlay":
            show_data_overlay(data2d, labels, save_path)
        case "separate":
            show_data_separate(data2d, labels, save_path)


def _show_reduce_data(
    reduce_data,
    save_path: str,
    mode="overlay",
    label="class",
) -> None:
    data2d = reduce_data.data2d
    match label:
        case "class":
            class_name = reduce_data.classes
            save_path = f"{save_path}-class"
        case "subclass":
            class_name = reduce_data.classes
            save_path = f"{save_path}-subclass"

            sub_class_name = reduce_data.subclasses
            class_name = np.array(
                [f"{class_name[i]}-{sub_class_name[i]}" for i in range(len(class_name))]
            )

    match mode:
        case "overlay":
            show_data_overlay(data2d, class_name, save_path)
        case "separate":
            show_data_separate(data2d, class_name, save_path)


def show_data_overlay(data2d, class_name, save_path) -> None:
    node_size = 10
    data = data2d
    num = np.unique(class_name)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=400)

    # 创建离散的颜色映射
    num_colors = len(num)
    cmap = ListedColormap(COLORS[:num_colors])
    norm = BoundaryNorm(range(num_colors + 1), cmap.N)

    for i, class_label in enumerate(num):
        index = np.where(class_name == class_label)
        colors = cmap(norm([i]))
        scatter = ax.scatter(
            data[index, 0], data[index, 1], s=node_size, label=class_label, color=colors
        )

    # 添加颜色条，并设置刻度
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        boundaries=np.arange(num_colors + 1),
    )
    cbar.set_ticks(np.arange(num_colors))
    cbar.set_ticklabels(num)

    sns.despine(left=True, right=True, top=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")

    # plt.savefig(f"{label}-overlay.png", bbox_inches="tight", dpi=400)
    plt.savefig(f"{save_path}-overlay.png", bbox_inches="tight", dpi=400)


def show_data_separate(data2d, class_name, save_path) -> None:
    node = 10

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

        # 隐藏空的图的坐标轴
        if i == length - 1:
            for k in range(i + 1, default_row * col):
                row, j = divmod(k, col)
                if col == 1 and default_row == 1:
                    ax = axes
                elif col == 1:
                    ax = axes[row]
                else:
                    ax = axes[row, j]
                ax.axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.tight_layout()
    plt.savefig(f"{save_path}-separate.png", bbox_inches="tight", dpi=400)
