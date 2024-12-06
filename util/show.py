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
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, rand_score
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

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

def calculate_rand_index(y_true, y_pred):
    """
    通过真实标签和聚类标签自动统计分布并计算 Rand Index。
    :param y_true: 真实标签的列表
    :param y_pred: 聚类标签的列表
    :return: Rand Index
    """
    # 确保真实标签和聚类标签长度相同
    if len(y_true) != len(y_pred):
        raise ValueError("真实标签和聚类标签的长度必须一致")

    # 统计每个真实标签的聚类标签分布
    cluster_distribution = defaultdict(list)
    for true_label, cluster_label in zip(y_true, y_pred):
        cluster_distribution[true_label].append(cluster_label)

    # 构造扩展的真实标签和聚类标签
    y_pred_expanded = []
    y_true_expanded = []
    for label, clusters in cluster_distribution.items():
        y_pred_expanded.extend(clusters)
        y_true_expanded.extend([label] * len(clusters))

    # 计算 Rand Index
    return rand_score(y_true_expanded, y_pred_expanded)


def print_weighted_purity_by_class(cluster_details):
    """
    计算并打印每个主要类别的加权纯度。

    参数:
    - cluster_details: dict, 每个聚类簇的详细信息。
    """
    # 计算每个主要类别的加权纯度
    class_purity_weights = {}

    for cluster, details in cluster_details.items():
        major_class = details["Major Class"]
        purity = details["Purity"]
        weight = details["Weight"]
        
        # 加入或更新主要类别的加权纯度累积
        if major_class not in class_purity_weights:
            class_purity_weights[major_class] = {
                "Weighted Sum": 0,
                "Total Weight": 0
            }
        class_purity_weights[major_class]["Weighted Sum"] += purity * weight
        class_purity_weights[major_class]["Total Weight"] += weight

    # 计算每个类别的加权纯度
    class_weighted_purity = {
        cls: round(data["Weighted Sum"] / data["Total Weight"], 4)
        for cls, data in class_purity_weights.items()
    }
    
    # 打印加权纯度结果
    print("\nWeighted Purity by Major Class:")
    for major_class, weighted_purity in class_weighted_purity.items():
        print(f"{major_class}: {weighted_purity:.4f}")

def print_purity_details(total_purity, cluster_details):
    """
    简洁打印总体纯度和每个簇的主要信息，包括样本数和总样本数。

    参数:
    - total_purity: float, 总体加权平均纯度。
    - cluster_details: dict, 每个聚类簇的详细信息。
    """
    print(f"Overall Purity: {total_purity:.4f}")
    print("Cluster Details:")
    for cluster, details in cluster_details.items():
        major_class = details["Major Class"]
        purity = details["Purity"]
        weight = details["Weight"]
        major_class_count = details["Major Class Count"]
        cluster_total = details["Cluster Total"]
        print(f"{cluster}: {{Major Class: {major_class}, "
              f"Purity: {purity:.4f}, Weight: {weight:.4f}, "
              f"Samples: {major_class_count}/{cluster_total}}}")

def purity_score_with_details(y_true, y_pred):
    # 将字符串标签编码为整数
    le_true = LabelEncoder()
    le_pred = LabelEncoder()
    
    y_true_encoded = le_true.fit_transform(y_true)
    y_pred_encoded = le_pred.fit_transform(y_pred)
    
    # 计算混淆矩阵
    contingency_matrix = confusion_matrix(y_true_encoded, y_pred_encoded)
    total_samples = np.sum(contingency_matrix)
    
    # 初始化变量存储每个簇的主要类别和对应的样本数量
    cluster_purity_details = {}
    weighted_purity_sum = 0  # 存储加权纯度的累积
    cluster_weights = {}  # 存储每个簇的权重
    
    # 遍历每个聚类簇
    for cluster_idx in range(contingency_matrix.shape[1]):  # 按列（预测簇）遍历
        # 找到该聚类簇中主要类别的索引和数量
        major_class_idx = np.argmax(contingency_matrix[:, cluster_idx])
        major_class_count = contingency_matrix[major_class_idx, cluster_idx]
        cluster_total = np.sum(contingency_matrix[:, cluster_idx])
        
        # 计算该簇的纯度
        cluster_purity = float(major_class_count / cluster_total)
        
        # 权重为该簇的样本数量
        cluster_weight = cluster_total / total_samples
        cluster_weights[f"Cluster {cluster_idx}"] = cluster_weight
        
        # 加权纯度
        weighted_purity_sum += cluster_purity * cluster_weight
        
        # 记录主要类别及其占比
        cluster_purity_details[f"Cluster {cluster_idx}"] = {
            "Major Class": str(le_true.inverse_transform([major_class_idx])[0]),  # 解码主要类别
            "Major Class Count": int(major_class_count),
            "Cluster Total": int(cluster_total),
            "Purity": cluster_purity,
            "Weight": cluster_weight,
        }
    
    total_purity = weighted_purity_sum

    return total_purity, cluster_purity_details

