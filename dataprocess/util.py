# 这里是存放DataProcess的工具函数的地方
import re
import glob
from typing import List, Any

from numpy.typing import NDArray


def generate_dataset_name(class_name: str, base_dir: str, data_numpy: NDArray) -> str:
    """
    生成数据集的名称
    参数：
    dataset: Dataset, 数据集
    返回：
    str, 数据集的名称
    """
    dataset_name = class_name
    dataset_index = generate_new_index(base_dir)
    dataset_name_base = generate_dataset_name_base(data_numpy)
    return f"{dataset_name}-{dataset_index}-{dataset_name_base}"


def parser_fits_path(dirpath: str) -> List[str]:
    """
    解析fits文件的路径
    参数：
    dirpath: str, 文件路径
    返回：
    List[str], 文件路径列表
    """
    if dirpath[-1] != "/":
        dirpath += "/"

    all_fits_files_path = glob.glob(dirpath + "*.fits")

    if len(all_fits_files_path) == 0:
        raise FileNotFoundError(f"No fits files found in {dirpath}")

    return all_fits_files_path


def generate_dataset_name_base(dataset: NDArray[Any]) -> str:
    """
    获取数据集的名称
    参数：
    dataset: List[SpectralData], 数据集
    返回：
    str, 数据集的名称
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    total_num = len(dataset)
    star_num = dataset["class"][dataset["class"] == "STAR"].shape[0]
    yso_num = dataset["class"][dataset["class"] == "QSO"].shape[0]
    galaxy_num = dataset["class"][dataset["class"] == "GALAXY"].shape[0]

    return f"SN{total_num}-STAR{star_num}-QSO{yso_num}-GALAXY{galaxy_num}"


def generate_new_index(dataset_dir: str) -> str:
    """
    生成新的索引
    参数：
    dataset_dir: str, 数据集的路径
    返回：
    str, 新的索引
    """
    index_set_list = []

    dataset_files = glob.glob(dataset_dir + "/*.npy")

    patten_index = r"(\d+)-SN"
    for dataset_file in dataset_files:
        match = re.search(patten_index, dataset_file)
        if match:
            index_set_list.append(match.group(1))

    if len(index_set_list) == 0:
        return "000"

    return f"{int(max(index_set_list)) + 1:03d}"
