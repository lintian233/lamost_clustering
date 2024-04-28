# 这里是存放DataProcess的工具函数的地方
import re
import glob
from typing import List, Any

from numpy.typing import NDArray

from config.config import DATASETBASEPATH

def check_dataset_index(dataset_index: str) -> bool:
    """
    检查数据集的名称是否合法, 所有合法的数据集名称都是以"Dataset"结尾的，且在dataprocess目录下有对应的类文件
    参数：
    dataset_name: str, 数据集的名称
    返回：
    bool, 是否合法
    >>> check_dataset_index("LamostDataset-001")
    True
    >>> check_dataset_index("SDSSDataset-002")
    True
    >>> check_dataset_index("NONSENDataset-003")
    False
    """
    class_python_files = glob.glob("dataprocess/*Dataset.py")
    patten = r"\\(\w+Dataset).py"
    
    class_names = []
    for class_python_file in class_python_files:
        match = re.search(patten, class_python_file)
        if match:
            class_names.append(match.group(1))
    

    telescope = dataset_index.split("-")[0]
    index = dataset_index.split("-")[1]
    if telescope in class_names and index.isdigit():
        return True
    
    return False


def find_dataset_path(dataset_index: str) -> str:
    """
    找到数据集的路径
    参数：
    dataset_index: str, 数据集的索引
    返回：
    str, 数据集的路径, 如果没有找到raise FileNotFoundError
    """
    if not check_dataset_index(dataset_index):
        raise ValueError("Invalid dataset index format")

    base_path = DATASETBASEPATH
    if base_path[-1] != "/":
        base_path += "/"
    
    dataset_dirs = glob.glob(base_path + "*Dataset/")
    for item in dataset_dirs:
        current = glob.glob(item + "*.npy")
        for i in current:
            if dataset_index in i:
                return i
    
    raise FileNotFoundError(f"Dataset {dataset_index} not found")


def generate_dataset_name(class_name: str, base_dir: str, data_numpy: NDArray) -> str:
    """
    生成数据集的名称
    参数：
    dataset: Dataset, 数据集
    返回：
    str, 数据集的名称
    >>> generate_dataset_name("LamostDataset", "/Data/Dataset/Lamost", np.array([]))
    'LamostDataset-000-SN0-STAR0-QSO0-GALAXY0'
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

    if dataset_dir[-1] != "/":
        dataset_dir += "/"

    dataset_files = glob.glob(dataset_dir + "*.npy")

    patten_index = r"(\d+)-SN"
    for dataset_file in dataset_files:
        match = re.search(patten_index, dataset_file)
        if match:
            index_set_list.append(match.group(1))

    if len(index_set_list) == 0:
        return "000"

    return f"{int(max(index_set_list)) + 1:03d}"
