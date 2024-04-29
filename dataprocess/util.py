# 这里是存放DataProcess的工具函数的地方
import re
import glob
from typing import List, Any


from numpy.typing import NDArray
from astropy.io.fits.header import Header


from config.config import DATASETBASEPATH


def check_dataset_index(dataset_index: str) -> bool:
    """
    检查数据集的索引是否合法，
    所有合法的数据集索引都应该以数字结尾，并且在数据集目录下有对应的数据文件。

    参数：
    dataset_index: str, 数据集的索引

    返回：
    bool, 是否合法

    示例：
    假设文件夹结构如下：

    DATASETBASEPATH = "Data/" \n
    DATASETBASEPATH/LamostDataset/LamostDataset-000-SN0-STAR0-QSO0-GALAXY0.npy \n
    DATASETBASEPATH/LamostDataset/LamostDataset-001-SN0-STAR0-QSO0-GALAXY0.npy \n

    >>> check_dataset_index("LamostDataset-000")
    True
    >>> check_dataset_index("LamostDataset-001")
    True
    >>> check_dataset_index("LamostDataset-002")
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
    找到数据集的路径, 如果没有找到则抛出 FileNotFoundError 异常。
    DTATASET_BASE_PATH 在config/config.py中定义。
    DATASET_BASE_PATH 下数据集的文件夹以"Dataset"结尾。
    数据集以"DATASET_NAME-INDEX-SN0-STAR0-QSO0-GALAXY0.npy"的格式命名。

    参数：
    dataset_index: str, 数据集的索引

    返回：
    str, 数据集的路径

    示例：
    文件夹结构如下：

    DATASETBASEPATH = "Data/" \n
    DATASETBASEPATH/LamostDataset/LamostDataset-000-SN0-STAR0-QSO0-GALAXY0.npy \n
    DATASETBASEPATH/LamostDataset/LamostDataset-001-SN0-STAR0-QSO0-GALAXY0.npy \n

    >>> find_dataset_path("NonsenDataset-000")
    ValueError: 'Invalid dataset index format'
    >>> find_dataset_path("LamostDataset-000")
    'Data/LamostDataset/LamostDataset-000-SN0-STAR0-QSO0-GALAXY0.npy'
    >>> find_dataset_path("LamostDataset-001")
    'Data/LamostDataset/LamostDataset-001-SN0-STAR0-QSO0-GALAXY0.npy'
    >>> find_dataset_path("LamostDataset-002")
    FileNotFoundError: 'Dataset LamostDataset-002 not found'
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
    dataset_name: str, 数据集的基础名称
    base_dir: str, 数据集的存储路径
    data_numpy: NDArray[Any], 数据集的numpy数组

    返回：
    str, 完整的数据集名称

    示例：
    class_name = "LamostDataset" \n
    base_dir = "/Data/Dataset/Lamost" \n
    data = np.array([]) \n
    >>> generate_dataset_name(class_name, base_dir, data) # NNN is a new index
    'LamostDataset-NNN-SN0-STAR0-QSO0-GALAXY0'
    """
    dataset_name = class_name
    dataset_index = generate_new_index(base_dir)
    dataset_name_base = generate_dataset_name_base(data_numpy)
    return f"{dataset_name}-{dataset_index}-{dataset_name_base}"


def parser_fits_path(dirpath: str) -> List[str]:
    """
    这个函数解析给定目录中的FITS文件的路径。

    参数:
    dirpath (str): 要搜索FITS文件的目录的路径。

    返回:
    List[str]: 在目录中找到的每个FITS文件的路径列表。

    示例:
    >>> parser_fits_path("Data/Fits/")
    ['Data/Fits/1.fits', 'Data/Fits/2.fits', 'Data/Fits/3.fits']

    >>> parser_fits_path("NonsenPath/")
    FileNotFoundError: 在NonsenPath/中没有找到fits文件
    """

    if dirpath[-1] != "/":
        dirpath += "/"

    all_fits_files_path = glob.glob(dirpath + "*.fits")

    if len(all_fits_files_path) == 0:
        raise FileNotFoundError(f"No fits files found in {dirpath}")

    return all_fits_files_path


def generate_dataset_name_base(dataset: NDArray[Any]) -> str:
    """
    这个函数用于获取数据集的名称。

    注意:
    数据集名称的生成规则是根据数据集中各类别样本的数量，
    按照'SN{数量}-STAR{数量}-QSO{数量}-GALAXY{数量}'的格式生成。
    如果某一类别的样本数量为0，则该类别的数量在名称中表示为0。

    参数:
    dataset (NDArra[Any]): 数据集。
    Any: SpectralDataType类型的数据集。

    返回:
    str: 数据集的名称。

    示例:
    假设数据集中包含10个STAR类别的样本，没有其他类别的样本。
    >>> generate_dataset_name_base(data)
    'SN10-STAR10-QSO0-GALAXY0'
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    total_num = len(dataset)
    header_set = dataset["header"]
    star_num = 0
    yso_num = 0
    galaxy_num = 0

    for header in header_set:
        obj_class = Header.fromstring(header)["CLASS"]
        if obj_class == "STAR":
            star_num += 1
        elif obj_class == "QSO":
            yso_num += 1
        elif obj_class == "GALAXY":
            galaxy_num += 1

    return f"SN{total_num}-STAR{star_num}-QSO{yso_num}-GALAXY{galaxy_num}"


def generate_new_index(dataset_dir: str) -> str:
    """
    这个函数用于生成新的数据集索引。

    注意：
    此函数用于生成新的数据集索引，索引是根据目录下已有的数据集文件的index加一得到的。
    如果目录下没有数据集文件，则新的索引为 '000'。

    参数：
    dataset_dir: str, 数据集的路径

    返回：
    str, 新的索引

    示例:

    假设目录路径为 "Data/Dataset/Lamost"

    在 "Data/Dataset/Lamost/" 目录下有如下文件：
    "Data/Dataset/Lamost/LamostDataset-001-SN0-STAR0-QSO0-GALAXY0.npy"
    "Data/Dataset/Lamost/LamostDataset-002-SN0-STAR0-QSO0-GALAXY0.npy"
    >>> generate_new_index(dirpath)
    '003'
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
