# 这里是存放DataProcess的工具函数的地方
import re
import glob
from typing import List, Any
import os

import numpy as np
from numpy.typing import NDArray
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.table import Table
from astropy.io import fits
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from tqdm import tqdm
from numba import jit

from config.config import DATASETBASEPATH
from .SpectralData import (
    SpectralData,
    LamostSpectraData,
    SDSSSpectraData,
    StdSpectraData,
)


def check_dataset_index(dataset_index: str) -> bool:
    """
    检查数据集的索引是否合法，
    所有合法的数据集索引都应该以数字结尾，并且在dataprocess文件夹下有对应的类实现。

    参数：
    dataset_index: str, 数据集的索引

    返回：
    bool, 是否合法

    示例：
    存放类实现的文件夹：\n
    dataprocess/ \n
        -LamostDataset.py \n
        -SdssDataset.py \n


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
    找到数据集的路径
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
    DATASETBASEPATH/LamostDataset/LamostDataset-000-SN0-STAR0-QSO0-GALAXY0.fits \n
    DATASETBASEPATH/LamostDataset/LamostDataset-001-SN0-STAR0-QSO0-GALAXY0.fits \n

    >>> find_dataset_path("NonsenDataset-000")
    ValueError: 'Invalid dataset index format'
    >>> find_dataset_path("LamostDataset-000")
    'Data/LamostDataset/LamostDataset-000-SN0-STAR0-QSO0-GALAXY0.fits'
    >>> find_dataset_path("LamostDataset-001")
    'Data/LamostDataset/LamostDataset-001-SN0-STAR0-QSO0-GALAXY0.fits'
    >>> find_dataset_path("LamostDataset-002")
    FileNotFoundError: 'Dataset LamostDataset-002 not found'
    """

    if not check_dataset_index(dataset_index):
        raise ValueError(f"Invalid dataset index format:{dataset_index}")

    base_path = DATASETBASEPATH
    if base_path[-1] != "/":
        base_path += "/"

    dataset_dirs = glob.glob(base_path + "*Dataset/")
    for item in dataset_dirs:
        current = glob.glob(item + "*.fits")
        for i in current:
            if dataset_index in i:
                return i

    raise FileNotFoundError(f"Dataset {dataset_index} not found")


def generate_dataset_name(
    class_name: str, base_dir: str, labels_list: NDArray[Any]
) -> str:
    """
    生成数据集的名称

    参数：
    dataset_name: str, 数据集的类名称
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
    dataset_name_base = generate_dataset_name_base(labels_list)
    return f"{dataset_name}-{dataset_index}-{dataset_name_base}"


def parser_fits_path(dirpath: str) -> List[str]:
    """
    解析给定目录中的FITS文件的路径。

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
    all_dir = glob.glob(dirpath + "*")
    all_fits_files_path = []
    for d in all_dir:
        if os.path.isdir(d):
            all_fits_files_path.extend(parser_fits_path(d))

    all_fits_files_path.extend(glob.glob(dirpath + "*.fits"))

    if len(all_fits_files_path) == 0:
        raise FileNotFoundError(f"No fits files found in {dirpath}")

    return all_fits_files_path


def generate_dataset_name_base(dataset: NDArray[Any]) -> str:
    """
    用于获取数据集的名称。

    注意:
    数据集名称的生成规则是根据数据集中各类别样本的数量，
    按照'SN{数量}-STAR{数量}-QSO{数量}-GALAXY{数量}'的格式生成。

    参数:
    dataset (NDArra[Any]): 标签列表。

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
    star_num = 0
    qso_num = 0
    galaxy_num = 0

    star_num = len(dataset[dataset == "STAR"])
    qso_num = len(dataset[dataset == "QSO"])
    galaxy_num = len(dataset[dataset == "GALAXY"])
    return f"SN{total_num}-STAR{star_num}-QSO{qso_num}-GALAXY{galaxy_num}"


def generate_new_index(dataset_dir: str) -> str:
    """
    用于生成新的数据集索引。

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
    "Data/Dataset/Lamost/LamostDataset-001-SN0-STAR0-QSO0-GALAXY0.fits"
    "Data/Dataset/Lamost/LamostDataset-002-SN0-STAR0-QSO0-GALAXY0.fits"
    >>> generate_new_index(dirpath)
    '003'
    """
    index_set_list = []

    if dataset_dir[-1] != "/":
        dataset_dir += "/"

    dataset_files = glob.glob(dataset_dir + "*.fits")

    patten_index = r"(\d+)-SN"
    for dataset_file in dataset_files:
        match = re.search(patten_index, dataset_file)
        if match:
            index_set_list.append(match.group(1))

    if len(index_set_list) == 0:
        return "000"

    return f"{int(max(index_set_list)) + 1:03d}"


def init_lamost_dataset(
    hdulist: HDUList, length, n_jobs: int = 1
) -> List[SpectralData]:
    # [0,1],[2,3],[4,5],[6,7]
    ilist = np.arange(0, length * 2, 2)
    set_loky_pickler("dill")
    spectrum_data = Parallel(n_jobs=n_jobs)(
        delayed(LamostSpectraData)(HDUList([hdulist[i], hdulist[i + 1]]))
        for i in tqdm(ilist)
    )
    return spectrum_data


def init_sdss_dataset(hdulist: HDUList, length, n_jobs: int = 1) -> List[SpectralData]:
    # [0,1,2,3]
    ilist = np.arange(0, length * 4, 4)
    set_loky_pickler("dill")
    spectrum_data = Parallel(n_jobs=n_jobs)(
        delayed(SDSSSpectraData)(
            HDUList([hdulist[i], hdulist[i + 1], hdulist[i + 2], hdulist[i + 3]])
        )
        for i in tqdm(ilist)
    )
    return spectrum_data


def init_std_dataset(hdulist: HDUList, length, n_jobs: int = 1) -> List[SpectralData]:
    ilist = np.arange(0, length * 2, 2)
    set_loky_pickler("dill")
    spectrum_data = Parallel(n_jobs=n_jobs)(
        delayed(StdSpectraData)(HDUList([hdulist[i], hdulist[i + 1]]))
        for i in tqdm(ilist)
    )
    return spectrum_data


def resample(
    raw_wavelength: NDArray, raw_flux: NDArray, origin: str, ormask: NDArray
) -> NDArray:
    step = (8900 - 3850) / 3700
    if origin == "LAMOST":
        if sum(ormask) != 0:
            useful = False
            flux = np.zeros(3700)
            wavelength = np.zeros(3700)
            return wavelength, flux, useful

        flux = np.zeros(3700)
        for i in range(3700):
            x = 3850 + i * step

            left = np.where(raw_wavelength < x)[0][-1]
            right = np.where(raw_wavelength > x)[0][0]

            l_weight = (raw_wavelength[right] - x) / (
                raw_wavelength[right] - raw_wavelength[left]
            )
            r_weight = (x - raw_wavelength[left]) / (
                raw_wavelength[right] - raw_wavelength[left]
            )

            flux[i] = raw_flux[left] * l_weight + raw_flux[right] * r_weight

        length = np.sqrt(sum(flux**2))
        flux = flux / length
        wavelength = np.linspace(3850, 8900, 3700)

        return wavelength, flux, True

    elif origin == "SDSS":
        if not get_useful(ormask):
            useful = False
            flux = np.zeros(3700)
            wavelength = np.zeros(3700)
            return wavelength, flux, useful
        flux = np.zeros(3700)

        for i in range(3700):
            x = 3850 + i * step

            left = np.where(raw_wavelength < x)[0][-1]
            right = np.where(raw_wavelength > x)[0][0]

            l_weight = (raw_wavelength[right] - x) / (
                raw_wavelength[right] - raw_wavelength[left]
            )
            r_weight = (x - raw_wavelength[left]) / (
                raw_wavelength[right] - raw_wavelength[left]
            )

            flux[i] = raw_flux[left] * l_weight + raw_flux[right] * r_weight

        length = np.sqrt(sum(flux**2))
        flux = flux / length
        wavelength = np.linspace(3850, 8900, 3700)

        return wavelength, flux, True


def to_std_spectral_data(spec_data: SpectralData) -> StdSpectraData:

    class_name = spec_data.__class__.__name__
    if class_name == "LamostSpectraData":
        origin = "LAMOST"
    elif class_name == "SDSSSpectraData":
        origin = "SDSS"
    else:
        raise ValueError(f"Unsupported SpectralData class: {class_name}")

    obsid = spec_data.OBSID
    clas = spec_data.CLASS
    subclas = spec_data.SUBCLASS
    ormask = spec_data.ORMASK
    wavelength, flux, useful = resample(
        spec_data.WAVELENGTH, spec_data.FLUX, origin, ormask
    )

    header = fits.Header()
    header["OBSID"] = obsid
    header["CLASS"] = clas
    header["SUBCLASS"] = subclas
    header["ORIGIN"] = origin
    header["USEFUL"] = useful

    primary_hdu = fits.PrimaryHDU(header=header)

    col1 = fits.Column(name="FLUX", format="E", array=flux)
    col2 = fits.Column(name="WAVELENGTH", format="E", array=wavelength)

    cols = fits.ColDefs([col1, col2])
    bintable_hdu = fits.BinTableHDU.from_columns(cols)

    hdulist = fits.HDUList([primary_hdu, bintable_hdu])

    return StdSpectraData(hdulist)


def get_useful(ormask: NDArray) -> bool:
    num_spectra = len(ormask)
    num_bits = 29

    mask_table = np.zeros((num_spectra, num_bits), dtype=int)

    for i, ormask in enumerate(ormask):
        for bit in range(num_bits):
            if (ormask & (1 << bit)) != 0:
                mask_table[i, bit] = 1

    # important_bits = [1, 2, 3, 4, 6, 7, 16, 20, 21, 23]
    # important_bits = [2, 3, 4, 5, 18, 23, 27, 28]
    important_bits = [2, 4, 5]

    for bit in important_bits:
        if sum(mask_table[:, bit]) > 0:
            return False

    return True
