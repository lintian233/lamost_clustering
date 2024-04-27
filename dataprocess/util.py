#这里是存放DataProcess的工具函数的地方
from .SpectralData import SpectralData, SpectralDataType
from .Dataset import Dataset
from typing import List, Any
import re
import glob

import numpy as np
from numpy.typing import NDArray

def generate_dataset_name(dataset: Dataset) -> str:
    """
    生成数据集的名称
    参数：
    dataset: Dataset, 数据集
    返回：
    str, 数据集的名称
    """
    dataset_name = dataset.__class__.__name__
    dataset_index = generate_new_index(dataset.__dir_base_path)
    dataset_name_base = generate_dataset_name_base(dataset)
    return f"{dataset_name}-{dataset_index}-{dataset_name_base}"

def parser_fits_path(dirpath: str) -> List[str]:
    """
    解析fits文件的路径
    参数：
    dirpath: str, 文件路径
    返回：
    List[str], 文件路径列表
    """
    if dirpath[-1] != '/':
        dirpath += '/'

    all_fits_files_path = glob.glob(dirpath + '*.fits')
    
    if len(all_fits_files_path) == 0:
        raise FileNotFoundError(f"No fits files found in {dirpath}")
    
    return all_fits_files_path
    

def generate_dataset_name_base(dataset:Dataset) -> str:
    """
    获取数据集的名称
    参数：
    dataset: List[SpectralData], 数据集
    返回：
    str, 数据集的名称
    """
    if len(dataset) == 0 : 
        raise ValueError("Dataset is empty")

    dataset = dataset.to_numpy()

    total_num = len(dataset)
    star_num = dataset['class'][dataset['class'] == 'STAR'].shape[0]
    yso_num = dataset['class'][dataset['class'] == 'QSO'].shape[0]
    galaxy_num = dataset['class'][dataset['class'] == 'GALAXY'].shape[0]

    return f"SN{total_num}-STAR{star_num}-QSO{yso_num}-GALAXY{galaxy_num}"


def generate_new_index(dataset_dir:str) -> str:
    """
    生成新的索引
    参数：
    dataset_dir: str, 数据集的路径
    返回：
    str, 新的索引
    """
    index_set_list = []

    dataset_files = glob.glob(dataset_dir + '/*.npy')
    
    patten_index = r'(\d+)-SN'
    for dataset_file in dataset_files:
        match = re.search(patten_index, dataset_file)
        if match:
            index_set_list.append(match.group(1))

    if len(index_set_list) == 0:
        return '000'
        
    return f'{int(max(index_set_list)) + 1:03d}'



def save_dataset(dataset: Dataset) -> None:
    """
    保存数据集到文件
    参数：
    dataset_path: str, 数据集的路径
    dataset: List[SpectralData], 数据集
    返回：
    无
    """
    if dataset.__name is None:
        raise ValueError("Dataset name is None")
    
    
    
    dataset_name = dataset.__name
    dataset_path = dataset.__dir_base_path + dataset_name + '.npy'
    np.save(dataset_path, dataset.to_numpy())
    
