import os

import numpy as np
import umap
from numpy.typing import NDArray

import dataprocess.DataProcess as dp
from dataprocess.SpectralData import SpectralData
from config.config import REDUCEDATAPATH
from reducer.ReduceData import ReduceData
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from tqdm import tqdm


def get_data_from_dataset_index(dataset_index: str) -> tuple:
    """
    根据数据集索引获取数据集参数

    参数：
    dataset_index: str, 数据集索引

    返回：
    tuple: (
        data: 流量数据
        classes: 类别
        subclasses: 子类别
        obsid: 观测ID
        )
    """
    dataset = dp.load_dataset(dataset_index)
    data = np.zeros((len(dataset), 3000))
    classes = np.full(len(dataset), "0", dtype="U15")
    subclasses = np.full(len(dataset), "0", dtype="U15")
    obsid = np.full(len(dataset), "0", dtype="U15")

    for i, spectral_data in enumerate(tqdm(dataset)):
        wave = spectral_data.WAVELENGTH[:3000]
        data[i] = spectral_data.FLUX[:3000]
        classes[i] = spectral_data.CLASS
        subclasses[i] = spectral_data.SUBCLASS
        obsid[i] = spectral_data.OBSID

    return data, classes, subclasses, obsid


def if_reduced(dataset_index: str):
    """
    判断该数据集是否曾经被降维过
    是返回True，否则返回False

    参数：
    dataset_index: str, 数据集索引

    返回：
    bool: 是否被降维过
    """

    if os.path.exists(REDUCEDATAPATH + dataset_index):
        return True
    else:
        return False


def get_data2d(dataset_index: str):
    """
    根据数据集索引获取数据集的二维数据

    参数：
    dataset_index: str, 数据集索引

    返回：
    np.ndarray: 二维数据
    """

    if not if_reduced(dataset_index):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            metric="euclidean",
            learning_rate=1,
            min_dist=0.1,
        )
        data2d = reducer.fit_transform(get_data_from_dataset_index(dataset_index)[0])

    elif len(os.listdir(REDUCEDATAPATH + dataset_index)) == 0:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            metric="euclidean",
            learning_rate=1,
            min_dist=0.1,
        )
        data2d = reducer.fit_transform(get_data_from_dataset_index(dataset_index)[0])
    else:
        filename = os.listdir(REDUCEDATAPATH + dataset_index)[0]
        data = get_reduce_data(REDUCEDATAPATH + dataset_index + "/" + filename)
        data2d = data.data2d

    return data2d


def get_reduce_data(path: str) -> ReduceData:
    """
    读取降维数据
    返回ReduceData对象

    参数：
    path: str, 降维数据的地址

    返回：
    ReduceData, 降维数据类
    """

    data = np.load(path, allow_pickle=True)
    data2d = data[0]
    datand = data[1]
    classes = data[2]
    subclasses = data[3]
    obsid = data[4]
    return ReduceData(data2d, datand, classes, subclasses, obsid)


def numpy_from_reduce_data(data: ReduceData) -> np.ndarray:
    """
    将ReduceData对象转换为numpy数组

    参数：
    data: ReduceData, 降维数据类

    返回：
    np.ndarray, 降维数据的numpy数组
    """
    return np.array(
        [data.data2d, data.datand, data.classes, data.subclasses, data.obsid]
    )


def get_save_name(method, hyperparameters: dict) -> str:
    """
    根据降维方法和超参数生成保存降维数据的文件名

    参数：
    method: str, 降维方法
    hyperparameters: dict, 超参数字典

    返回：
    str, 保存文件名

    示例：
    >>> get_save_name("UMAP", {"n_neighbors": 5, "metric": "euclidean"})
    "UMAP-n_neighbors-5-metric-euclidean"
    """
    save_name = method + "-"
    for key in hyperparameters:
        save_name += key + "-" + str(hyperparameters[key]) + "-"
    return save_name[:-1]
