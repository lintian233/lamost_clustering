import os

import numpy as np
import umap

import dataprocess.DataProcess as dp
from config.config import REDUCEDATAPATH
from reducer.ReduceData import ReduceData


def get_data_from_dataset_index(dataset_index: str) -> np.ndarray:
    dataset = dp.load_dataset(dataset_index)
    data = np.zeros((len(dataset), 3000))
    classes = np.full(len(dataset), "0", dtype="U15")
    subclasses = np.full(len(dataset), "0", dtype="U15")
    obsid = np.full(len(dataset), "0", dtype="U15")

    for i in range(len(dataset)):
        data[i] = dataset[i].data[0][0][:3000]

    for i in range(len(dataset)):
        classes[i] = dataset[i].header["CLASS"]

    for i in range(len(dataset)):
        subclasses[i] = dataset[i].header["SUBCLASS"]

    for i in range(len(dataset)):
        obsid[i] = dataset[i].header["OBSID"]

    return data, classes, subclasses, obsid


def if_reduced(dataset_index: str):
    if os.path.exists(REDUCEDATAPATH + dataset_index):
        return True
    else:
        return False


def get_data2d(dataset_index: str):
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
    data = np.load(path, allow_pickle=True)
    data2d = data[0]
    datand = data[1]
    classes = data[2]
    subclasses = data[3]
    obsid = data[4]
    return ReduceData(data2d, datand, classes, subclasses, obsid)


def numpy_from_reduce_data(data: ReduceData) -> np.ndarray:
    return np.array(
        [data.data2d, data.datand, data.classes, data.subclasses, data.obsid]
    )


def get_save_name(method, hyperparameters: dict) -> str:
    save_name = method + "-"
    for key in hyperparameters:
        save_name += key + "-" + str(hyperparameters[key]) + "-"
    return save_name[:-1]
