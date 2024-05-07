import numpy as np
from dataclasses import dataclass

from numpy.typing import NDArray
from typing import Any


"""
因为ReduceData的二维和n维数组非常的大，对于data2d,datand,labels,用npy储存,超参数列表用json储存
存储在一个文件夹内，文件夹名字为目录结构如下。
Data/
    -ReduceData/
        -PCA-001-LAMOSTDataset-001-PARAMS-1-2-3-4-5-6-7-8-9-10/
            -data2d
            -datand
            -labels
            -hyperparameters
"""


@dataclass
class ReduceData:
    data2d: NDArray[np.float64]
    datand: NDArray[np.float64]
    classes: NDArray[np.str_]
    subclasses: NDArray[np.str_]
    obsid: NDArray[np.str_]

    def __init__(
        self,
        data2d: NDArray[np.float64],
        datand: NDArray[np.float64],
        classes: NDArray[np.str_],
        subclasses: NDArray[np.str_],
        obsid: NDArray[np.str_]
    ):
        self.data2d = data2d
        self.datand = datand
        self.classes = classes
        self.subclasses = subclasses
        self.obsid = obsid

    @classmethod
    def from_numpy(cls, data2d, datand, classes, subclasses, obsid) -> "ReduceData":
        return cls(data2d, datand, classes, subclasses, obsid)
