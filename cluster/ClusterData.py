import numpy as np
from dataclasses import dataclass

from numpy.typing import NDArray
from typing import Any

"""
因为ReduceData的二维和n维数组非常的大，对于data2d,datand,labels,用npy储存,超参数列表用json储存
存储在一个文件夹内，文件夹名字为目录结构如下。
Data/
    -ClusterData/
                -PCA-001-Lamostdataset-000-PARM-12-2-3-4-5-90-1-4-5/
                                                                     -data2d.npy
                                                                     -labels.npy
"""


@dataclass
class ClusterData:
    data2d: NDArray[np.float64]
    datand: NDArray[np.float64]
    classes: NDArray[np.str_]
    subclasses: NDArray[np.str_]
    labels: NDArray[np.str_]
    obsid: NDArray[np.str_]
    info: NDArray[np.str_] = None

    def __init__(
        self,
        data2d: NDArray[np.float64],
        datand: NDArray[np.float64],
        classes: NDArray[np.str_],
        subclasses: NDArray[np.str_],
        labels: NDArray[np.str_],
        obsid: NDArray[np.str_],
        info: NDArray[np.str_] = None,
    ):
        self.data2d = data2d
        self.datand = datand
        self.classes = classes
        self.subclasses = subclasses
        self.labels = labels
        self.obsid = obsid
        self.info = info

    @classmethod
    def from_numpy(
        cls,
        data2d,
        datand,
        classes,
        subclasses,
        labels,
        obsid,
        info=None,
    ) -> "ClusterData":
        """
        从numpy数组中创建ReduceData对象
        """

        return cls(data2d, datand, classes, subclasses, labels, obsid, info)
