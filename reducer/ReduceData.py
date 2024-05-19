import numpy as np
from dataclasses import dataclass

from numpy.typing import NDArray
from typing import Any, List


"""
每一次降维的结果都会保存在result_dir/data_index/下
文件名为method-超参数1-值1-超参数2-值2.npy

dir/
    LAMOSTDataset-001/
        -PCA-hyperparameters
            -data2d
            -datand
            -class
            -subclass
            -obsid
"""


@dataclass
class ReduceData:
    data2d: NDArray[np.float64]
    datand: NDArray[np.float64]
    classes: NDArray[np.str_]
    subclasses: NDArray[np.str_]
    obsid: NDArray[np.str_]
    info: List[str] = None

    def __init__(
        self,
        data2d: NDArray[np.float64],
        datand: NDArray[np.float64],
        classes: NDArray[np.str_],
        subclasses: NDArray[np.str_],
        obsid: NDArray[np.str_],
    ):
        self.data2d = data2d
        self.datand = datand
        self.classes = classes
        self.subclasses = subclasses
        self.obsid = obsid

    @classmethod
    def from_numpy(cls, data2d, datand, classes, subclasses, obsid) -> "ReduceData":
        """
        从numpy数组中创建ReduceData对象
        """

        return cls(data2d, datand, classes, subclasses, obsid)
