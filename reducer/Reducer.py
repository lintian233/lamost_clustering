from abc import ABC, abstractmethod
from typing import Any
import os

import pandas as pd
import numpy as np

from config.config import REDUCEDATAPATH
from reducer.ReduceData import ReduceData
from .util import get_reduce_data
from dataprocess.Dataset import Dataset


"""

"""


class Reducer(ABC):
    index: str
    reducer: Any
    result_dir: str
    hyperparameters: dict[str, Any]
    dimension: int

    def __init__(self) -> None:
        self.result_dir = REDUCEDATAPATH
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.all_result = ""

    @abstractmethod
    def reduce(self, dataset: Dataset) -> ReduceData:
        """
        根据数据集的索引进行降维
        将降维结果保存在result_dir中
        返回ReduceData对象
        此方法需要在子类中实现
        """
        pass
