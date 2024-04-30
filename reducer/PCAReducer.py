from sklearn.decomposition import PCA

import numpy as np
from numpy.typing import NDArray

from dataprocess import DataProcess as dp 
from .Reducer import Reducer


class PCAReducer(Reducer):

    def __init__(self, dimension:int) -> None:
        super().__init__()
        self.dimension = dimension
        self.reducer = PCA
        
        
    def reduce(self, data:NDArray) -> str:
        """
        实现PCA降维，将降维结果保存在result_dir中
        """
        reduce_data = self.reducer(n_components=self.dimension).fit_transform(data)
        
        return reduce_data
