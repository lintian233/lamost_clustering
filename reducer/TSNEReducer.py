from .Reducer import Reducer
from sklearn.manifold import TSNE


class TSNEReducer(Reducer):

    def __init__(self, 
                 dimension:int, 
                 perplexity:int, 
                 learning_rate:int, 
                 n_iter:int) -> None:
        super().__init__()
        self.dimension = dimension
        self.reducer = TSNE
        self.hyperparameters = {
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "n_iter": n_iter
        }
        

    def reduce(self, data) -> str:
        """
        实现TSNE降维，将降维结果保存在result_dir中
        """
        reduce_data = self.reducer(n_components=self.dimension, 
                                   perplexity=self.hyperparameters["perplexity"], 
                                   learning_rate=self.hyperparameters["learning_rate"],
                                   n_iter=self.hyperparameters["n_iter"]
                                  ).fit_transform(data)
        return reduce_data
