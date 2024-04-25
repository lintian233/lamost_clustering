
from dataprocess import SpectralData
from reducer import ReduceData
from cluster import ClusterData

class Show:
    def __init__(self):
        pass
    
    @staticmethod
    def show_spectraldata(data: SpectralData) -> None:
        raise NotImplementedError("Show_SpectralData method not implemented")
    
    @staticmethod
    def show_reduce_data(data: ReduceData) -> None:
        raise NotImplementedError("Show_ReduceData method not implemented")
    
    @staticmethod
    def show_cluster_data(data: ClusterData) -> None:
        raise NotImplementedError("Show_ClusterData method not implemented")
    
    