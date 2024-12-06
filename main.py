import numpy as np

from dataprocess import LamostDataset, SDSSDataset
from dataprocess import DataProcess
from cluster import Cluster, HDBSCANCluster, SpectralCluster
from reducer import ReduceManager
from reducer import UMAPReducer, PCAReducer, TSNEReducer
from util import show_data, show_spectra_data, rand_score, calculate_rand_index
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from util import purity_score_with_details, print_purity_details, print_weighted_purity_by_class

#LamostDataset().add_dataset_parallel("origin_data/Lamost/all-lamost/2k")
#dataset = DataProcess.load_dataset("LamostDataset-003")
#dataset = DataProcess.preprocessing(dataset)
dataset = DataProcess.load_dataset("StdDataset-003")

umapreducer = UMAPReducer(
            dimension=20,
            metric="chebyshev",
            min_dist=0.01,
            n_neighbors=35
        )

reducedata = umapreducer.reduce(dataset)

show_data(reducedata)

hdbscancluster = HDBSCANCluster(
min_cluster_size=40,min_samples=5,
cluster_selection_epsilon=0.005, cluster_selection_method = 'eom'
        )

clusterdata = hdbscancluster.fit(reducedata)

show_data(clusterdata)
clustered_labels = (clusterdata.labels >= 0)

select_labels = reducedata.subclasses[clustered_labels]
select_labels_pred =clusterdata.labels[clustered_labels]
print(select_labels)

rand_score_ = rand_score(select_labels, select_labels_pred)
print(f"Rand score: {rand_score_}")
print("Adjusted rand score: ", adjusted_rand_score(select_labels, select_labels_pred))
print("Adjusted mutual info score: ", adjusted_mutual_info_score(select_labels, select_labels_pred))

total_purity, cluster_details = purity_score_with_details(select_labels, select_labels_pred)
print_purity_details(total_purity, cluster_details)

spectralcluster = SpectralCluster(
            n_clusters=20,
            assign_labels="discretize",
            n_components=10,
            n_neighbors=20,
        )

clusterdata = spectralcluster.fit(reducedata)

show_data(clusterdata)

#clusterdata.subclasses = reducedata.subclasses

rand_score_ = rand_score(clusterdata.subclasses, clusterdata.labels)
print(f"Rand score: {rand_score_}")
print("Adjusted rand score: ", adjusted_rand_score(clusterdata.subclasses, clusterdata.labels))
print("Adjusted mutual info score: ", adjusted_mutual_info_score(clusterdata.subclasses, clusterdata.labels))

total_purity, cluster_details = purity_score_with_details(clusterdata.subclasses, clusterdata.labels)
print_purity_details(total_purity, cluster_details)
print_weighted_purity_by_class(cluster_details)
