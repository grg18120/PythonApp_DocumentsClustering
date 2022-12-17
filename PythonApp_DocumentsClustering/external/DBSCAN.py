from sklearn.cluster import DBSCAN
import numpy as np

X = np.array([[1, 2], [2, 2], [2, 3],
              [8, 7], [8, 8], [25, 80]])
dbscan = DBSCAN(
    eps = 3, # The maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples = 2,# number of samples in a neighborhood for a point to be considered as a core point includes the point itself.
    algorithm = 'auto', # find count(neirest neighbours) of each points {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
    leaf_size = 2, # when a node converted to leaf (points into node <= leaf_size) default = 30
    p = 2, # The power of the Minkowski metric to be used to calculate distance between points. If None, then p=2
)

kmeansFited= dbscan.fit(X)

print(f"core samples indxs: {kmeansFited.core_sample_indices_}")
print(f"core samples : {kmeansFited.components_}")
print(f"labels prediction (Not clustered samples are given the label -1) : {kmeansFited.labels_}")
print(f"core samples : {kmeansFited.n_features_in_}")
