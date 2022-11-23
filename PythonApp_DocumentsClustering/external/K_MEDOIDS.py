# pip install scikit-learn-extra
# https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
# https://scikit-learn-extra.readthedocs.io/en/stable/modules/cluster.html#k-medoids

from sklearn_extra.cluster import KMedoids
import numpy as np
from sklearn.datasets import make_blobs

n_samples = 20
X, labels_true = make_blobs(n_samples=n_samples, random_state=1)
print(X)
print(f'Labels Ground Truth = {labels_true}') 

kmedoids = KMedoids(
    n_clusters = 3, # The number of clusters(medoids) to generate (default= 8) 
    metric = 'euclidean', # What distance metric to use (default= 'euclidean')
    method = 'alternate', # Which algorithm to use. 'alternate' is faster while 'pam' is more accurate. (default= 'alternate')
    init= 'k-medoids++', # medoid initialization method. 'random'(ALTERNATIVE), 'heuristic'(ALTERNATIVE) , 'k-medoids++'(ALTERNATIVE), 'build' (PAM)
    max_iter = 3, # Maximum number of iterations of algorithm for a single run
    random_state = 1 # static random medoids (= int number), used to initialise medoids when init='random' 
)

kmedoidsFited= kmedoids.fit(X)
print(f'Clusters(Medoids) = {kmedoidsFited.cluster_centers_}')
print(f'Labels prediction = {kmedoidsFited.labels_}') 
print(f'Sum of distances of samples to their closest cluster center= {kmedoidsFited.inertia_}')
print(f'Index sto pinaka twn data X ta opoia einai ta medoids pou epilexthikan= {kmedoidsFited.medoid_indices_}')
