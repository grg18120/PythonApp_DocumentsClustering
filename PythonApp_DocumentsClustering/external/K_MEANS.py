from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 10
random_state = 1
X, labels_true = make_blobs(n_samples=n_samples, random_state=random_state)
print(f'Labels Ground Truth = {labels_true}') 

kmeans = KMeans(
    n_clusters=3, # The number of clusters(centroids) to generate.
    init='k-means++', # Create cendroids with 1) k-means++ 2) 'random': initialize centoids from picking random points form X
    random_state=None, # static random cendroids (= int number)
    n_init = 10, # Number of time the k-means algorithm will be run with different centroid seeds
    max_iter = 300, # Maximum number of iterations of the k-means algorithm for a single run
    tol = 1e-4 # Sum Square Distance(Cluster_prev, Clusters_now)<=tol then clusters stop moving                
)

#sample_weight = [1,1,1,1,1]
#kmeansFited= kmeans.fit(X,sample_weight)
kmeansFited= kmeans.fit(X)
print(f'Clusters(Centroids) = {kmeansFited.cluster_centers_}')
print(f'Labels prediction = {kmeansFited.labels_}') 
print(f'Sum of distances of samples to their closest cluster center = {kmeansFited.inertia_}')
print(f'Number of iterations has run to stop centroids moving= {kmeansFited.n_iter_}')

#print(kmeansFited.n_features_in_)
#print(kmeansFited.feature_names_in_)





