from sklearn import metrics 

# "external" evaluation
# clustering is compared to an existing "ground truth" classification


labels_true = [0, 0, 0, 1, 1, 1] # classes C
labels_pred = [0, 0, 1, 1, 2, 2] # clusters K


# Rand-Index metric
# [0, 1] for the unadjusted Rand index and [-1, 1] for the adjusted Rand index.
evaluationRI = metrics.rand_score(labels_true, labels_pred)
print(f'Rand-Index evaluation = {evaluationRI}')
evaluationARI = metrics.adjusted_rand_score(labels_true, labels_pred)
print(f'Adjusted Rand-Index evaluation = {evaluationARI}\n')

# Fowlkes-Mallows metric
# Values close to zero indicate two label assignments that are largely independent, 
# while values close to one indicate significant agreement. 
# Further, values of exactly 0 indicate purely independent label assignments and a FMI of exactly 1 indicates that 
# the two label assignments are equal 
evaluationFMI = metrics.fowlkes_mallows_score(labels_true, labels_pred)
print(f'Fowlkes-Mallows evaluation = {evaluationFMI}\n')

# Homogeneity(h), completeness(c) and V-measure metric
# V-measure = [(1+b)*h*c] / [(b*h + c)]   b = 1.0(default)
# 0.0 is as bad as it can be, 1.0 is a perfect score.
h = metrics.homogeneity_score(labels_true, labels_pred)
c = metrics.completeness_score(labels_true, labels_pred)
evaluationV = metrics.v_measure_score(labels_true, labels_pred, beta = 1.0) # = ((1+b)*h*c) / ((b*h + c))   b = 1.0
print(f'homogenity = {h} & completeness = {c}')
print(f'V-measure evaluation = {evaluationV}\n')

# Mutual Information
evaluationMI = metrics.mutual_info_score(labels_true, labels_pred)
print(f'Mutual Information evaluation = {evaluationMI}')
evaluationAMI = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
print(f'Adjusted Mutual Information evaluation = {evaluationMI}')
evaluationNMI = metrics.normalized_mutual_info_score(labels_true, labels_pred)
print(f'Normalized Mutual Information evaluation = {evaluationMI}\n')
