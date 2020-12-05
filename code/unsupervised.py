import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import IncrementalPCA
from sklearn import metrics



print("Loading data ...\n")
dataset_matrix = np.load("processed_data.npz.npy")
target = np.load("target.npz.npy")

for principal_components_num in range(50, 300, 30):
    print("Using", principal_components_num, "principal components")

    print("PCA ...")
    pca = IncrementalPCA(n_components=principal_components_num, batch_size=500)
    reduced = pca.fit_transform(dataset_matrix)

    print("Fitting Guassian Mixture ...")
    gmm = GaussianMixture(n_components=13)
    prediction = gmm.fit_predict(reduced)
    score = metrics.normalized_mutual_info_score(target, prediction)
    print("--------NIM score:", score, "--------\n")
