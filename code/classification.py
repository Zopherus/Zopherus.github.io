import json
import os
import numpy as np
from sklearn import model_selection
from sklearn.decomposition import IncrementalPCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("Loading data ...")
dataset_matrix = np.load("processed_data.npz.npy")
target = np.load("target.npz.npy")

X_training, X_testing, y_training, y_testing = model_selection.train_test_split(dataset_matrix, target, test_size=0.5)


print("PCA ...")
print(dataset_matrix.shape)
pca = IncrementalPCA(n_components=50, batch_size=500)
X_training = pca.fit_transform(X_training)
X_testing = pca.transform(X_testing)


classifier = SVC()

print("Trainig SVM ...")
classifier.fit(X_training, y_training)

pred = classifier.predict(X_testing)

print("-------accuracy: ", accuracy_score(pred, y_testing), "----------")