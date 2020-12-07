import matplotlib.pyplot as plt

import sys
from tqdm import tqdm


import numpy as np

from sklearn import model_selection
from sklearn.decomposition import IncrementalPCA
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("Loading data ...")
dataset_matrix = np.load("processed_data_4000.npz.npy")
target = np.load("target_4000.npz.npy")

# splitting into training and testing
X_training, X_testing, y_training, y_testing = model_selection.train_test_split(dataset_matrix, target, test_size=0.2)
print(X_training.shape)
print(y_training.shape)

print("PCA ...")
print("original data: ", dataset_matrix.shape)
pca = IncrementalPCA(n_components=50, batch_size=1000)
X_training = pca.fit_transform(X_training)
X_testing = pca.transform(X_testing)
print("training data: ", X_training.shape)
print("testing data: ", X_testing.shape)

num_classes = y_training.shape[1]

binary_target = list()
for i in range(num_classes):
    binary_target.append(y_training[:, i])

classifiers = list()
for i in range(num_classes):
    if np.all(binary_target[i] == 0):
        classifiers.append(0)
    elif np.all(binary_target[i] == 1):
        classifiers.append(1)
    else:
        classifier = SVC()
        classifier.fit(X_training, binary_target[i])
        classifiers.append(classifier)

# test
pred_list = list()
for i in range(num_classes):
    if type(classifiers[i]) != int:
        pred_list.append(classifiers[i].predict(X_testing))
    elif classifiers[i] == 0:
        pred_list.append(np.zeros(X_testing.shape[0]))
    else:
        pred_list.append(np.ones(X_testing.shape[0]))
pred = np.zeros((X_testing.shape[0], num_classes))
for i in range(len(pred_list)):
    pred[:, i] = pred_list[i]
print(np.sum(pred))

target = y_testing
scores = {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples')
            }
print(scores)