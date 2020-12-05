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

import torch.optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class NNClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=n_classes)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        out = self.relu(self.fc1(X))
        out = self.fc2(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out

class ArXivDataset(Dataset):
    def __init__(self, BOW_matrix, target):
        self.BOW_matrix = BOW_matrix
        self.target = target
    def __len__(self):
        return self.target.shape[0]
    
    def __getitem__(self, index):
        return (torch.tensor(self.BOW_matrix[index].T), torch.tensor(self.target[index]))

def get_loader(BOW_matrix, target, batch_size):
    dataset = ArXivDataset(BOW_matrix, target)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=6)
    return loader

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples')
            }

batch_size = 32
print("Loading data ...")
dataset_matrix = np.load("processed_data.npz.npy")
target = np.load("target.npz.npy")

X_training, X_testing, y_training, y_testing = model_selection.train_test_split(dataset_matrix, target, test_size=0.3)

# print("PCA ...")
# print("original data: ", dataset_matrix.shape)
# pca = IncrementalPCA(n_components=300, batch_size=1000)
# X_training = pca.fit_transform(X_training)
# X_testing = pca.transform(X_testing)
# print("training data: ", X_training.shape)
# print("testing data: ", X_testing.shape)

training_data_loader = get_loader(X_training, y_training, batch_size)
testing_data_loader = get_loader(X_testing, y_testing, batch_size)

device = torch.device('cpu')
model = NNClassifier(5000, target.shape[1])
model.train()
learning_rate = 1e-3
num_epoch = 40
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

for epo in range(num_epoch):
    batch_losses = []
    loop = tqdm(enumerate(training_data_loader), total=len(training_data_loader), leave=False)
    for batch_index, (BOW, targets) in loop:
        BOW, targets = BOW.to(device), targets.to(device)

        optimizer.zero_grad()

        model_result = model(BOW.type(torch.float)).to(device)
        loss = F.binary_cross_entropy(model_result, targets.type(torch.float))

        batch_loss_value = loss.item()
        loss.backward()
        optimizer.step()
        batch_losses.append(batch_loss_value)

    loss_value = np.mean(batch_losses)
    print(f"epoch{epo} --- loss: {loss_value}")

    model.eval()
    with torch.no_grad():
        model_result = []
        targets = []
        for BOW, batch_targets in testing_data_loader:
            BOW = BOW.to(device)
            model_batch_result = model(BOW.type(torch.float))
            model_result.extend(model_batch_result.cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())
        result = calculate_metrics(np.array(model_result), np.array(targets))
        print(result)
        model.train()
# classifier = SVC(class_weight="balanced")
# classifier = SVC()

# print("Trainig SVM ...")
# classifier.fit(X_training, y_training)

# print("Infering on test data ...")
# pred = classifier.predict(X_testing)
# print(metrics.precision_recall_fscore_support(y_testing, pred, beta=0.5, average=None))
# # print("-------balanced accuracy:", balanced_accuracy_score(pred, y_testing), "----------")

# # plot_confusion_matrix(classifier, X_testing, y_testing)
# # plt.show()