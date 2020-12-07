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
    def __init__(self, n_features, n_classes, hidden_size1, hidden_size2, activation):
        super().__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=hidden_size1)
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2)
        self.fc3 = nn.Linear(in_features=hidden_size2, out_features=n_classes)
        self.activation = activation
    
    def forward(self, X):
        out = self.activation(self.fc1(X))
        out = self.activation(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
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

def cross_validation(data_point, target, hidden_size1, hidden_size2, activation, weight_decay, k_fold=5, batch_size=512):
    total_size = len(data_point)
    fraction = 1 / k_fold
    segment = int(total_size * fraction)
    validation_score = 0
    training_score = 0
    for i in range(k_fold):
        training_left = list(range(0, i * segment))
        testing_indices = list(range(i * segment, (i + 1) * segment))
        training_right = list(range((i + 1) * segment, total_size))
        trainig_indices = training_left + training_right


        X_training = data_point[trainig_indices, :]
        X_testing = data_point[testing_indices, :]
        y_training = target[trainig_indices, :]
        y_testing = target[testing_indices, :]


        training_data_loader = get_loader(X_training, y_training, batch_size)
        testing_data_loader = get_loader(X_testing, y_testing, batch_size)

        model = NNClassifier(n_features=X_training.shape[1], n_classes=target.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, activation=activation).to(device)
        optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
        num_epoch = 40

        for epo in range(num_epoch):
            batch_losses = []
            # loop = tqdm(enumerate(training_data_loader), total=len(training_data_loader), leave=False)
            for BOW, targets in training_data_loader:
                BOW, targets = BOW.to(device), targets.to(device)

                optimizer.zero_grad()

                model_result = model(BOW.type(torch.float)).to(device)
                loss = F.binary_cross_entropy(model_result, targets.type(torch.float))

                batch_loss_value = loss.item()
                loss.backward()
                optimizer.step()
                batch_losses.append(batch_loss_value)

            loss_value = np.mean(batch_losses)

        model.eval()
        with torch.no_grad():
            model_result = []
            targets = []
            for BOW, batch_targets in training_data_loader:
                BOW = BOW.to(device)
                model_batch_result = model(BOW.type(torch.float))
                model_result.extend(model_batch_result.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
            result = calculate_metrics(np.array(model_result), np.array(targets))
            training_score += result["samples/f1"]

        with torch.no_grad():
            model_result = []
            targets = []
            for BOW, batch_targets in testing_data_loader:
                BOW = BOW.to(device)
                model_batch_result = model(BOW.type(torch.float))
                model_result.extend(model_batch_result.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
            result = calculate_metrics(np.array(model_result), np.array(targets))
            validation_score += result["samples/f1"]
    validation_score = validation_score / k_fold
    training_score = training_score / k_fold
    return validation_score, training_score

        

batch_size = 128
print("Loading data ...")
dataset_matrix = np.load("processed_data_4000.npz.npy")
target = np.load("target_4000.npz.npy")

# splitting into training and testing
X_training, X_testing, y_training, y_testing = model_selection.train_test_split(dataset_matrix, target, test_size=0.2)
print(X_training.shape)
print(y_training.shape)

# # data augmentation
# low_amount_labels = set()
# low_amount_labels_threshold = 3000
# label2count = dict()
# for y in y_training:
#     label = 0
#     for value in y:
#         if value == 1:
#             if label not in label2count.keys():
#                 label2count.update({label: 1})
#             else:
#                 label2count[label] += 1
#         label += 1
# for label in label2count.keys():
#     if label2count[label] < low_amount_labels_threshold:
#         low_amount_labels.add(label)

# low_amount_data = dict()
# for i in tqdm(range(X_training.shape[0])):
#     label = 0
#     for value in y_training[i]:
#         if value == 1:
#             if label in low_amount_labels:
#                 if label not in low_amount_data.keys():
#                     low_amount_data.update({label: [[X_training[i]], [y_training[i]]]})
#                 else:
#                     low_amount_data[label][0].append(X_training[i])
#                     low_amount_data[label][1].append(y_training[i])
#         label += 1

# X_training_list = [x for x in X_training]
# y_training_list = [y for y in y_training]

# for label in low_amount_data.keys():
#     print("augmenting label", label)
#     data = low_amount_data[label]
#     # iteratively sample 2 data points with replacement
#     # and then interpolate them to create a new data point
#     for i in range(1000):
#         sample = np.random.choice(len(data[0]), 2)
#         interpolation_coefficient = np.random.beta(2, 2)
#         interpolation_coefficient = 0.5
#         new_data_point = interpolation_coefficient * data[0][sample[0]] + (1 - interpolation_coefficient) * data[0][sample[1]]
#         new_target = np.any([data[1][sample[0]].astype(int), data[1][sample[1]].astype(int)], axis=0)
#         X_training_list.append(new_data_point)
#         y_training_list.append(new_target)
# X_training = np.array(X_training_list)
# y_training = np.array(y_training_list)
# print(X_training.shape)
# print(y_training.shape)


print("PCA ...")
print("original data: ", dataset_matrix.shape)
pca = IncrementalPCA(n_components=800, batch_size=1000)
X_training = pca.fit_transform(X_training)
X_testing = pca.transform(X_testing)
print("training data: ", X_training.shape)
print("testing data: ", X_testing.shape)


device = torch.device('cuda')
activation_functions = {nn.ReLU(), torch.tanh}
size_hidden1 = range(100, 500, 50)
size_hidden2 = range(100, 500, 50)
regularization_coefficients = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

print(cross_validation(X_training, y_training, hidden_size1=300, hidden_size2=100, activation=torch.tanh, weight_decay=1e-10))



# validation_score = list()
# training_score = list()
# for coefficient in regularization_coefficients:
#     result = cross_validation(X_training, y_training, hidden_size1=300, hidden_size2=100, activation=torch.tanh, weight_decay=coefficient)
#     validation_score.append(result[0])
#     training_score.append(result[1])

# fig, ax = plt.subplots()
# ax.set_xscale('log')
# ax.set_xticks(regularization_coefficients)
# ax.plot(regularization_coefficients, validation_score, 'rs-', label="validation score")
# ax.plot(regularization_coefficients, training_score, 'b^-', label="training score")
# ax.set_xlabel("Regularization Coefficients")
# ax.set_ylabel("F1-measure")
# ax.set_title("Example-based F1-measure under different Regularization Coefficients")
# ax.legend()
# plt.savefig('RC')

# validation_score = list()
# training_score = list()
# for size1 in size_hidden1:
#     result = cross_validation(X_training, y_training, hidden_size1=size1, hidden_size2=100, activation=torch.tanh, weight_decay=1e-4)
#     validation_score.append(result[0])
#     training_score.append(result[1])

# fig, ax = plt.subplots()
# ax.plot(size_hidden1, validation_score, 'rs-', label="validation score")
# ax.plot(size_hidden1, training_score, 'b^-', label="training score")
# ax.set_xlabel("size of hidden layer 1")
# ax.set_ylabel("F1-measure")
# ax.set_title("Example-based F1-measure under size of hidden layer 1")
# ax.legend()
# plt.savefig('HL1')

# activation_optimal = None
# size_hidden1_optimal = 50
# size_hidden2_optimal = 50
# regularization_coefficient_optimal = 1e-1

# max_score = 0
# for activation in activation_functions:
#     for size1 in size_hidden1:
#         for size2 in size_hidden2:
#             for coefficient in regularization_coefficients:
#                 score = cross_validation(X_training, y_training, hidden_size1=size1, hidden_size2=size2, activation=activation, weight_decay=coefficient)
#                 print(score)
#                 if score > max_score:
#                     max_score = score
#                     activation_optimal = activation
#                     size_hidden1_optimal = size1
#                     size_hidden2_optimal = size2
#                     regularization_coefficient_optimal = coefficient

# print(activation_optimal, size_hidden1_optimal, size_hidden2_optimal, regularization_coefficient_optimal)

# model = NNClassifier(5000, target.shape[1]).to(device)
# model.train()
# learning_rate = 1e-3
# num_epoch = 40
# weight_decay = 1e-3
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# criterion = nn.BCELoss()

# for epo in range(num_epoch):
#     batch_losses = []
#     loop = tqdm(enumerate(training_data_loader), total=len(training_data_loader), leave=False)
#     for batch_index, (BOW, targets) in loop:
#         BOW, targets = BOW.to(device), targets.to(device)

#         optimizer.zero_grad()

#         model_result = model(BOW.type(torch.float)).to(device)
#         loss = F.binary_cross_entropy(model_result, targets.type(torch.float))

#         batch_loss_value = loss.item()
#         loss.backward()
#         optimizer.step()
#         batch_losses.append(batch_loss_value)

#     loss_value = np.mean(batch_losses)
#     print(f"epoch{epo} --- loss: {loss_value}")

#     model.eval()
#     with torch.no_grad():
#         model_result = []
#         targets = []
#         for BOW, batch_targets in dev_data_loader:
#             BOW = BOW.to(device)
#             model_batch_result = model(BOW.type(torch.float))
#             model_result.extend(model_batch_result.cpu().numpy())
#             targets.extend(batch_targets.cpu().numpy())
#         result = calculate_metrics(np.array(model_result), np.array(targets))
#         print(result['samples/f1'])
#         model.train()

# model.eval()
# with torch.no_grad():
#     model_result = []
#     targets = []
#     for BOW, batch_targets in testing_data_loader:
#         BOW = BOW.to(device)
#         model_batch_result = model(BOW.type(torch.float))
#         model_result.extend(model_batch_result.cpu().numpy())
#         targets.extend(batch_targets.cpu().numpy())
#     result = calculate_metrics(np.array(model_result), np.array(targets))
#     print(result)