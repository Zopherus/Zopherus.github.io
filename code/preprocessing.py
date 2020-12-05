import os
import sys
import string
import numpy as np
import matplotlib.pyplot as plt
import re
from multiprocessing import Pool


from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import load_files
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer


import pickle

import json
import time
from tqdm import tqdm
import ijson

from yellowbrick.features import Rank2D

print("Reading json file ...")
reading_file_start = time.time()


##--------- input json file where each line is a json object. -----------------------------
with open(os.path.join("arxiv-metadata-oai-snapshot.json")) as file:
    lines = file.readlines()
print("\t use ",(time.time() - reading_file_start), "seconds")


print("Loading dataset ...")
data = [] # a list of json object representing an article
abstract_list = [] # a list of abstracts
target = [] # a list of categories for each abstract

# load data, encode class label
num_lines = 500000
num_labels_threshold = 4000
line_count = 0
label2index = dict()
label_index = 0
label2count = dict()

# count class numbers
for line in lines:
    if line_count >= num_lines:
        break
    line_count += 1
    article = json.loads(line)
    categories_entry = article["categories"].split()
    has_CS = False
    class_labels = []
    for category in categories_entry:
        if category.split(".")[0] == "cs":
            has_CS = True
            class_labels.append(category.split(".")[1])
    if has_CS:
        for label in class_labels:
            if label not in label2count.keys():
                label2count.update({label: 1})
            else:
                label2count[label] += 1
sorted_label_count = sorted(label2count.items(), key=lambda x: x[1], reverse=True)
print(sorted_label_count)
desired_class = set()
for label, count in label2count.items():
    if count >= num_labels_threshold:
        desired_class.add(label)

label2index = dict()
index = 0
for label in desired_class:
    label2index.update({label: index})
    index += 1
print(label2index)

# load valid data
line_count = 0
for line in lines:
    if line_count >= num_lines:
        break
    line_count += 1
    article = json.loads(line)
    categories_entry = article["categories"].split()
    has_CS = False
    class_labels = []
    for category in categories_entry:
        if category.split(".")[0] == "cs":
            has_CS = True
            class_labels.append(category.split(".")[1])
    if has_CS:
        has_desired_label = False
        label_vector = np.zeros(len(label2index))
        for label in class_labels:
            if label in desired_class:
                has_desired_label = True
                label_vector[label2index[label]] = 1
        if has_desired_label:
            abstract_list.append(article["abstract"])
            target.append(label_vector)
target = np.array(target)
np.save(f"target_{num_labels_threshold}.npz", target)
print(target.shape)

# label_count = []
# for label, count in sorted_label_count:
#     label_count.append(count)
# plt.bar(range(0,len(sorted_label_count)), label_count)
# plt.xlabel("class")
# plt.ylabel("frequency")
# plt.title("class distribution for 100,000 articles")
# plt.savefig("class-distribution.png", dpi=600)

#data cleaning
print("Cleaning ...")
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
stopwords = set(stopwords.words("english"))
cleaned_abstract_list = []
word_Lemmatized = WordNetLemmatizer()
for abstract in tqdm(abstract_list):
    words = word_tokenize(abstract)
    content = []
    for word, tag in pos_tag(words):
        if word.isalpha() and word not in stopwords:
            word = word_Lemmatized.lemmatize(word, pos=tag_map[tag[0]])
            content.append(word)
    cleaned_abstract_list.append(words)


# build vocabulary, only include words with top k highest frequencies
print("Creating bag of words ...")
bow_start = time.time()
word_frequency = dict()
for abstract in tqdm(cleaned_abstract_list):
    for word in abstract:
        if word not in word_frequency.keys():
            word_frequency.update({word: 1})
        else:
            word_frequency[word] += 1
sorted_word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True) # sort the word frequency table by the frequency

vocabulary = set()
k = 5000
count = 1
for word, _ in sorted_word_frequency:
    if count > k:
        break
    # if count < 10:
    #     print(word)
    vocabulary.add(word)
    count += 1


# create map from word to index
word2index = dict()
index = 0
for word in vocabulary:
    word2index.update({word:index})
    index += 1

# create bag of words
print(len(abstract_list), len(word2index), sep=" ")
dataset_matrix = np.zeros((len(abstract_list),len(word2index)))
data_index = 0
for abstract in tqdm(cleaned_abstract_list):
    for word in abstract:
        if word in vocabulary:
            dataset_matrix[data_index, word2index[word]] += 1
    data_index += 1

# # correlation = np.corrcoef(dataset_matrix[:,0:50].T)
# # plt.figure()
# # plt.matshow(correlation)
# # plt.colorbar()
# # plt.title("Correlation Between Features")
# # plt.savefig("original-correlation.png", dpi=600)

# # correlation_matrix = np.corrcoef(dataset_matrix[:, 0:2000].T)
# # index2correlation = dict()
# # for i in range(correlation_matrix.shape[0]):
# #     for j in range(correlation_matrix.shape[1]):
# #         if i != j:
# #             index2correlation.update({(i,j): correlation_matrix[i, j]})
# # sorted_correlation = sorted(index2correlation.items(), key=lambda x: x[1], reverse=True)
# # for i in range(50):
# #     index, correlation = sorted_correlation[i]
# #     print(vocabulary[index[0]], " ,", vocabulary[index[1]])
# #     print(correlation)

# inverse document frequency
document_frequency = np.zeros((len(word2index), 1))
for abstract in tqdm(cleaned_abstract_list):
    for word in abstract:
        if word in word2index.keys():
            document_frequency[word2index[word]] += 1
idf = np.log10(len(abstract_list) / (document_frequency + 1))


# tf-idf

dataset_matrix = dataset_matrix * idf.T
np.save(f'processed_data_{num_labels_threshold}.npz', dataset_matrix)

# correlation = np.corrcoef(dataset_matrix[:,0:50].T)
# plt.figure()
# plt.matshow(correlation)
# plt.colorbar()
# plt.title("Correlation Between Features After tf-idf")
# plt.savefig("correlation-tf-idf.png", dpi=600)



# visualizer = Rank2D(algorithm="pearson")
# visualizer.fit(dataset_matrix[:, 0:50], target)
# visualizer.transform(dataset_matrix[:, 0:50])
# visualizer.show()

# pca = IncrementalPCA(n_components=4000, batch_size=5000)
# data_reduced = pca.fit_transform(dataset_matrix)
# variance_explained = pca.explained_variance_ratio_.cumsum()
# plt.figure()
# plt.plot(range(1, 4001, 1), variance_explained, 'ro-')
# plt.xlabel("number of principal components")
# plt.ylabel("variance explained")
# plt.title("variance explained in different number of components")
# plt.savefig("pac-variance.png", dpi=600)


# pca = IncrementalPCA(n_components=2000, batch_size=5000)
# data_reduced = pca.fit_transform(dataset_matrix)
# gmm = GaussianMixture(n_components=40, max_iter=200)
# prediction = gmm.fit_predict(data_reduced)
# print("NMI socre = ", metrics.normalized_mutual_info_score(target, prediction))
# prediction_count_dict = dict()
# for label in prediction:
#     if label not in prediction_count_dict.keys():
#         prediction_count_dict.update({label: 1})
#     else:
#         prediction_count_dict[label] += 1
# sorted_prediction_count = sorted(prediction_count_dict.items(), key=lambda x: x[1], reverse=True)
# print(sorted_prediction_count)
# plt.figure()
# plt.bar(range(0, len(sorted_prediction_count)), [p[1] for p in sorted_prediction_count])
# plt.xlabel("class")
# plt.ylabel("frequency")
# plt.title("class distribution in the result of GMM for 100,000 articles \n n_components=50")
# plt.savefig("GMM-distribution.png", dpi=600)

# num_principal_comppnent = [90]
# scores = []
# for i in range(len(num_principal_comppnent)):
#     # PCA
#     print("Start PCA")
#     pca_start = time.time()
#     pca = IncrementalPCA(n_components=num_principal_comppnent[i], batch_size=4000) # performance under different number of clusters?
#     # chunk_size = 1000
#     # for i in range(dataset_matrix.shape[0] // chunk_size):
#     #     pca.partial_fit(dataset_matrix[i*chunk_size: (i+1)*chunk_size])
#     data_reduced = pca.fit_transform(dataset_matrix)
#     print("Finish PCA using",(time.time() - pca_start), "s")
#     print(data_reduced.shape)

#     # K-means vs GMM

#     # # K-means
#     # print("Start K-means")
#     # kmeans_start = time.time()
#     # kmeans = KMeans(n_clusters=len(label2index))
#     # prediction = kmeans.fit_predict(data_reduced)
#     # print("Finish K-means using ", (time.time() - kmeans_start), "s")


#     # GMM
#     print("Start GMM")
#     n_clusters = range(2, 100)
#     scores = []
#     for n in n_clusters:
#         gmm_start = time.time()
#         gmm = GaussianMixture(n_components=n, max_iter=200)
#         prediction = gmm.fit_predict(data_reduced)
#         print("Finish GMM using", (time.time() - gmm_start), "s")


#         print(prediction.shape)

#         # Evaluate clustering using normalized mutual information
#         score = metrics.normalized_mutual_info_score(target, prediction)
#         scores.append(score)
#         print("score: ", score)
#     plt.figure()
#     plt.plot(n_clusters, scores, 'ro-')
#     plt.axis([2,100,0,1])
#     plt.xlabel("number of clusters")
#     plt.ylabel("NMI score")
#     plt.title("NMI score for GMM with different number of components")
#     plt.savefig("nc-score.png", dpi=600)

#     gmm_start = time.time()
#     gmm = GaussianMixture(n_components=40, max_iter=200)
#     prediction = gmm.fit_predict(data_reduced)
#     print("Finish GMM using", (time.time() - gmm_start), "s")


#     print(prediction.shape)

#     # Evaluate clustering using normalized mutual information
#     score = metrics.normalized_mutual_info_score(target, prediction)
#     scores.append(score)
#     print("score: ", score)
# plt.figure()
# plt.plot(num_principal_comppnent, scores, 'ro-')
# plt.xlabel("number of principal components")
# plt.ylabel("NMI score")
# plt.savefig("pc-score.png", dpi=600)

