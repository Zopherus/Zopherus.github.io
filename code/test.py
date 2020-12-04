import os
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

import DataCleaningUtil

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
line_count = 0
label2index = dict()
label_index = 0
label2count = dict()

discarded_class = {"GL"}
desired_class = {"LG", "NI", "CL", "CR"}
for line in lines:
    if line_count >= 1000000:
        break
    line_count += 1
    article = json.loads(line)
    class_labels = article["categories"].split()
    if class_labels[0].split(".")[0] == "cs":
        # if (class_labels[0].split(".")[1] == "CV" and cv_count < 4000) or (class_labels[0].split(".")[1] == "PL" and pl_count < 4000):
        #     if class_labels[0].split(".")[1] == "CV":
        #         cv_count += 1
        #     if class_labels[0].split(".")[1] == "PL":
        #         pl_count += 1
        if class_labels[0].split(".")[1] in desired_class:
            abstract_list.append(article["abstract"])
            if class_labels[0].split(".")[1] not in label2index.keys():
                label2index.update({class_labels[0].split(".")[1]: label_index})
                label_index += 1
            if class_labels[0].split(".")[1] not in label2count.keys():
                label2count.update({class_labels[0].split(".")[1]: 1})
            else:
                label2count[class_labels[0].split(".")[1]] += 1
            target.append(label2index[class_labels[0].split(".")[1]])
target = np.array(target)
sorted_label_count = sorted(label2count.items(), key=lambda x: x[1], reverse=True)
print(sorted_label_count)
label_count = []
for label, count in sorted_label_count:
    label_count.append(count)
print(label2index)
print(len(label2index))
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
for abstract in tqdm(abstract_list):
    words = word_tokenize(abstract)
    content = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(words):
        if word.isalpha() and word not in stopwords:
            word = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            content.append(word)
    cleaned_abstract_list.append(content)

# NUM_PROCESSESS = 6
# pool = Pool(processes=NUM_PROCESSESS)
# chunked_abstract_list = []
# chunck_size = len(abstract_list) // NUM_PROCESSESS
# for i in range(0, len(abstract_list), chunck_size):
#     chunked_abstract_list.append(abstract_list[i: i + chunck_size])
# print(len(chunked_abstract_list))
# results = [pool.apply(DataCleaningUtil.clean, args=(abstract_list,)) for sub_abstract_list in chunked_abstract_list]
# cleaned_abstract_list = []
# for result in results:
#     for abstract in result:
#         cleaned_abstract_list.append(abstract)


# Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(cleaned_abstract_list, target, test_size=0.3)
# Tfidf_vect = TfidfVectorizer(max_features=5000)
# Tfidf_vect.fit(cleaned_abstract_list)
# Train_X_Tfidf = Tfidf_vect.transform(Train_X)
# Test_X_Tfidf = Tfidf_vect.transform(Test_X)
# print("start traning SVM")
# classifier = SVC()
# classifier.fit(Train_X_Tfidf, Train_Y)

# pred = classifier.predict(Test_X_Tfidf)

# print("accuracy: ", accuracy_score(pred, Test_Y))




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
k = 10000
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
dataset_matrix = np.zeros( (len(abstract_list),len(word2index)))
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
for abstract in cleaned_abstract_list:
    for word in abstract:
        if word in word2index.keys():
            document_frequency[word2index[word]] += 1
idf = np.log10(len(abstract_list) / (document_frequency + 1))


# tf-idf

dataset_matrix = dataset_matrix * idf.T

# correlation = np.corrcoef(dataset_matrix[:,0:50].T)
# plt.figure()
# plt.matshow(correlation)
# plt.colorbar()
# plt.title("Correlation Between Features After tf-idf")
# plt.savefig("correlation-tf-idf.png", dpi=600)

# print(dataset_matrix.shape)
# print("Finished creating bag of words matrix using ",(time.time() - bow_start), "s")

print("PCA ...")
print(dataset_matrix.shape)
pca = IncrementalPCA(n_components=100, batch_size=1000)
data_reduced = pca.fit_transform(dataset_matrix)

print(data_reduced.shape)

# data_reduced = dataset_matrix

X_training, X_testing, y_training, y_testing = model_selection.train_test_split(data_reduced, target, test_size=0.5)


classifier = SVC()

print("Trainig SVM ...")
classifier.fit(X_training, y_training)

pred = classifier.predict(X_testing)

print("-------accuracy: ", accuracy_score(pred, y_testing), "----------")




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

