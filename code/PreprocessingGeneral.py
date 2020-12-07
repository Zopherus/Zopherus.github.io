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
num_lines = 50000
num_labels_threshold = 0
line_count = 0
label2index = dict()
label_index = 0
label2count = dict()

discarded_class = {"econ", "q-fin"}
# count class numbers
for line in lines:
    if line_count >= num_lines:
        break
    line_count += 1
    article = json.loads(line)
    categories_entry = article["categories"].split()
    class_labels = []
    for category in categories_entry:
        class_labels.append(category.split(".")[0])
    for label in class_labels:
        if label not in discarded_class:
            if "-" in label:
                if label.split("-")[0] == "hep":
                    label = "hep"
                elif label.split("-")[0] == "nucl":
                    label = "nucl"
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
    class_labels = []
    for category in categories_entry:
        class_labels.append(category.split(".")[0])
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
# np.save("target_toy.npz", target)
np.save(f"target_general.npz", target)
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
    cleaned_abstract_list.append(content)


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
# np.save('processed_data_toy.npz', dataset_matrix)
np.save(f'processed_data_general.npz', dataset_matrix)