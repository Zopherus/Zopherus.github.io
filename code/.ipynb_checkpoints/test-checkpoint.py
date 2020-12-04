import os
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from sklearn.datasets import load_files
from sklearn.decomposition import IncrementalPCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import metrics
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import json
import time
from tqdm import tqdm
import ijson

#  extract a list of words from an abstract
def word_extraction(abstract):
    stopwords = set(stopwords.words('english'))
    words = word_tokenize(abstract)
    cleaned_text = [w.lower() for w in words if w not in stopwords]
    return cleaned_text

# tokenize all abstracts and return the vocabulary
def tokenzie(abstract_list):
    words = []
    for abstract in abstract_list:
        w = word_extraction(abstract)
        words.extend(w)
    vocabulary = sorted(list(set(words)))
    return vocabulary



print("Start reading file")
reading_file_start = time.time()


##--------- input json file where each line is a json object. -----------------------------
with open(os.path.join("arxiv-metadata-oai-snapshot.json")) as file:
    lines = file.readlines()
print("Done reading file using ",(time.time() - reading_file_start), "s")

data = [] # a list of json object representing an article
abstract_list = [] # a list of abstracts
target = [] # a list of categories for each abstract

# load data, encode class label
line_count = 0
label2index = dict()
label_index = 0
for line in lines:
    if line_count >= 10000:
        break
    line_count += 1
    article = json.loads(line)
    class_labels = article["categories"].split()
    if class_labels[0].split(".")[0] == "cs":
        abstract_list.append(article["abstract"])
        if class_labels[0].split(".")[1] not in label2index.keys():
            label2index.update({class_labels[0].split(".")[1]: label_index})
            label_index += 1
        target.append(label2index[class_labels[0].split(".")[1]])
target = np.array(target)
print(label2index)

# build vocabulary, only include words with top k highest frequencies
print("Start creating bag of words")
bow_start = time.time()
word_frequency = dict()
stopwords = set(stopwords.words("english"))
for abstract in tqdm(abstract_list):
    words = word_tokenize(abstract)
    for word in words:
        word = word.lower() # case consistency
        if word not in stopwords:
            if word not in word_frequency.keys():
                word_frequency.update({word: 1})
            else:
                word_frequency[word] += 1
sorted_word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True) # sort the word frequency table by the frequency

vocabulary = set()
k = 3000
count = 1
for word, _ in sorted_word_frequency:
    if count > k:
        break
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
for abstract in tqdm(abstract_list):
    words = [w.lower() for w in word_tokenize(abstract)] #unparsed
    for word in words:
        if word in vocabulary:
            dataset_matrix[data_index, word2index[word]] += 1
    data_index += 1

correlation = np.corrcoef(dataset_matrix[:,0:20].T)
plt.matshow(correlation)
plt.colorbar()
plt.title("Correlation between original frequencies")
plt.show()

# inverse document frequency
document_frequency = np.zeros((len(word2index), 1))
for abstract in abstract_list:
    words = [w.lower for w in word_tokenize(abstract)]
    for word in words:
        if word in word2index.keys():
            document_frequency[word2index[word]] += 1
idf = np.log10(len(abstract_list) / (document_frequency + 1))


# tf-idf
dataset_matrix = dataset_matrix * idf.T

print(dataset_matrix.shape)
print("Finished creating bag of words matrix using ",(time.time() - bow_start), "s")

correlation = np.corrcoef(dataset_matrix[:,0:20].T)
plt.matshow(correlation)
plt.colorbar()
plt.title("Correlation between frequencies after tf-idf")
plt.show()



# num_principal_comppnent = [300]
# scores = []
# for i in range(len(num_principal_comppnent)):
#     # PCA
#     print("Start PCA")
#     pca_start = time.time()
#     pca = IncrementalPCA(n_components=num_principal_comppnent[i], batch_size=1000) # performance under different number of clusters?
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
#     gmm_start = time.time()
#     gmm = GaussianMixture(n_components=len(label2index), max_iter=200)
#     prediction = gmm.fit_predict(data_reduced)
#     print("Finish GMM using", (time.time() - gmm_start), "s")


#     print(prediction.shape)

#     # Evaluate clustering using normalized mutual information
#     score = metrics.normalized_mutual_info_score(target, prediction)
#     scores.append(score)
#     print("score: ", score)
# plt.plot(num_principal_comppnent, scores, 'ro-')
# plt.show()

