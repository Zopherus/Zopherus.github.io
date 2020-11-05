![infographics](/images/proposal.png)

## Introduction
Document classification is a traditional problem in text mining and Natural Language Processing. Its applications are of broad range and one of them is management of library or any searching service on scholarly paper. It is believed that in the near future library work will be impacted significantly by artificial intelligence and machine learning systems (Griffey, 2019). The goal of our project is to design and compare machine learning systems that assign topics to research papers given their abstracts.

The dataset we use is the [arXiv Dataset](https://www.kaggle.com/Cornell-University/arxiv) on Kaggle. It contains more than 1.7 million scholarly paper across STEM and is a mirror of the original ArXiv dataset maintained by Cornell University. Among the features, two most useful for us is categories (will be used in the supervised learning part) and abstract.

## Unsurpervised Learning
We conducted unsupervised learning on our dataset of Computer Science articles via clustering. The purpose of doing the unsupervised learning is to generate a clustering that will assist in training a better supervised classifier later. 
### Data Cleaning and Feature Extraction
We examined up to 100000 articles from the 1.7 million articles (of all fields) offered on arXiv, and selected the articles in the Computer Science category (5245 CS articles). The Computer Science category itself has several dozen subcategories, so we use those subcategories as ground truth labels to evaluate our clustering.

From the Computer Science articles, we examine their abstracts and preprocess the text before we build the features representations of them. We remove the stop words from all the abstracts. The reference of English stop words we use is from NLTK (The Natural Language Toolkit) (Bird et al., 2009). Many of the stop words are pronouns, prepositions, and conjunctions such as “i”, “me” and “if”. Besides, we only keep the words that do not contain digits. The last step is to lemmatize the words. After lemmatization, “cat” and “cats” will be treated as the same word.

For performance reasons, we trim the vocabulary to only the most frequently used K (hyperparameter) number of words. Then, we extract the features using the bag-of-words method: each article represents a data point, where each feature corresponds to the frequency of a word appearing.

We used the term frequency–inverse document frequency (tf-idf) (Sparck Jones, K. 1972) statistic to modify our bag of words matrix to have each word weighted. The purpose of this modification is that we do not want to treat the words that occur in almost any documents in the same way as we treat the words that only occur in a small number of documents. We plot correlation coefficients between features. However, the correlation matrix before and after were nearly identical. This is expected since the correlation coefficients are normalized against the standard deviation of the data.

<img src="https://latex.codecogs.com/gif.latex?\rho=\dfrac{Cov(X,Y)}{\sigma_x&space;\sigma_y}" title="\rho=\dfrac{Cov(X,Y)}{\sigma_x \sigma_y}" />

<img src="/images/original-correlation.png" width="200" height="200" />

The visualization for the covariance matrices before and after applying tf-idf show the change of scale.

## Supervised Learning
We plan to train various supervised classifiers including Naive Bayes, SVM, and neural network on our data. Note that different models utilize different feature extraction techniques. For neural network, we will use the cross-entropy loss function.

## Result
We look to see how closely the unsupervised learning clusters matches the ground truth clusters and whether it is prone to outliers. We can compare our clusters to the ground truth clusters to see which ones better match the data. For the supervised learning, we will see the accuracy to which the classifier works on other papers.

## Discussion
The best outcome would be for the supervised learning classification to have a high accuracy. This would mean that the words in the title and abstract tell you what topic a paper is in. We plan to submit our work to the Kaggle competition as well as seeing if Arxiv or jounals would want to use this for automatic topic selection.

## Reference
1. Griffey, J., Yelton, A., Kim, B., & Boman, C. (2019). Artificial Intelligence and Machine Learning in Libraries. ALA TechSource.
2. Kowsari, K.; Jafari Meimandi, K.; Heidarysafa, M.; Mendu, S.; Barnes, L.; Brown, D. Text Classification Algorithms: A Survey. Information 2019, 10, 150.
3. Anupriya, P. & Karpagavalli, S.. (2015). LDA based topic modeling of journal abstracts. 1-5. 10.1109/ICACCS.2015.7324058. 
