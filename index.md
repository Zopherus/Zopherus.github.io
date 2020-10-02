![infographics](/images/proposal.png)

## Introduction
Document classification is a traditional problem in text mining and Natural Language Processing. Its applications are of broad range and one of them is management of library or any searching service on scholarly paper. It is believed that in the near future library work will be impacted significantly by artificial intelligence and machine learning systems (Griffey, 2019). The goal of our project is to design and compare machine learning systems that assign topics to research papers given their abstracts.

The dataset we use is the [arXiv Dataset](https://www.kaggle.com/Cornell-University/arxiv) on Kaggle. It contains more than 1.7 million scholarly paper across STEM and is a mirror of the original ArXiv dataset maintained by Cornell University. Among the features, two most useful for us is categories (will be used in the supervised learning part) and abstract.
## Method
### Feature Extraction and Dimensionality Reduction
Bag-of-words and word embedding are two classic methods to get a feature representation of a block of text. They are preferred in different models. We will then reduce the dimensionality of the data by applying techniques such as Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), or non-negative matrix factorization (NMF) (Kowsari, 2019). 

### Unsupervised Learning
For unsupervised learning, we extract the features in the baf-of-words style. After applying dimensionality reduction, we train K-means and Gaussian Mixture Model(GMM) on the data. We go through the hyperparameter tuning process to determine the optimal hyperparameters, such as number of classes, for those models.

### Supervised Learning
We plan to train various supervised classifiers including Naive Bayes, SVM, and neural network on our data. Note that different models utilize different feature extraction techniques. For neural network, we will use the cross-entropy loss function.

## Result
We look to see how closely the unsupervised learning clusters matches the ground truth clusters and whether it is prone to outliers. We can compare our clusters to the ground truth clusters to see which ones better match the data. For the supervised learning, we will see the accuracy to which the classifier works on other papers.

## Discussion
The best outcome would be for the supervised learning classification to have a high accuracy. This would mean that the words in the title and abstract tell you what topic a paper is in. We plan to submit our work to the Kaggle competition as well as seeing if Arxiv or jounals would want to use this for automatic topic selection.

## Reference
1. Griffey, J., Yelton, A., Kim, B., & Boman, C. (2019). Artificial Intelligence and Machine Learning in Libraries. ALA TechSource.
2. Kowsari, K.; Jafari Meimandi, K.; Heidarysafa, M.; Mendu, S.; Barnes, L.; Brown, D. Text Classification Algorithms: A Survey. Information 2019, 10, 150.
3. Anupriya, P. & Karpagavalli, S.. (2015). LDA based topic modeling of journal abstracts. 1-5. 10.1109/ICACCS.2015.7324058. 
