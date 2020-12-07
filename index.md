![infographics](/images/proposal.png)

## Introduction
Document classification is a traditional problem in text mining and Natural Language Processing. Its applications are of broad range and one of them is management of library or any searching service on scholarly paper. It is believed that in the near future library work will be impacted significantly by artificial intelligence and machine learning systems (Griffey, 2019). The goal of our project is to design and compare machine learning systems that assign topics to research papers given their abstracts.

The dataset we use is the [arXiv Dataset](https://www.kaggle.com/Cornell-University/arxiv) on Kaggle. It contains more than 1.7 million scholarly paper across STEM and is a mirror of the original ArXiv dataset maintained by Cornell University. Among the features, two most useful for us is categories (will be used in the supervised learning part) and abstract.

## Unsupervised Learning
We conducted unsupervised learning on our dataset of Computer Science articles via clustering. The purpose of doing the unsupervised learning is to generate a clustering that will assist in training a better supervised classifier later. 
### Data Cleaning and Feature Extraction
We examined up to 100000 articles from the 1.7 million articles (of all fields) offered on arXiv, and selected the articles in the Computer Science category (5245 CS articles). The Computer Science category itself has several dozen subcategories, so we use those subcategories as ground truth labels to evaluate our clustering.

From the Computer Science articles, we examine their abstracts and preprocess the text before we build the features representations of them. We remove the stop words from all the abstracts. The reference of English stop words we use is from NLTK (The Natural Language Toolkit) (Bird et al., 2009). Many of the stop words are pronouns, prepositions, and conjunctions such as “i”, “me” and “if”. Additionally, we only keep the words that do not contain digits. The last step is to lemmatize the words. After lemmatization, “cat” and “cats” will be treated as the same word.

For performance reasons, we trim the vocabulary to only the most frequently used K (a hyperparameter) number of words. The default value of K we used is 10000. This also results in the dataset having very few outliers since most articles will be similar to others in the count of commonly used words. We confirmed that there were no outliers using DBSCAN as well. Then, we extract the features using the bag-of-words method: each article represents a data point, where each feature corresponds to the frequency of a word appearing.

We used the term frequency–inverse document frequency (tf-idf) (Sparck Jones, K. 1972) statistic to modify our bag of words matrix to have each word weighted. The purpose of this modification is that we do not want to treat the words that occur in almost any documents in the same way as we treat the words that only occur in a small number of documents. We plot correlation coefficients between features. However, the correlation matrix before and after were nearly identical. This is expected since the correlation coefficients are normalized against the standard deviation of the data.

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\rho=\dfrac{Cov(X,Y)}{\sigma_x&space;\sigma_y}" title="\rho=\dfrac{Cov(X,Y)}{\sigma_x \sigma_y}" />
</p>

Below is the visualization of the correlation matrices before and after applying tf-idf. We can see that most pairs of the features (unique words in the vocabulary) are not correlated.

<img src="/images/original-correlation.png" width="400" height="400" />
<img src="/images/correlation-tf-idf.png" width="400" height="400" />
<em> Figure 1</em>

However, note that these matrices only show the first 50 features. A sorting of pairs of words based on their correlation coefficients shows that there are many correlated features. Some exmaples are:

1. astrobiology, geology (1.0)
2. lifestyle, religion (1.0)
3. Zombie, Ubuntu (1.0)

This indicates that it may be beneficial to apply PCA on our data before doing clustering.

### Dimensionality Reduction and Clustering
We apply principal component analysis to our data. We do this for two purposes. The first goal is to reduce the dimension of the data so running the clustering algorithm such as K-means and GMM is less expensive. Another goal of doing PCA is to avoid the curse of dimensionality since the Bag-of-Word representation usually has a large number of features. Below is the variance explained for different number of principal components.
<p>
  <img src="/images/pac-variance.png" width="600" height="400" />
  <em> Figure 2</em>
</p>

We trained Gaussian Mixture Model (GMM) on the data using 80 principal components and set the number of clusters in GMM to be 50. Below is the distribution of classes in the ground truth and the distribution of classes in the clustering of GMM.
<p>
  <img src="/images/class-distribution.png" width="600" height="400" />
  <em> Figure 3.1</em>
</p>

<p>
  <img src="/images/GMM-distribution.png" width="600" height="400" />
  <em> Figure 3.2</em>
</p>

We go through the hyperparameter tuning process to determine the optimal hyperparameters, such as number of classes, for those models. The results of clustering using GMM may help us train a better supervised classifier later. One technique that may utilize clustering methods is introduced by (Nigam et al. 2000), a semi-supervised method using EM and Naive bayes when labeled data is limited.

### Evaluation
We used the Normalized Mutual Information (NMI) metric to evaluate our clustering, which gives a score from 0 to 1 describing the correlation between the ground truth clusters and our cluster results. The NMI score is computed as

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?NMI(Y,C)=\dfrac{2I(Y,C)}{H(Y)&plus;H(C)}" title="NMI(Y,C)=\dfrac{2I(Y,C)}{H(Y)+H(C)}" />
</p>

### Experiment and Hyperparameter Tuning
Many hyper-parameters are involved in our processing of the texts. For instance, only the words of top k frequencies are used as features in the bag-of-word representation since it is impractical to use all of the unique words as features. There are other hyper-parameters such as the number of principal components in PCA and the number of clusters in GMM.

We compare the NMI scores of GMM when different number of principal components is used in PCA. Below is the result of applying GMM (the number of latent components is set to be the number of classes in the groun truth) when different number of principal components in used in PCA.

<p>
  <img src="/images/pc-score.png" width="600" height="400" />
  <em> Figure 4</em>
</p>

Note that the best NMI scores occur when the number of principal components is around 80-100 and it goes down if more principal components are used. This shows that the dimensionality reduction procedure does improve the result of clustering in GMM.

Although the number of classes in the ground truth is known, we still use the ELBOW method to determine the best choice for the number of clustering. In the ELBOW method, we use 5245 CS articles and we set the maximum number of vocabulary to be 10000 and the number of principal components to be 200. Below is the result of the ELBOW method.

<p>
  <img src="/images/nc-score.png" width="600" height="400" />
  <em> Figure 5</em>
</p>
We can observe that there is little change in NMI score if we set the number of component in GMM to be greater than 30. We can see the reason in the comparison between the distribution of the ground truth and the distribution of the clustering (Figure3.1 and Figure3.2). Note that GMM does a great job of constraining the number of nonzero clusters in the range from 30 to 40 even if the we make GMM to assume that there is 50 latent components in Figure3.2. This explains the little change of NMI score if the number of componenets in GMM is set tio be greater than 30. However, the NMI score of 0.34 is not very satisfactory but it is acceptable considering the imbalanced class distribution and the number of classes being such large. Text classification tends to perform much better with little supervision as opposed to none: common techniques such as sentiment analysis and opinion mining have been performed mostly with supervised learning (Dasgupta et al. 2009)

### Result
In conclusion, we can see from the result of GMM that applying dimensionality reduction using PCA does reduce the neagtive effect of high dimensionality. We observe that the number of principal component that best fits our goal is in the range of 80 to 100. We also observe that the clustering given buy GMM is robust since the number of nonzero clusters is similar to the number of classes in the ground truth. Although the NMI score is low, we believe the performance can be improved if a little bit of supervization is involved. The observations and results from the unsupervised learning will help us implement the semi-supervised or supervised learning in the remaining portion of this project. 


## Supervised Learning
We scale our models up by using 65093 CS articles for the supervised learning section. We choose neural network as the supervised learning technique purposefully. We reobserved the arXiv dataset and found two problems that may negatively affect our supervised learning models. The first problem is that the distribution of classes is very imbalanced as shown in Figure. The second problem is that the difference between many subcategoires in the CS category is ambiguous even to human. This suggests that it may be a bad idea to only use the first-ranked label for each data point. Therefore we form this supervised learning problem as a multiclass, multilabel classification problem where each data point can take more than one labels. Then neural network appears to be a natural choice since it supports multilabel classification easily by making the output layer a vector of length equal to the number of all potential labels. Due to the fact that the class distribution is extremely imbalanced, we decided to only use the labels that occur more than 4000 times out of 65093 articles.

### Baseline
We chose an intuitive model as our baseline model. We transform the multilabel classification problem to multiple binary classification problems. Then we use a support vector machine for each binary classification problem. Given a new article, we infer if it has each label using the corresponding binary classifier.

<p>
  <img src="/images/SVM.png" width="600" height="400" />
  <em> Figure 6</em>
</p>

This model is very simple and intuitive but it cannot exploit label correlation (Zhang, 2018). It provides a baseline performance that is essentially better than random guessing.

### Neural Network
We use a fully connected neural network with 2 hidden layers. The output layer is a vector of length equal to the number of potential labels. The activation function for the output layer is sigmoid function. Note that softmax activation cannot be used here since we are modeling the probability distribution against each potential class instead of a distribution over all potential classes. The choice of loss function is straightforward. We compute the binary cross entropy for each label entry and add them together. We also add L2 regularization in the loss function to constraint the complexity of the neural network.

<p>
  <img src="/images/nn.png" width="600" height="600" />
  <em> Figure 7</em>
</p>


### Evaluation Metrics
Evaluation for a multilabel classifier is more challenging than that for a one-label classifier. Counting exact match is usually not preferred since it is too strict. Therefore the evaluation metrics for multilabel classification should be different from those of one-label classification. We considered four types of evaluation metrics summarized by (Sorower, 2017). They are example-based metrics, label-based macro averaged metrics, and label-based micro averaged metrics. One can compute precision, recall, and F1-measure for each of those types of metrics. Here we use the example-based F1-measure. F1-measure is good at balancing precision and recall. Example based method evaluates the average difference between the predicted labels and the actual labels for each example and it can capture the notion of partially correct (Sorower, 2017).

Let <img src="https://latex.codecogs.com/gif.latex?Z_i" title="Z_i" /> be the predicted labels for the ith article and <img src="https://latex.codecogs.com/gif.latex?Y_i" title="Y_i" /> be the ground truth labels for the ith article. If we consider n data points, then

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?{F_1}^{example}&space;=&space;\dfrac{1}{n}&space;\sum_{i=1}^{n}&space;\dfrac{2|Y_i&space;\cap&space;Z_i|}{|Y_i|&space;&plus;&space;|Z_i|}" title="{F_1}^{example} = \dfrac{1}{n} \sum_{i=1}^{n} \dfrac{2|Y_i \cap Z_i|}{|Y_i| + |Z_i|}" />
</p>

Note that the expected example-based F1-score under the assumption that classes are distributed evenly is 0.5.

### Hyperparameter Tuning
For our neural network, there are three sets of hyperparameters: the choice of activation function in hidden layers, the sizes of hidden layers, and the constraint coefficient for L2 regularization. We search through the hyperparameters space exhaustively and find the hyperparameters with the highest exmpled-based F1-measure using cross validation. The grid search reports that optimal hyperparameters are: tanh as the activation function, the size of the first hidden layers is 300, the size of the second hidden layer is 100, and the regularization coefficient is 1e-4. To visualize the impact of the a single hyperparameter, we present the plot of the F1-score against values of various hyperparameters.

<p>
  <img src="/images/HL1.png" width="600" height="400" />
  <em> Figure 8</em>
</p>

Figure 8 shows the influence of the size of hidden layer 1 on the performance of our neural network. The blue line shows that the trainig F1-score keep improving as the size of hidden layer 1 increases. The reason may be that a larger size of the hidden layer give our neural network more complexity to fit the data. However, the testing F1-scores do not show a notable change. This indicates that the neural network are overfitting when we give it larger hidden layers.

<p>
  <img src="/images/RC.png" width="600" height="400" />
  <em> Figure 9</em>
</p>

Figure 9 shows influence of different coefficients for L2-regularization on the performance of our neural network. It is expected that the traning F1-score will decrease monotonically as the coeffiicient increase since it constraints the complexity of the model. We can see that the optimal performance is reached at around 1e-4. A regularization coefficient larger than it will constraints our model too tightly.

The last but not least thing we want to mention in our hyperparameter tuning section is about PCA. We have the intuition that using different number of components from PCA will have noticeable influence on the performance of our neural network. Therefore we visualize the influence of number of components in PCA in Figure 10. Again, the blue line shows that larger number of principal components, or larger input data dimension, gives higher training score. However, the testing score stop increasing after 500 principal components. Previously, in the unsupervised section, we reported that 500 principal components only explain around 50% of the variance in the original data (Figure 2).

<p>
  <img src="/images/PCA.png" width="600" height="400" />
  <em> Figure 10</em>
</p>

### Result
We compare the performance of the neural network to that of the method of multiple SVMs and the expected performance of random guessing. As mentioned previously, sssumming that the class distribution is balanced, a random guessing classifier would reach a expected F1-score of 0.5. Our data is very imbalanced so we run a simulation to see the expected F1-score of a random guessing classifier.b The simulation of random guessing based on the proportion of each class reports an average F1-score of 0.101.

We fit the previously mentioned group of SVMs model on the test dataset and it reaches a F1-score of 0.67. On the same test dataset, the neural network with the optimal hyperparameter reported by the grid search reaches a F1-score of 0.733. Compared to the result of guessing baseline, the group of SVMs already learns much more pattern in the data. The neural network learns even better (more than 9% percent better) than the group of SVMs. The main reason why the method of group of SVMs does not perform well compared to the neural network may be that it fails to learn the correlation between labels since each SVM is trained independently.

<!--
## Result
We look to see how closely the unsupervised learning clusters matches the ground truth clusters and whether it is prone to outliers. We can compare our clusters to the ground truth clusters to see which ones better match the data. For the supervised learning, we will see the accuracy to which the classifier works on other papers.
-->

<!--
## Discussion
The best outcome would be for the supervised learning classification to have a high accuracy. This would mean that the words in the title and abstract tell you what topic a paper is in. We plan to submit our work to the Kaggle competition as well as seeing if Arxiv or jounals would want to use this for automatic topic selection.
-->

## Reference
1. Griffey, J., Yelton, A., Kim, B., & Boman, C. (2019). Artificial Intelligence and Machine Learning in Libraries. ALA TechSource.
2. Kowsari, K.; Jafari Meimandi, K.; Heidarysafa, M.; Mendu, S.; Barnes, L.; Brown, D. Text Classification Algorithms: A Survey. Information 2019, 10, 150.
3. Anupriya, P. & Karpagavalli, S.. (2015). LDA based topic modeling of journal abstracts. 1-5. 10.1109/ICACCS.2015.7324058. 
4. Nigam, K., McCallum, A. K., Thrun, S., & Mitchell, T. (2000). Text classification from labeled and unlabeled documents using EM. Machine learning, 39(2-3), 103-134.
5. Jones, K. S. (1972). A statistical interpretation of term specificity and its application in retrieval. Journal of documentation.
6. Dasgupta, S., & Ng, V. (2009, August). Topic-wise, sentiment-wise, or otherwise? Identifying the hidden dimension for unsupervised text classification. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing (pp. 580-589).
7. Bird, S., Klein, E., & Loper, E. (2009). Natural language processing with Python: analyzing text with the natural language toolkit. " O'Reilly Media, Inc.".
8. Zhang, M. L., Li, Y. K., Liu, X. Y., & Geng, X. (2018). Binary relevance for multi-label learning: an overview. Frontiers of Computer Science, 12(2), 191-202.
9. Sorower, M. S. (2010). A literature survey on algorithms for multi-label learning. Oregon State University, Corvallis, 18, 1-25.
