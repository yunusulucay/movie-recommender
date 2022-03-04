# Movie Recommender

The main purpose of this project is the implementation of Clustering Algorithms. However, it is aimed to create a series of recommendation systems with clustering and some other techniques. I used elbow method and silhouette score to select cluster number. These are our clustering metrics. In addition to this, clustering was applied using text data. Here, a series recommendation system was created by applying 4 different methods.

<img src="https://user-images.githubusercontent.com/42489236/156833183-cf94bbdd-fb58-41b9-bc28-a4fb35eddc95.png" width="900" height="1500">

1- Clustering:
I implemented 4 different clustering models. These are Agglomerative Clustering, K-Means Clustering, Density-Based Clustering, Gaussian Mixture Model.

2- Clustering with AutoEncoder:
AutoEncoder used when clustering. AutoEncoder basically consists of two structures in the form of encoder and decoder. Input and output are the same. It tries to achieve the same output by bottlenecking the input value. In this way, we get the features that express the input value the most.

3- Clustering with CrossTab:
PCA was used. A user-based clustering is used. A clustering was carried out by arranging the data as watched/not watched(0/1) for each user.

4- Association Rules Mining:
Here, it is aimed to create a prediction model based on the relationship between the movies. For example, there are 2 people. For example, user1 watched {Lord of the Rings, The Silence of the Lambs, Forrest Gump}, user2 watched {Lord of the Rings, The Silence of the Lambs, Jurassic Park} and user3 watched The Lord of the Rings, it would be appropriate to recommend The Silence of the Lambs to him.

![image](https://user-images.githubusercontent.com/42489236/156826723-2f71151f-1327-4152-8655-c2bf1b035de0.png)
Some visualization about data

![image](https://user-images.githubusercontent.com/42489236/156830186-432296fa-c9a0-4300-ac8e-394dcb33de2c.png)

**Comments About who watched Ace Ventura: Pet Detective and Forrest Gump movies(Association Rules Mining)**

%26 of people watched Ace Ventura: Pet Detective movie (antecedent support)

%50 of people watched Forrest Gump (consequent support)

%21 of people watched both of them (support)

%81 of people who watched Ace Ventura also watched Forrest Gump (confidence)

The people who watched them both is %8 more than who watched them separately (leverage)

The rate of the movies related each other is 2.68 (conviction)

# Resources

**Clustering**

- https://neptune.ai/blog/clustering-algorithms

- https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html

**AutoEncoder**

- https://towardsdatascience.com/credit-card-customer-clustering-with-autoencoder-and-k-means-16654d54e64e

- https://medium.com/@iampatricolee18/autoencoder-k-means-clustering-epl-players-by-their-career-statistics-f38e2ea6e375

**Association Rules Mining**

- https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/

- https://medium.com/@mervetorkan/association-rules-with-python-9158974e761a

- https://www.veribilimiokulu.com/python-ile-birliktelik-kurallari-analizi-association-rules-analysis-with-python/

- https://analyticsindiamag.com/guide-to-association-rule-mining-from-scratch/


