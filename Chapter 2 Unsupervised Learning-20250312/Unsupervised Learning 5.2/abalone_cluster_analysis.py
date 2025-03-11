# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:18:56 2023

@author: user
"""

#################################
#################################
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import completeness_score
from sklearn import metrics
import numpy as np
from sklearn.mixture import GaussianMixture #Import GMM module
from sklearn.cluster import AgglomerativeClustering

n_clusters = 15

data = pd.read_csv('abalone.csv') #Load the data file
X = pd.get_dummies(data.iloc[:,:-1],drop_first=True)
#Apply K-means to dataset
kmeans = KMeans(n_clusters=n_clusters).fit(X) #perform K-means clustering with number of clusters = 3
centroids = kmeans.cluster_centers_ #Extract the cluster centroids
labels = kmeans.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
# results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
# print(results)
acc = metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(acc)


###Apply hierarchical clustering to the dataset
clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(X) #perform K-means clustering with number of clusters = 3
labels = clusters.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
# print(results)
acc = metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(acc)

###Apply GMM to the dataset
gmm = GaussianMixture(n_components=n_clusters).fit(X) 
labels = gmm.predict(X)
label_encoder = LabelEncoder()
# results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
# print(results)
acc = metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(acc)


#Apply Ensemble clustering to Ecoli dataset
no_samples = data.shape[0]
no_estimators = 50
#Declare the weight of each vote
vote = 1/no_estimators
#co_association matrix is no_estimators X no_estimators (no_estimators patterns)
co_association = np.zeros((no_samples, no_samples))

#for each of your estimators
for est in range(no_estimators):
    #fit the data and grab the labels
    kmeans = KMeans(n_clusters=30,init='random',n_init=1).fit(X)
    labels = kmeans.labels_
    #find all associations and transform it into a numpy array
    res = [[int(i == j) for i in labels] for j in labels]
    res = np.array(res)
    #Vote and update the co_association matriz
    res = res * vote
    co_association = co_association + res
distance_matrix = 1-co_association
cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',compute_distances=True).fit(distance_matrix) 
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
# results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
# print(results)
acc =  metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(acc)


