# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:55:38 2021

@author: user
"""
###Illustrative example
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
data = pd.read_csv('Data\\kmeans_data1.csv') #Load the data file
#pd.plotting.scatter_matrix(data); #Visualize the data
kmeans = KMeans(n_clusters=5,n_init=1).fit(data) #perform K-means clustering with number of clusters = 2
centroids = kmeans.cluster_centers_ #Extract the cluster centroids
labels = kmeans.labels_ #Extract the labels of clusters
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(data.iloc[labels==2,0],data.iloc[labels==2,1],'y.');  #plot the data with label = 1 (color: b)
plt.plot(data.iloc[labels==3,0],data.iloc[labels==3,1],'r.');  #plot the data with label = 1 (color: b)
plt.plot(data.iloc[labels==4,0],data.iloc[labels==4,1],'k.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid


import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
from sklearn.cluster import AgglomerativeClustering

np.random.seed(8)
n_clusters = 3

data = pd.read_csv('Data\\ecoli.csv') #Load the data file
# pd.plotting.scatter_matrix(data); #Visualize the data
kmeans = KMeans(n_clusters=n_clusters).fit(data.iloc[:,:-1]) #perform K-means clustering with number of clusters = 3
centroids = kmeans.cluster_centers_ #Extract the cluster centroids
labels = kmeans.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(results)
acc = metrics.adjusted_mutual_info_score(label_encoder.fit_transform(data.iloc[:,-1]),labels).ravel()
print(acc)



all_labels = [];
data = pd.read_csv('Data\\ecoli.csv') #Load the data file
for ii in range(50):
    kmeans = KMeans(n_clusters=6,init='random',n_init=1).fit(data.iloc[:,:-1]) #perform K-means clustering with number of clusters = 3
    labels = kmeans.labels_ #Extract the labels of clusters
    all_labels.append(labels)
all_labels = np.array(all_labels).T
cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean').fit(all_labels) 
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(results)
acc =  metrics.adjusted_mutual_info_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(acc)
