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

n_clusters = 3

data = pd.read_csv('abalone.csv') #Load the data file

###Combine the labels
labels = data.iloc[:,-1]
ind = (labels>=1) & (labels<=7); data.loc[ind,-1] = 1
ind = (labels>=8) & (labels<=11); data.loc[ind,-1] = 2
ind = (labels>=12) & (labels<=29); data.loc[ind,-1] = 3

X = pd.get_dummies(data.iloc[:,:-1],drop_first=True)
#Apply K-means to dataset
kmeans = KMeans(n_clusters=n_clusters).fit(X) #perform K-means clustering with number of clusters = 3
centroids = kmeans.cluster_centers_ #Extract the cluster centroids
labels = kmeans.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(results)
acc = metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(acc)


###Apply hierarchical clustering to the dataset
clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(X) #perform K-means clustering with number of clusters = 3
labels = clusters.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(results)
acc = metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(acc)

###Apply GMM to the dataset
gmm = GaussianMixture(n_components=n_clusters).fit(X) 
labels = gmm.predict(X)
label_encoder = LabelEncoder()
results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(results)
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


#####Visuzliation
from sklearn.manifold import TSNE    
X = pd.get_dummies(data.iloc[:,:-1],drop_first=True)
X_embedded = TSNE(n_components=2,init='random', perplexity=3).fit_transform(X)

markers = ['b.','g.','r.','c.','m.','y.','k.','bv','gv','rv','cv','mv','yv','kv','b1','g1','r1','c1','m1','y1','k1','b2','g2','r2','c2','m2','y2','k2','b3','g3','r3','c3','m3','y3','k3']
labels = data.iloc[:,-1]
ulabels = labels.unique()
plt.figure(); 
for ii in range(ulabels.shape[0]):
    label = ulabels[ii]; 
    plt.plot(X_embedded[labels==label][:,0],X_embedded[labels==label][:,1],markers[ii]);  


#####Visuzliation
markers = ['b.','g.','r.','c.','m.','y.','k.','bv','gv','rv','cv','mv','yv','kv','b1','g1','r1','c1','m1','y1','k1','b2','g2','r2','c2','m2','y2','k2','b3','g3','r3','c3','m3','y3','k3']
labels = data.iloc[:,-1]
ulabels = labels.unique()
d1,d2 = 1,2 ###The dimensions of the data
plt.figure(); #Plot the figure with label = 1
for ii in range(ulabels.shape[0]):
    label = ulabels[ii]; 
    plt.plot(data[labels==label].iloc[:,d1],data[labels==label].iloc[:,d2],markers[ii]);  
    
markers = ['b.','g.','r.','c.','m.','y.','k.','bv','gv','rv','cv','mv','yv','kv','b1','g1','r1','c1','m1','y1','k1','b2','g2','r2','c2','m2','y2','k2','b3','g3','r3','c3','m3','y3','k3']
labels = data.iloc[:,-1]
ulabels = labels.unique()
d1,d2 = 4,6 ###The dimensions of the data
plt.figure(); #Plot the figure with label = 1
for ii in range(ulabels.shape[0]):
    label = ulabels[ii]; 
    plt.plot(data[labels==label].iloc[:,d1],data[labels==label].iloc[:,d2],markers[ii]);  
        