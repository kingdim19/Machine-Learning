# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:42:14 2023

@author: user
"""

#################################
#################################
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
import numpy as np
n_clusters = 15

data = pd.read_csv('abalone.csv') #Load the data file
pd.plotting.scatter_matrix(data); 


markers = ['b.','g.','r.','c.','m.','y.','k.','bv','gv','rv','cv','mv','yv','kv','b1','g1','r1','c1','m1','y1','k1','b2','g2','r2','c2','m2','y2','k2','b3','g3','r3','c3','m3','y3','k3']
labels = data.iloc[:,-1]
ulabels = labels.unique()
d1,d2 = 1,2 ###The dimensions of the data
plt.figure(); 
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
    
    
#################################
#################################
#Other Visualization Tools t-SNE
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


#Other Visualization Tools LLE
from sklearn.manifold import LocallyLinearEmbedding as LLE  
X = pd.get_dummies(data.iloc[:,:-1],drop_first=True)
X_embedded = LLE(n_components=2).fit_transform(X)

markers = ['b.','g.','r.','c.','m.','y.','k.','bv','gv','rv','cv','mv','yv','kv','b1','g1','r1','c1','m1','y1','k1','b2','g2','r2','c2','m2','y2','k2','b3','g3','r3','c3','m3','y3','k3']
labels = data.iloc[:,-1]
ulabels = labels.unique()
plt.figure(); 
for ii in range(ulabels.shape[0]):
    label = ulabels[ii]; 
    plt.plot(X_embedded[labels==label][:,0],X_embedded[labels==label][:,1],markers[ii]);  


