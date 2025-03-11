# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:56:28 2020

@author: bensonlam
"""

#Chapter 2.4 Machine Learning with Python
###########################################################################
###########################################################################
###########################################################################
#####################################DBScan###############################
###########################################################################
###########################################################################
###########################################################################
#Example 3
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import DBSCAN #Import DBScan module
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\iris.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
cluster = DBSCAN (eps=0.5,min_samples=4).fit(data.iloc[:,:-1]) #DBScan
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)

#Example 4
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import DBSCAN #Import DBScan module
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
from numpy import savetxt
data = pd.read_csv('Data\\ecoli.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
cluster = DBSCAN (eps=0.3,min_samples=1).fit(data.iloc[:,:-1]) #DBScan
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
matrix = confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels)
print(matrix)
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
savetxt('confusion_matrix.csv',matrix,delimiter=',')






###########################################################################
###########################################################################
###########################################################################
#####################################Optics###############################
###########################################################################
###########################################################################
###########################################################################
#Example 1
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import OPTICS, cluster_optics_dbscan #Import Opticsmodule
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\iris.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
clust = OPTICS(min_samples=2).fit(data.iloc[:,:-1])
plt.figure(); plt.plot(clust.reachability_); plt.title('reachability distance')
labels = cluster_optics_dbscan(reachability=clust.reachability_,core_distances=clust.core_distances_,ordering=clust.ordering_,eps=0.5)
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)

#Example 4
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import OPTICS, cluster_optics_dbscan #Import Opticsmodule
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\ecoli.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
clust = OPTICS(min_samples=2).fit(data.iloc[:,:-1])
plt.figure(); plt.plot(clust.reachability_); plt.title('reachability distance')
labels = cluster_optics_dbscan(reachability=clust.reachability_,core_distances=clust.core_distances_,ordering=clust.ordering_,eps=0.5)
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
