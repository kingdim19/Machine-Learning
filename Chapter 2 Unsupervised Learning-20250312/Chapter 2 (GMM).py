# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:36:52 2020

@author: bensonlam
"""

#Chapter 2.4 Machine Learning with Python
###########################################################################
###########################################################################
###########################################################################
#####################################GMM###############################
###########################################################################
###########################################################################
###########################################################################
#Example 1
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.mixture import GaussianMixture #Import GMM module
data = pd.read_csv('Data\\kmeans_data1.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
gmm = GaussianMixture(n_components=2).fit(data) #perform EM clustering with number of clusters = 2
centroids = gmm.means_#Extract the cluster centroids
labels = gmm.predict(data)
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid


#Example 2
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.mixture import GaussianMixture #Import GMM module
data = pd.read_csv('Data\\kmeans_data2.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
gmm = GaussianMixture(n_components=2).fit(data) #perform EM clustering with number of clusters = 2
centroids = gmm.means_#Extract the cluster centroids
labels = gmm.predict(data)
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid

#Example 3
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.mixture import GaussianMixture #Import GMM module
data = pd.read_csv('Data\\kmeans_data3.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
gmm = GaussianMixture(n_components=2).fit(data) #perform EM clustering with number of clusters = 2
centroids = gmm.means_#Extract the cluster centroids
labels = gmm.predict(data)
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid



#Example Heterogeneous Data
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.mixture import GaussianMixture #Import GMM module
data = pd.read_csv('Data\\heterogeneous_data.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
gmm = GaussianMixture(n_components=2).fit(data.iloc[:,:-1]) #perform EM clustering with number of clusters = 2
centroids = gmm.means_#Extract the cluster centroids
labels = gmm.predict(data.iloc[:,:-1])
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid



####################################################################
####################################################################
#Example 1- Compare
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.mixture import GaussianMixture #Import GMM module
data = pd.read_csv('Data\\kmeans_data1 - with label.csv') #Load the data file
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
pd.plotting.scatter_matrix(data); #Visualize the data
gmm = GaussianMixture(n_components=2).fit(data.iloc[:,:-1]) #perform EM clustering with number of clusters = 2
centroids = gmm.means_#Extract the cluster centroids
labels = gmm.predict(data.iloc[:,:-1])
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid

import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\kmeans_data1 - with label.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
kmeans = KMeans(n_clusters=2).fit(data.iloc[:,:-1]) #perform K-means clustering with number of clusters = 3
centroids = kmeans.cluster_centers_ #Extract the cluster centroids
labels = kmeans.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid




####################################################################
####################################################################
#Example 2- Compare
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.mixture import GaussianMixture #Import GMM module
data = pd.read_csv('Data\\kmeans_data2 - with label.csv') #Load the data file
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
pd.plotting.scatter_matrix(data); #Visualize the data
gmm = GaussianMixture(n_components=2).fit(data.iloc[:,:-1]) #perform EM clustering with number of clusters = 2
centroids = gmm.means_#Extract the cluster centroids
labels = gmm.predict(data.iloc[:,:-1])
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid

import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\kmeans_data2 - with label.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
kmeans = KMeans(n_clusters=2).fit(data.iloc[:,:-1]) #perform K-means clustering with number of clusters = 3
centroids = kmeans.cluster_centers_ #Extract the cluster centroids
labels = kmeans.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid


####################################################################
####################################################################
#Example Heterogeneous Data
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.mixture import GaussianMixture #Import GMM module
data = pd.read_csv('Data\\heterogeneous_data.csv') #Load the data file
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
pd.plotting.scatter_matrix(data); #Visualize the data
gmm = GaussianMixture(n_components=2).fit(data.iloc[:,:-1]) #perform EM clustering with number of clusters = 2
centroids = gmm.means_#Extract the cluster centroids
labels = gmm.predict(data.iloc[:,:-1])
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid

import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\heterogeneous_data.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
kmeans = KMeans(n_clusters=2).fit(data.iloc[:,:-1]) #perform K-means clustering with number of clusters = 3
centroids = kmeans.cluster_centers_ #Extract the cluster centroids
labels = kmeans.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
plt.plot(centroids[:,0],centroids[:,1],'ro') #plot the cluster centroid