# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:36:52 2020

@author: bensonlam
"""

#Chapter 2.4 Machine Learning with Python
###########################################################################
###########################################################################
###########################################################################
#####################################Hierarchical Clustering###############
###########################################################################
###########################################################################
###########################################################################
#Example 1 - Single link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\spiral.csv') #Load the data file
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single').fit(data.iloc[:,:-1]) 
#apply Hierarchical clustering with single linkage
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)

#Example 1 - Complete link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\spiral.csv') #Load the data file
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete').fit(data.iloc[:,:-1]) 
#apply Hierarchical clustering with single linkage
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)

#Example 1 - Average link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\spiral.csv') #Load the data file
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average').fit(data.iloc[:,:-1]) 
#apply Hierarchical clustering with single linkage
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)

###########################################################################
###########################################################################
#Example 2 - Single link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\hierarchical0.csv') #Load the data file
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single').fit(data.iloc[:,:-1]) 
#apply Hierarchical clustering with single linkage
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)

#Example 2 - Complete link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\hierarchical0.csv') #Load the data file
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete').fit(data.iloc[:,:-1]) 
#apply Hierarchical clustering with single linkage
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)

#Example 2 - Average link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\hierarchical0.csv') #Load the data file
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average').fit(data.iloc[:,:-1]) 
#apply Hierarchical clustering with single linkage
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)



###########################################################################
###########################################################################
#Example 3 - Single link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\hierarchical1.csv') #Load the data file
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single').fit(data.iloc[:,:-1]) 
#apply Hierarchical clustering with single linkage
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)

#Example 3 - Complete link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\hierarchical1.csv') #Load the data file
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete').fit(data.iloc[:,:-1]) 
#apply Hierarchical clustering with single linkage
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)

#Example 3 - Average link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix #Import confusion matrix module
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data\\hierarchical1.csv') #Load the data file
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average').fit(data.iloc[:,:-1]) 
#apply Hierarchical clustering with single linkage
labels = cluster.labels_ #Extract the labels of clusters
label_encoder = LabelEncoder()
print(confusion_matrix(label_encoder .fit_transform(data.iloc[:,-1]),labels))
plt.figure(); #Plot the figure
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  #plot the data with label = 0 (color: g)
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.');  #plot the data with label = 1 (color: b)
