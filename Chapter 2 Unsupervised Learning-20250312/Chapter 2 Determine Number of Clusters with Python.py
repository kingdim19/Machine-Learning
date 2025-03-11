# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:12:52 2021

@author: user
"""
######################################################
######################################################
#Silhouette analysis
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X1 = [3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8]
X2 = [5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3]
data_frame = np.array([X1,X2]).T
plt.figure(); plt.scatter(X1,X2)

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []
for num_clusters in range_n_clusters:
 
    # initialise kmeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data_frame)
    cluster_labels = kmeans.labels_
     
    # silhouette score
    silhouette_avg.append(silhouette_score(data_frame, cluster_labels)); 

plt.figure();
plt.plot(range_n_clusters,silhouette_avg,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k')

#Example 1 
from sklearn.metrics import silhouette_score
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
data = pd.read_csv('Data\\kmeans_data1.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data

K = range(2,11); silhouette_avg = []
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    cluster_labels = kmeans.labels_
    silhouette_avg.append(silhouette_score(data, cluster_labels)); 
plt.figure();
plt.plot(K,silhouette_avg,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette score For Optimal k')


#Example 2
from sklearn.metrics import silhouette_score
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
data = pd.read_csv('Data\\kmeans_data2.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data

K = range(2,11); silhouette_avg = []
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    cluster_labels = kmeans.labels_
    silhouette_avg.append(silhouette_score(data, cluster_labels)); 
plt.figure();
plt.plot(K,silhouette_avg,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette score For Optimal k')

#Example 3
from sklearn.metrics import silhouette_score
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
data = pd.read_csv('Data\\iris.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data

K = range(2,11); silhouette_avg = []
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data.iloc[:,:-1])
    cluster_labels = kmeans.labels_
    silhouette_avg.append(silhouette_score(data.iloc[:,:-1], cluster_labels)); 
plt.figure();
plt.plot(K,silhouette_avg,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette score For Optimal k')

######################################################
######################################################
#Elbow Curve Method
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.cluster import KMeans
X1 = [3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8]
X2 = [5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3]
plt.figure(); plt.scatter(X1,X2)


data_frame = np.array([X1,X2]).T
Sum_of_squared_distances = []
K = range(1,11)
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data_frame)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.figure();
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')

#Example 1 
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
data = pd.read_csv('Data\\kmeans_data1.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data

K = range(1,11); Sum_of_squared_distances = []
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.figure();
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')


#Example 2
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
data = pd.read_csv('Data\\kmeans_data2.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data

K = range(1,11); Sum_of_squared_distances = []
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.figure();
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')

#Example 3
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.cluster import KMeans #Import K-means module
data = pd.read_csv('Data\\iris.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data

K = range(1,11); Sum_of_squared_distances = []
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data.iloc[:,:-1])
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.figure();
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')



######################################################
######################################################
####AIC and BIC
#Example 1 
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.mixture import GaussianMixture #Import GMM module
data = pd.read_csv('Data\\kmeans_data1.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
K = range(2,11); aic,bic = [],[]
for num_clusters in K :
    gmm = GaussianMixture(n_components=num_clusters).fit(data) #perform EM clustering with number of clusters = 2
    aic.append(gmm.aic(data))
    bic.append(gmm.bic(data))
    
plt.figure();
plt.plot(K,aic,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('AIC score') 
plt.title('AIC For Optimal k')

plt.figure();
plt.plot(K,bic,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('BIC score') 
plt.title('BIC For Optimal k')

#Example 2
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.mixture import GaussianMixture #Import GMM module
data = pd.read_csv('Data\\kmeans_data2.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
K = range(2,11); aic,bic = [],[]
for num_clusters in K :
    gmm = GaussianMixture(n_components=num_clusters).fit(data) #perform EM clustering with number of clusters = 2
    aic.append(gmm.aic(data))
    bic.append(gmm.bic(data))
    
plt.figure();
plt.plot(K,aic,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('AIC score') 
plt.title('AIC For Optimal k')

plt.figure();
plt.plot(K,bic,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('BIC score') 
plt.title('BIC For Optimal k')


#Example 3
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
from sklearn.mixture import GaussianMixture #Import GMM module
data = pd.read_csv('Data\\iris.csv') #Load the data file
pd.plotting.scatter_matrix(data); #Visualize the data
K = range(2,11); aic,bic = [],[]
for num_clusters in K :
    gmm = GaussianMixture(n_components=num_clusters).fit(data.iloc[:,:-1]) #perform EM clustering with number of clusters = 2
    aic.append(gmm.aic(data.iloc[:,:-1]))
    bic.append(gmm.bic(data.iloc[:,:-1]))
    
plt.figure();
plt.plot(K,aic,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('AIC score') 
plt.title('AIC For Optimal k')

plt.figure();
plt.plot(K,bic,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('BIC score') 
plt.title('BIC For Optimal k')


######################################################
######################################################
#Example 1 - Single link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
import scipy.cluster.hierarchy as shc
data = pd.read_csv('Data\\kmeans_data1.csv') #Load the data file
plt.figure(); #Plot the figure
plt.plot(data.iloc[:,0],data.iloc[:,1],'.');  

##Visualize the data
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
data = pd.read_csv('Data\\kmeans_data1.csv') #Load the data file
plt.figure(); #Plot the figure
plt.plot(data.iloc[:,0],data.iloc[:,1],'.');  

#Example 1 - Single link
plt.figure(); #Plot the figure
dend = shc.dendrogram(shc.linkage(data, method='single'))
plt.title('Single Link')

#Example 1 - Complete link
plt.figure(); #Plot the figure
dend = shc.dendrogram(shc.linkage(data, method='complete'))
plt.title('Complete Link')

#Example 1 - Average link
plt.figure(); #Plot the figure
dend = shc.dendrogram(shc.linkage(data, method='average'))
plt.title('Average Link')


######################################################
######################################################
#Example 2 - Single link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
import scipy.cluster.hierarchy as shc
data = pd.read_csv('Data\\kmeans_data2.csv') #Load the data file
plt.figure(); #Plot the figure
plt.plot(data.iloc[:,0],data.iloc[:,1],'.');  

##Visualize the data
data = pd.read_csv('Data\\kmeans_data2.csv') #Load the data file
plt.figure(); #Plot the figure
plt.plot(data.iloc[:,0],data.iloc[:,1],'.');  

#Example 2 - Single link
plt.figure(); #Plot the figure
dend = shc.dendrogram(shc.linkage(data, method='single'))
plt.title('Single Link')

#Example 2 - Complete link
plt.figure(); #Plot the figure
dend = shc.dendrogram(shc.linkage(data, method='complete'))
plt.title('Complete Link')

#Example 2 - Average link
plt.figure(); #Plot the figure
dend = shc.dendrogram(shc.linkage(data, method='average'))
plt.title('Average Link')

######################################################
######################################################
#Example 3 - Single link
import pandas as pd #Import pandas module
import matplotlib.pyplot as plt #Import the visualization module
import scipy.cluster.hierarchy as shc
data = pd.read_csv('Data\\iris.csv') #Load the data file

#Example 3 - Single link
plt.figure(); #Plot the figure
dend = shc.dendrogram(shc.linkage(data.iloc[:,:-1], method='single'))
plt.title('Single Link')

#Example 3 - Complete link
plt.figure(); #Plot the figure
dend = shc.dendrogram(shc.linkage(data.iloc[:,:-1], method='complete'))
plt.title('Complete Link')

#Example 3 - Average link
plt.figure(); #Plot the figure
dend = shc.dendrogram(shc.linkage(data.iloc[:,:-1], method='average'))
plt.title('Average Link')