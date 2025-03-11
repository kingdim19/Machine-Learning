# %%
# # Pairs Trading- finding pairs based on Clustering
# 
# In this case study, we will use clustering methods to select pairs for a
# pairs trading strategy.

# ## Content

# * [1. Problem Definition]
# * [2. Getting Started - Load Libraries and Dataset]
#     * [2.1. Load Libraries]
#     * [2.2. Load Dataset]
# * [3. Exploratory Data Analysis]
#     * [3.1 Descriptive Statistics]
#     * [3.2. Data Visualisation]
# * [4. Data Preparation]
#     * [4.1 Data Cleaning]
#     * [4.2.Data Transformation] 
# * [5. Evaluate Algorithms and Models]      
#     * [5.1. k-Means Clustering]
#         * [5.1.1. Finding right number of clusters]
#         * [5.1.2. Clustering and Visualisation]
#     * [5.2. Hierarchical Clustering (Agglomerative Clustering)]
#         * [5.2.1. Building Hierarchy Graph/ Dendogram]
#         * [5.2.2. Clustering and Visualisation]
#     * [5.3. Affinity Propagation Clustering]
#         * [5.3.1. Visualising the cluster]
#     * [5.4. Cluster Evaluation]
# * [6. Pair Selection]
#     * [6.1. Cointegration and Pair Selection Function]
#     * [6.2. Pair Visualisation]

# # 1. Problem Definition

# Our goal in this case study is to perform clustering analysis on the stocks
# of S&P500 and come up with pairs for a pairs trading strategy.
# 
# The data of the stocks of S&P 500, are obtained using pandas_datareader from
# yahoo finance. It includes price data from 2018 onwards.

# # 2. Getting Started- Loading the data and python packages

# %%
# ## 2.1. Loading the python packages

# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import datetime
import pandas_datareader as dr

# Import Model Packages 
from sklearn.cluster import KMeans, AgglomerativeClustering,AffinityPropagation, DBSCAN
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import cluster, covariance, manifold

# Other Helper Packages and functions
import matplotlib.ticker as ticker
from itertools import cycle

# For printing tables
def pretty_print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt='.3f'))

# %%
# ## 2.2. Loading the Data

# The data already obtained from yahoo finance is imported.
dataset = read_csv('SP500Data.csv',index_col=0)

# Disable the warnings
import warnings
warnings.filterwarnings('ignore')

# %%
# # 3. Exploratory Data Analysis

# ## 3.1. Descriptive Statistics

# Print the dataset shape
print(f'Dataset shape: {dataset.shape}')

# %%

# peek at data
# set_option('display.width', 100)
pretty_print_df(dataset.head(5).iloc[:,:8])

# %%

# describe data
# set_option('precision', 3)
pretty_print_df(dataset.describe().iloc[:,:8])

# %%
# ## 3.2. Data Visualisation

# We will take a detailed look into the visualisation post clustering.

# %%
# ## 4. Data Preparation

# ## 4.1. Data Cleaning
# We check for the NAs in the rows, either drop them or fill them with the last
# value of the column.

# Checking for any null values and removing the null values'''
print('Null Values =',dataset.isnull().values.any())

# %%
# Getting rid of the columns with more than 30% missing values. 

missing_fractions = dataset.isnull().mean().sort_values(ascending=False)

print(missing_fractions.head(10))

# %%
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

dataset.drop(labels=drop_list, axis=1, inplace=True)
print(f'Dataset shape: {dataset.shape}')

# %%
# Given that there are null values drop the row containing the null values.

# Fill the missing values with the last value available in the dataset. 
dataset=dataset.fillna(method='ffill')
pretty_print_df(dataset.head(2).iloc[:,:8])

# %%
# ## 4.2. Data Transformation

# For the purpose of clustering, we will be using annual
# returns and variance as the variables as they are the indicators of the stock
# performance and its volatility. Let us prepare the return and volatility
# variables from the data.

# Calculate average annual percentage return and volatilities over a
# theoretical one year period
returns = dataset.pct_change().mean() * 252
returns = pd.DataFrame(returns)
returns.columns = ['Returns']
returns['Volatility'] = dataset.pct_change().std() * np.sqrt(252)
data = returns

# All the variables should be on the same scale before applying clustering,
# otherwise a feature with large values will dominate the result. We use
# StandardScaler in sklearn to standardize the dataset’s features onto unit
# scale (mean = 0 and variance = 1).

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(data)
rescaledDataset = pd.DataFrame(scaler.fit_transform(data),
    columns = data.columns, index = data.index)

# summarize transformed data
X = rescaledDataset
pretty_print_df(X.head(2))

# The parameters to clusters are the indices and the variables used in the
# clustering are the columns. Hence the data is in the right format to be fed
# to the clustering algorithms.

# %%
# # 5. Evaluate Algorithms and Models

# We will look at the following models:
# 
# 1. KMeans
# 2. Hierarchical Clustering (Agglomerative Clustering)
# 3. Affinity Propagation 

# ## 5.1. K-Means Clustering

# ### 5.1.1. Finding optimal number of clusters

# In this step we look at the following metrices:
# 
# 1. Sum of square errors (SSE) within clusters
# 2. Silhouette score.

distorsions = []
max_loop=20
for k in range(2, max_loop):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, max_loop), distorsions)
plt.xticks([i for i in range(2, max_loop)], rotation=75)
plt.grid(True)
plt.show()

# Inspecting the sum of squared errors chart, it appears the elbow “kink”
# occurs 5 or 6 clusters for this data. Certainly, we can see that as the
# number of clusters increase pass 6, the sum of square of errors within
# clusters plateaus off.

# %%
# #### Silhouette score

from sklearn import metrics

silhouette_score = []
for k in range(2, max_loop):
        kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
        kmeans.fit(X)        
        silhouette_score.append(metrics.silhouette_score(X, kmeans.labels_, random_state=10))
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, max_loop), silhouette_score)
plt.xticks([i for i in range(2, max_loop)], rotation=75)
plt.grid(True)
plt.show()

# From the silhouette score chart, we can see that there are various parts of
# the graph where a kink can be seen. Since there is not much a difference in
# SSE after 6 clusters, we would prefer 6 clusters in the K-means model.

# %%
# ### 5.1.2.  Clustering and Visualisation

# Let us build the k-means model with six clusters and
# visualize the results.

nclust=6

# Fit with k-means
k_means = cluster.KMeans(n_clusters=nclust, random_state=1)
k_means.fit(X)

# Extracting labels 
target_labels = k_means.predict(X)

# Visualizing how your clusters are formed is no easy task when the number of
# variables/dimensions in your dataset is very large. One of the methods of
# visualising a cluster in two-dimensional space.

centroids = k_means.cluster_centers_
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c = k_means.labels_, 
    cmap ="rainbow", label = X.index)
ax.set_title('k-Means results')
ax.set_xlabel('Mean Return')
ax.set_ylabel('Volatility')
plt.colorbar(scatter)

plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=11)
plt.show()

# %%
# Let us check the elements of the clusters

# show number of stocks in each cluster
clustered_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
# clustered stock with its cluster label
clustered_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
clustered_series = clustered_series[clustered_series != -1]

plt.figure(figsize=(12,7))
plt.barh(
    range(len(clustered_series.value_counts())), # cluster labels, y axis
    clustered_series.value_counts()
)
plt.title('Cluster Member Counts')
plt.xlabel('Stocks in Cluster')
plt.ylabel('Cluster Number')
plt.show()

# The number of stocks in a cluster range from around 40 to 120. Although, the
# distribution is not equal, we have significant number of stocks in each
# cluster.

# %%
# ## 5.2. Hierarchical Clustering (Agglomerative Clustering)

# In the first step we look at the hierarchy graph and check for the number of
# clusters.

# ### 5.2.1. Building Hierarchy Graph/ Dendogram

# The hierarchy class has a dendrogram method which takes the value returned by
# the linkage method of the same class. The linkage method takes the dataset
# and the method to minimize distances as parameters. We use 'ward' as the
# method since it minimizes then variants of distances between the clusters.

from scipy.cluster.hierarchy import dendrogram, linkage, ward

# Calulate linkage
Z= linkage(X, method='ward')
print(Z[0])

# %%
# The best way to visualize an agglomerate clustering algorithm is through a
# dendogram, which displays a cluster tree, the leaves being the individual
# stocks and the root being the final single cluster. The "distance" between
# each cluster is shown on the y-axis, and thus the longer the branches are,
# the less correlated two clusters are.

# Plot Dendogram
plt.figure(figsize=(10, 7))
plt.title("Stocks Dendrograms")
dendrogram(Z,labels = X.index)
plt.show()

# %%
# Once one big cluster is formed, the longest vertical distance without any
# horizontal line passing through it is selected and a horizontal line is drawn
# through it. The number of vertical lines this newly created horizontal line
# passes is equal to number of clusters.
# 
# Then we select the distance threshold to cut the dendrogram to obtain the
# selected clustering level. The output is the cluster labelled for each row of
# data. As expected from the dendrogram, a cut at 13 gives us four clusters.

distance_threshold = 13
clusters = fcluster(Z, distance_threshold, criterion='distance')
chosen_clusters = pd.DataFrame(data=clusters, columns=['cluster'])
print(chosen_clusters['cluster'].unique())

# %%
# ### 5.2.2.  Clustering and Visualisation

nclust = 4
hc = AgglomerativeClustering(n_clusters=nclust, affinity = 'euclidean', linkage = 'ward')
clust_labels1 = hc.fit_predict(X)

fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c =clust_labels1, cmap ="rainbow")
ax.set_title('Hierarchical Clustering')
ax.set_xlabel('Mean Return')
ax.set_ylabel('Volatility')
plt.colorbar(scatter)
plt.show()

# Similar to the plot of k-means clustering, we see that there are some
# distinct clusters separated by different colors. 

# %%
# ## 5.3. Affinity Propagation Clustering

ap = AffinityPropagation()
ap.fit(X)
clust_labels2 = ap.predict(X)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:,0],X.iloc[:,1], c =clust_labels2, cmap ="rainbow")
ax.set_title('Affinity')
ax.set_xlabel('Mean Return')
ax.set_ylabel('Volatility')
plt.colorbar(scatter)
plt.show()

# Similar to the plot of k-means clustering, we see that there are some
# distinct clusters separated by different colors. 

# %%
# ### 5.3.1 Cluster Visualisation

cluster_centers_indices = ap.cluster_centers_indices_
labels = ap.labels_

no_clusters = len(cluster_centers_indices)
print('Estimated number of clusters: %d' % no_clusters)

# %%
# Plot exemplars
X_temp=np.asarray(X)

fig = plt.figure(figsize=(8,6))
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(no_clusters), colors):
    class_members = labels == k
    cluster_center = X_temp[cluster_centers_indices[k]]
    plt.plot(X_temp[class_members, 0], X_temp[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, 
             markeredgecolor='k', markersize=14)
    for x in X_temp[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.show()

# %%
# show number of stocks in each cluster
clustered_series_ap = pd.Series(index=X.index, data=ap.labels_.flatten())

# clustered stock with its cluster label
clustered_series_all_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
clustered_series_ap = clustered_series_ap[clustered_series != -1]

plt.figure(figsize=(12,7))
plt.barh(
    range(len(clustered_series_ap.value_counts())), # cluster labels, y axis
    clustered_series_ap.value_counts()
)
plt.title('Cluster Member Counts')
plt.xlabel('Stocks in Cluster')
plt.ylabel('Cluster Number')
plt.show()

# %%
# ## 5.4. Cluster Evaluation

# If the ground truth labels are not known, evaluation must be performed using
# the model itself. The Silhouette Coefficient (sklearn.metrics.silhouette_score)
# is an example of such an evaluation, where a higher Silhouette Coefficient
# score relates to a model with better defined clusters. The Silhouette
# Coefficient is defined for each sample and is composed of two scores:

from sklearn import metrics
print("km", metrics.silhouette_score(X, k_means.labels_, metric='euclidean'))
print("hc", metrics.silhouette_score(X, hc.fit_predict(X), metric='euclidean'))
print("ap", metrics.silhouette_score(X, ap.labels_, metric='euclidean'))

# We go ahead with the affinity propagation and use 27 clusters as specified by
# this clustering method

# %%
# ### Visualising the return within a cluster

# The understand the intuition behind clustering, let us visualize the results
# of the clusters.

# all stock with its cluster label (including -1)
clustered_series = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
# clustered stock with its cluster label
clustered_series_all = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
clustered_series = clustered_series[clustered_series != -1]

# get the number of stocks in each cluster
counts = clustered_series_ap.value_counts()

# let's visualize some clusters
cluster_vis_list = list(counts[(counts<25) & (counts>1)].index)[::-1]
print(cluster_vis_list)

# %%
CLUSTER_SIZE_LIMIT = 9999
counts = clustered_series.value_counts()
ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]
print ("Clusters formed: %d" % len(ticker_count_reduced))
print ("Pairs to evaluate: %d" % (ticker_count_reduced*(ticker_count_reduced-1)).sum())

# %%
# plot a handful of the smallest clusters
print(cluster_vis_list[0:min(len(cluster_vis_list), 4)])

# %%
for clust in cluster_vis_list[0:min(len(cluster_vis_list), 4)]:
    tickers = list(clustered_series[clustered_series==clust].index)
    means = np.log(dataset.loc[:"2018-02-01", tickers].mean())
    data = np.log(dataset.loc[:"2018-02-01", tickers]).sub(means)
    data.plot(title='Stock Time Series for Cluster %d' % clust)
plt.show()

# Looking at the charts above, across all the clusters with small number of
# stocks, we see similar movement of the stocks under different clusters, which
# corroborates the effectiveness of the clustering technique.

# %%
# # 6. Pairs Selection

# ## 6.1. Cointegration and Pair Selection Function

def find_cointegrated_pairs(data, significance=0.05):
    # This function is from https://www.quantopian.com/lectures/introduction-to-pairs-trading
    n = data.shape[1]    
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(1):
        for j in range(i+1, n):
            S1, S2 = data[keys[i]], data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

from statsmodels.tsa.stattools import coint
cluster_dict = {}
for i, which_clust in enumerate(ticker_count_reduced.index):
    tickers = clustered_series[clustered_series == which_clust].index   
    score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(
        dataset[tickers]
    )
    cluster_dict[which_clust] = {}
    cluster_dict[which_clust]['score_matrix'] = score_matrix
    cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
    cluster_dict[which_clust]['pairs'] = pairs

pairs = []
for clust in cluster_dict.keys():
    pairs.extend(cluster_dict[clust]['pairs'])

print ("Number of pairs found : %d" % len(pairs))
print ("In those pairs, there are %d unique tickers." % len(np.unique(pairs)))

print(pairs)

# %%
# ## 6.2. Pair Visualisation

from sklearn.manifold import TSNE
import matplotlib.cm as cm
stocks = np.unique(pairs)
X_df = pd.DataFrame(index=X.index, data=X).T

in_pairs_series = clustered_series.loc[stocks]
stocks = list(np.unique(pairs))
X_pairs = X_df.T.loc[stocks]

X_tsne = TSNE(learning_rate=50, perplexity=3, random_state=1337).fit_transform(X_pairs)

plt.figure(facecolor='white',figsize=(16,8))
plt.clf()
plt.axis('off')
for pair in pairs:
    #print(pair[0])
    ticker1 = pair[0]
    loc1 = X_pairs.index.get_loc(pair[0])
    x1, y1 = X_tsne[loc1, :]
    #print(ticker1, loc1)

    ticker2 = pair[0]
    loc2 = X_pairs.index.get_loc(pair[1])
    x2, y2 = X_tsne[loc2, :]
      
    plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray')
    
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha=0.9,
    c=in_pairs_series.values, cmap=cm.Paired)
plt.title('T-SNE Visualization of Validated Pairs')

# zip joins x and y coordinates in pairs
for x,y,name in zip(X_tsne[:,0],X_tsne[:,1],X_pairs.index):
    label = name
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=11)
plt.show()

# %%
# **Conclusion**
# 
# The clustering techniques do not directly help in stock trend prediction.
# However, they can be effectively used in portfolio construction for finding
# the right pairs, which eventually help in risk mitigation and one can achieve
# superior risk adjusted returns.
# 
# We showed the approaches to finding the appropriate number of clusters in
# k-means and built a hierarchy graph in hierarchical clustering. A next step
# from this case study would be to explore and backtest various long/short
# trading strategies with pairs of stocks from the groupings of stocks.
# 
# Clustering can effectively be used for dividing stocks into groups with
# “similar characteristics” for many other kinds of trading strategies and can
# help in portfolio construction to ensure we choose a universe of stocks with
# sufficient diversification between them.
