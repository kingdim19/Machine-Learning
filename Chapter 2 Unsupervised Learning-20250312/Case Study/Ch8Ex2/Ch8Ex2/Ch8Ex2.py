# %% [markdown]
# 
# # Clustering individuals as per their demographics
# 
# In this case study, we will use clustering methods to identify different
# types of investors.

# %% [markdown]
# ## Content

# %% [markdown]
# * [1. Problem Definition]
# * [2. Getting Started - Load Libraries and Dataset]
#     * [2.1. Load Libraries]
#     * [2.2. Load Dataset]
# * [3. Exploratory Data Analysis]
#     * [3.1. Descriptive Statistics]  
#     * [3.2. Data Visualisation]
# * [4. Data Preparation]
#     * [4.1. Data Cleaning]
#     * [4.2. Data Transformation]
# * [5. Evaluate Algorithms and Models]
#     * [5.1. k-Means Clustering]
#         * [5.1.1. Finding right number of clusters]
#         * [5.1.2. Clustering and Visualisation]
#     * [5.2. Affinity Propagation Clustering]
#     * [5.3. Cluster Evaluation]
# * [6. Cluster Intuition]
#     

# %% [markdown]
# # 1. Problem Definition

# %% [markdown]
# The goal of this case study is to build a machine learning model to cluster
# individuals/investors based on the parameters related to the ability and
# willingness to take risk. We will focus on using common demographic and
# financial characteristics to accomplish this.
# 
# For this case study the data used is from survey of Consumer Finances which
# is conducted by the Federal Reserve Board.

# %% [markdown]
# # 2. Getting Started- Loading the data and python packages

# %% [markdown]
# ## 2.1. Loading the python packages

# %%
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import datetime
from tabulate import tabulate

#Import Model Packages 
from sklearn.cluster import KMeans, AgglomerativeClustering, \
    AffinityPropagation
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import cluster, covariance, manifold

#Other Helper Packages and functions
import matplotlib.ticker as ticker
from itertools import cycle

# For printing tables
def pretty_print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt='.3f'))

# %% [markdown]
# ## 2.2. Loading the Data

# %% [markdown]
# The dataset is same as the dataset used, which is further processed to give
# the following attributes for different investors for the year 2007.

# %%
# load dataset
dataset = pd.read_excel('ProcessedData.xlsx')

# %%
# Disable the warnings
import warnings
warnings.filterwarnings('ignore')


# %% [markdown]
# # 3. Exploratory Data Analysis

# %% [markdown]
# ## 3.1. Descriptive Statistics

# %%
# shape
print(f'Dataset shape: {dataset.shape}')

# %%
# peek at data
# set_option('display.width', 100)
pretty_print_df(dataset.head(5))

# %% [markdown]
# As we can see in the table above, there are 12 attributes for each of the
# individuals. These attributes can be categorized as demographic, financial
# and behavioral attributes. 

# %%
# describe data
pretty_print_df(dataset.describe())

# %% [markdown]
# ## 3.2. Data Visualization

# %% [markdown]
# Let us look at the correlation. We will take a detailed look into the
# visualization post clustering.

# %%
# correlation
correlation = dataset.corr()
plt.figure(figsize=(15,15))
plt.title('Correlation Matrix')
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.show()

# %% [markdown]
# ## 4. Data Preparation

# %% [markdown]
# ## 4.1. Data Cleaning
# Let us check for the NAs in the rows, either drop them or fill them with the
# mean of the column.

# %%
# Checking for any null values and removing the null values
print('Null Values =',dataset.isnull().values.any())

# %% [markdown]
# Given there isn't any missing data and the data is already in the categorical
# format no further data cleaning was performed. The ID column which is not
# needed is dropped.

# %%
X=dataset.copy("deep")
X=X.drop(['ID'], axis=1)
pretty_print_df(X.head())

# %% [markdown]
# ## 4.2. Data Transformation

# %% [markdown]
# The data available is comprised of attributes with similar scale. Hence no
# data transformation is required. 

# %% [markdown]
# # 5. Evaluate Algorithms and Models

# %% [markdown]
# In this step, we will look at the following models and perform further
# analysis and visualization.
# 
# 1. K-means
# 2. Affinity Propagation

# %% [markdown]
# ## 5.1. K-Means Clustering
# 

# %% [markdown]
# ### 5.1.1. Finding optimal number of clusters

# %% [markdown]
# In this step we look at the following metrices to get the optimum number of
# clusters. Typically, two metrics are used to evaluate the number of clusters
# in K-means model.
# 
# 1. Sum of square errors (SSE) within clusters
# 2. Silhouette score
# 

# %%
distorsions = []
max_loop=40
for k in range(2, max_loop):
    k_means = KMeans(n_clusters=k, random_state=10, n_init=10)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)
fig = plt.figure(figsize=(10, 5))
plt.plot(range(2, max_loop), distorsions)
plt.xticks([i for i in range(2, max_loop)], rotation=75)
plt.xlabel("Number of clusters")
plt.ylabel("Sum of Square Error")
plt.grid(True)
plt.show()

# %% [markdown]
# #### Silhouette score

# %%
from sklearn import metrics

silhouette_score = []
for k in range(2, max_loop):
    kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
    kmeans.fit(X)        
    silhouette_score.append(
        metrics.silhouette_score(X, kmeans.labels_, random_state=10)
    )
fig = plt.figure(figsize=(10, 5))
plt.plot(range(2, max_loop), silhouette_score)
plt.xticks([i for i in range(2, max_loop)], rotation=75)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.grid(True)
plt.show()

# %% [markdown]
# Looking at both the charts above, the optimum number of clusters seem to be
# around 7. We can see that as the number of clusters increase pass 6, the sum
# of square of errors within clusters plateaus off. From the second graph, we
# can see that there are various parts of the graph where a kink can be seen.
# Since there is not much a difference in SSE after 7 clusters, we would use 7
# clusters in the k-means model below.

# %% [markdown]
# ### 5.1.2. Clustering and Visualisation

# %%
nclust=7

# %%
# Fit with k-means
k_means = cluster.KMeans(n_clusters=nclust, random_state=10, n_init=10)
k_means.fit(X)

# %%
# Extracting labels 
target_labels = k_means.predict(X)

# %% [markdown]
# ## 5.2. Affinity Propagation

# %%
ap = AffinityPropagation(
    damping = 0.5, max_iter = 250, affinity = 'euclidean', random_state=10,
)
ap.fit(X)
clust_labels2 = ap.predict(X)

# %%
cluster_centers_indices = ap.cluster_centers_indices_
labels = ap.labels_
n_clusters_ = len(cluster_centers_indices)
print('Estimated number of clusters: %d' % n_clusters_)

# %% [markdown]
# ## 5.3. Cluster Evaluation
# 
# We evaluate the clusters using Silhouette Coefficient
# (sklearn.metrics.silhouette_score). Higher Silhouette Coefficient score means
# a model with better defined clusters. 

# %%
from sklearn import metrics
print("km", metrics.silhouette_score(X, k_means.labels_, metric='euclidean'))
print("ap", metrics.silhouette_score(X, ap.labels_, metric='euclidean'))

# %% [markdown]
# k-means has much higher Silhouette Coefficient as compared to the affinity
# propagation. 

# %% [markdown]
# # 6. Cluster Intuition
# In the next step, we will check each cluster and understand the intuition
# behind the clusters. 

# %%
cluster_output = pd.concat(
    [pd.DataFrame(X), pd.DataFrame(k_means.labels_, columns = ['cluster'])],
    axis = 1
)
output = cluster_output.groupby('cluster').mean()
pretty_print_df(output)

# %% [markdown]
# ### Demographics Features

# %%
output[['AGE','EDUC','MARRIED','KIDS','LIFECL','OCCAT']].plot.bar(
    rot=0, figsize=(18,5)
)
plt.show()

# %% [markdown]
# The plot here shows the average value of the attributes for each of the
# clusters. For example, comparing cluster 5 and cluster 3, cluster 5 has lower
# average age, yet higher average education. In terms of marriage,
# these two clusters are similar. So, the individuals in cluster 5
# will on an average have higher risk tolerance as compared to the individuals
# in cluster 3, based on the demographic attributes.

# %% [markdown]
# ### Financial Features and Features related to willingness to take risk

# %%
output[['HHOUSES','NWCAT','INCCL','WSAVED','SPENDMOR','RISK']].plot.bar(
    rot=0, figsize=(18,5)
)
plt.show()

# %% [markdown]
# The plot here shows the average value of the attributes for each of the
# cluster on the financial and behavioral attributes. For example, comparing
# cluster 5 and cluster 3, cluster 5 has higher average house ownership, higher
# average net worth and income, and a higher willingness to take risk. In terms
# of saving vs. income comparison and willingness to save, the two clusters are
# comparable. Therefore, we can posit that the individuals in cluster 5 will,
# on average, have a higher ability, higher willingness, to take risk
# compared with cluster 3.

# %%
sns.heatmap(output.T)
plt.show()

# %% [markdown]
# Combining the information from the demographics, financial, and behavioral
# attributes for cluster 0 and cluster 1, the overall ability to take risk for
# individual cluster 0 is higher as compared to cluster 1. Performing similar
# analysis across all other clusters, we summarize the results in the table
# below. The risk tolerance column represents the subjective assessment of the
# risk tolerance of each of the clusters.
# 
# <table>
# <tr>
# <td>Cluster</td>
# <td>Features</td>
# <td>Risk Capacity</td>
# </tr>
# <tr>
# <td>Cluster 0</td>
# <td>
# High age, high net worth and Income, highly risky life category, willingness
# to take risk, own house
# </td>
# <td>Medium</td>
# </tr>
# <tr>
# <td>Cluster 1</td>
# <td>
# Low age, low income and net worth, high willingness to take risk, many kids
# </td>
# <td>Low</td>
# </tr>
# <tr>
# <td>Cluster 2</td>
# <td>
# Low age, low income and net worth, high willingness to take risk, many
# kids, own house
# </td>
# <td>Low</td>
# </tr>
# <tr>
# <td>Cluster 3</td>
# <td>
# High age, low net worth and Income, highly risky life category, willingness
# to take risk, low education
# </td>
# <td>High</td>
# </tr>
# <tr>
# <td>Cluster 4</td>
# <td>
# Medium age, high income and net worth, willingness to take risk, many kids,
# own house
# </td>
# <td>High</td>
# </tr>
# <tr>
# <td>Cluster 5</td>
# <td>
# Low age, high networth and income, Less risky life category, willingness to
# spend more
# </td>
# <td>High</td>
# </tr>
# <tr>
# <td>Cluster 6</td>
# <td>
# Low age, low income and net worth, high willingness to take risk, no kids
# </td>
# <td>Medium</td>
# </tr>
# </table>

# %% [markdown]
# **_Conclusion_**

# %% [markdown]
# One of the key takeaways from this case study is the approach to understand
# the cluster intuition. We used visualization techniques to understand the
# expected behavior of a cluster member by qualitatively interpreting mean
# values of the variables in each cluster. 
# 
# We demonstrate the efficiency of the clustering technique in discovering the
# natural intuitive groups of different investors based on their risk
# tolerance.
# 
# Given, the clustering algorithms can successfully group investors based on
# different factors, such as age, income, and risk tolerance, it can further
# used by portfolio managers to understand the investor’s behavior and
# standardize the portfolio allocation and rebalancing across the clusters,
# making the investment management process faster and effective.
