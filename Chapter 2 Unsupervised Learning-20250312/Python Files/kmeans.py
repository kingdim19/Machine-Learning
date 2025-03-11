# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:40:43 2020

@author: bensonlam
"""
import numpy as np

def Kmeans_manh(data,n_clusters):
    epsilon = 1e-4; iter = 0; residual = 1;
    randint = np.random.randint(0,data.shape[0],size=n_clusters) #Randomly select two random integers
    X_new = data.iloc[randint,:].values #Find random points as the initial guesses
    while (residual > epsilon) & (iter <100):
        X_old = X_new.copy(); #Assign the updated one to the old one
        #Update cluster labels
        #Compute distance
        dist = [];
        for x in X_new: 
            dist.append(np.linalg.norm(data-x,axis=1,ord=1))
        labels = np.argmin(np.array(dist),axis=0)

        #Update cluster centroids
        for ii,x in enumerate(X_old): 
            X_new[ii,:] = np.median(data[labels==ii],axis=0)
        residual = np.linalg.norm(X_new-X_old)
        iter +=1
        return labels,X_new
