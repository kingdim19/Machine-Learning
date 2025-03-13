# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:29:40 2020

@author: user
"""
##################################################################################################
##################################################################################################
##################################################################################################
#Example: Riding Mowers	Part 1
import pandas as pd #import pandas module
import numpy as np #import numpy module
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier as KNN#import kNN

data = pd.read_csv('D:/Riding Mowers Data (Python).csv') #load the data
print(data.head()) #show the first several rows of the data
X,y = data.iloc[:,:-1],data.iloc[:,-1] #Assign the data to X and label to y
#knn = KNN(n_neighbors=3) #Set k-nn to be 3-nn
#knn.fit(X,y) #fit the data
#print( knn.predict(np.array([50,20]).reshape((1,-1)))) #predict the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
for k in range(1,10):
    knn=KNN(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    print(k,metrics.accuracy_score(y_pred,y_test))
#Example: Riding Mowers	Part 2
import pandas as pd #import pandas module
import numpy as np #import numpy module
from sklearn import metrics #import metrics
from sklearn.neighbors import KNeighborsClassifier #import kNN
from sklearn.model_selection import train_test_split #import train_test_split

data = pd.read_csv('Data\\Riding Mowers Data (Python).csv') #load the data
X,y = data.iloc[:,:-1],data.iloc[:,-1] #Assign the data to X and label to y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=0)
#Split the data randomly, 30% is for testing. the remaining is for training
for k in range(1,10): #Try 9 different k-nn settings
    knn = KNeighborsClassifier(n_neighbors=k) #Apply k-nnn with different values of k
    knn.fit(X_train,y_train) #fit the data
    y_pred = knn.predict(X_test) #Predict
    print('Correct rate with ' + str(k) + '-NN:' + str(metrics.accuracy_score(y_test, y_pred)))
    #metrics.accuracy_score(y_test, y_pred)-> calculate the accuracy
    
    

    
#Example: Riding Mowers	Part 3
import pandas as pd #import pandas module
import numpy as np #import numpy module
from sklearn import metrics #import metrics
from sklearn.neighbors import KNeighborsClassifier #import kNN
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split #import train_test_split


data = pd.read_csv('Data\\Riding Mowers Data (Python).csv') #load the data
X,y = data.iloc[:,:-1],data.iloc[:,-1] #Assign the data to X and label to y

#Grid Search CV
i = 0; #random state

knn = KNeighborsClassifier() 
parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11]}
##Apply GridSearchCV one time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=i)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=i); #Apply 5-Fold CV
clf = GridSearchCV(estimator=knn, param_grid=parameters, cv=inner_cv)
clf.fit(X_train, y_train)
non_nested_score = clf.best_score_
print(non_nested_score)
print(clf.score(X_test,y_test))

###Apply GridSearchCV 10 times
inner_cv = KFold(n_splits=5, shuffle=True, random_state=i) #Apply 5-Fold CV
outer_cv = KFold(n_splits=10, shuffle=True, random_state=i) #Apply 10-Fold CV
clf = GridSearchCV(estimator=knn, param_grid=parameters, cv=inner_cv)
nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv).mean()
print(nested_score)



    
#Example: Riding Mowers	Part 4
import pandas as pd #import pandas module
import numpy as np #import numpy module
from sklearn import metrics #import metrics
from sklearn.neighbors import KNeighborsClassifier #import kNN
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scipy.stats import randint

data = pd.read_csv('Data\\Riding Mowers Data (Python).csv') #load the data
X,y = data.iloc[:,:-1],data.iloc[:,-1] #Assign the data to X and label to y

#Randomized Search CV
i = 0; #random state

knn = KNeighborsClassifier() 
parameters = {'n_neighbors':randint(1,11)}
##Apply GridSearchCV one time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=i)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=i); #Apply 5-Fold CV
clf = RandomizedSearchCV(estimator=knn, param_distributions=parameters, cv=inner_cv)
clf.fit(X_train, y_train)
non_nested_score = clf.best_score_
print(non_nested_score)
print(clf.score(X_test,y_test))

###Apply RandomizedSearchCV 10 times
inner_cv = KFold(n_splits=5, shuffle=True, random_state=i) #Apply 5-Fold CV
outer_cv = KFold(n_splits=10, shuffle=True, random_state=i) #Apply 10-Fold CV
clf = RandomizedSearchCV(estimator=knn, param_distributions=parameters, cv=inner_cv)
nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv).mean()
print(nested_score)
