# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 17:11:07 2021

@author: DHRUV DOSHI
"""
#imports
import numpy as np 
import pandas as pd
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
from sklearn.metrics import davies_bouldin_score
import seaborn as sns

%matplotlib inline

#reading the dataset and mapping gender attribute to boolean
df = pd.read_csv("C:/Users/HP/Downloads/Mall_Customers.csv")
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df.head()

#standardizing the data
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
X = StandardScaler().fit_transform(X)
X[0:5]

#setting the epsilon value to 0.8
epsilon = 0.8

#minimumSamples will be set
dbscore = {}
no_of_clusters = {}

#finding the best value for minimum samples parameter
for minimum in range(3,12):
    db = DBSCAN(eps=epsilon, min_samples=minimum).fit(X)
    labels = db.labels_
    score = davies_bouldin_score(X, labels)
    dbscore[minimum] = score
    no_of_clusters[minimum] = len(np.unique(labels))-1
  
#plotting Minimum samples vs DB Score
plt.plot(dbscore.keys(),dbscore.values())
plt.xlabel(xlabel="Minimum samples")
plt.ylabel(ylabel="DB SCORE")

#plotting Minimum samples vs No of clusters
plt.plot(no_of_clusters.keys(),no_of_clusters.values())
plt.xlabel(xlabel="Minimum samples")
plt.ylabel(ylabel="No of Clusters")

# the best parameter values: epsilon=0.8 and minimumSamples=7
epsilon = 0.8
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_
df["Labels"] = labels

#plotting cluster frequency
sns.countplot(x=labels, data=df)
plt.xlabel(xlabel="Label")
plt.ylabel(ylabel="Count")
