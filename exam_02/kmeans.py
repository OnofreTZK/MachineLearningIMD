from numpy import mean
from numpy import std
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold # Import kfold function
from sklearn.model_selection import train_test_split # Import split percentage function
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the databases
print('>>> Reading the CSV')
df = pd.read_csv('./Abalone_NoClass.csv')
df.info()

# print(df.head)
print('KM version 1')
km_1 = KMeans(n_clusters=3)
km_1.fit(df)
centroids_1 = km_1.cluster_centers_

plt.scatter(df.iloc[:,0], df.iloc[:,9])
plt.scatter(centroids_1[:,0], centroids_1[:, 1], c='red', s=300)
plt.show()

print('KM version 2')
km_2 = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
km_2.fit(df)
km_2.fit_predict(df)

# Joining the labels
df['Cluster'] = km_2.labels_
df['Cluster'] = 'cluster' + df['Cluster'].astype(str)

# New attributes
print(df.head())

df.to_csv('kmeans_clustered_k3_abalone.csv')
