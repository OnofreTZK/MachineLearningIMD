from numpy import mean
from numpy import std
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import KFold # Import kfold function
from sklearn.model_selection import train_test_split # Import split percentage function
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc

# Read the databases
print('>>> Reading the CSV')
df = pd.read_csv('./Abalone_NoClass.csv')
df.info()

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
cluster.fit(df)

plt.figure(figsize=(10,7))
plt.title('Abalone Dendograms')
dend = shc.dendrogram(shc.linkage(df, method='complete'))
plt.savefig('agglo_dendrogram.png')

cluster.fit_predict(df)
print(cluster.labels_)

df['cluster'] = cluster.labels_

print(df.head)

df.to_csv('agglo_clustered_3_abalone.csv')
