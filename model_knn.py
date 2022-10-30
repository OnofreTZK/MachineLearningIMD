from numpy import mean
from numpy import std
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold # Import kfold function
from sklearn.model_selection import train_test_split # Import split percentage function
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the databases
print('>>> Reading the CSVs')
df_16_hog = pd.read_csv('./csv/sixteen_pixel_hog.csv', sep = ';')
df_16_pca = pd.read_csv('./csv/sixteen_pixel_pca.csv', sep = ';')
df_16_fs = pd.read_csv('./csv/sixteen_pixel_feature_selection.csv', sep = ';')
df_20_hog = pd.read_csv('./csv/twenty_pixel_hog.csv', sep = ';')

datasets = [df_16_hog, df_16_pca, df_16_fs, df_20_hog]

# Raw DataFrame structure
bases = ['base_16_hog', 'base_16_pca', 'base_16_fs', 'base_20_hog']
training_test = ['10-fold CV', '70/30', '80/20', '90/10']
ks = {
    'k1' : [],
    'k2' : [],
    'k3' : [],
    'k4' : [],
    'k5' : []
}

print('>>> Running kNN with 20 configurations on each dataset')
for ds, base in zip(datasets, bases):
    # Features and classes
    features = ds.columns.drop('class')
    X = ds.loc[:, features].values
    Y = ds.loc[:, ['class']].values

    # KFold 10
    kf = KFold(n_splits=10, random_state=1, shuffle=True)

    # Split percentage
    ## 70/30
    X_train_70, X_test_30, Y_train_70, Y_test_30 = train_test_split(X,
                                                                    Y,
                                                                    test_size=0.3,
                                                                    random_state=1)

    ## 80/20
    X_train_80, X_test_20, Y_train_80, Y_test_20 = train_test_split(X,
                                                                    Y,
                                                                    test_size=0.2,
                                                                    random_state=1)

    ## 90/10
    X_train_90, X_test_10, Y_train_90, Y_test_10 = train_test_split(X,
                                                                    Y,
                                                                    test_size=0.1,
                                                                    random_state=1)

    for k in range(1, 6):
        print(f'>>>>>> BASE: {base} | K : {k}')
        # Create model object
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

        # Accuracy
        # 10 fold cv
        scores = cross_val_score(knn, X, Y.ravel(), scoring='accuracy', cv=kf)
        print(f'10-fold CV {k}K Accuracy: {mean(scores)} ({std(scores)})')
        ks[f'k{k}'].append(mean(scores))

        # 70/30
        knn.fit(X_train_70, Y_train_70.ravel())
        Y_pred_70_30 = knn.predict(X_test_30)
        accuracy_70_30 = metrics.accuracy_score(Y_test_30, Y_pred_70_30)
        print(f'70/30 {k}K Accuracy: {accuracy_70_30}')
        ks[f'k{k}'].append(accuracy_70_30)

        # 80/20
        knn.fit(X_train_80, Y_train_80.ravel())
        Y_pred_80_20 = knn.predict(X_test_20)
        accuracy_80_20 = metrics.accuracy_score(Y_test_20, Y_pred_80_20)
        print(f'80/20 {k}K Accuracy: {accuracy_80_20}')
        ks[f'k{k}'].append(accuracy_80_20)

        # 90/10
        knn.fit(X_train_90, Y_train_90.ravel())
        Y_pred_90_10 = knn.predict(X_test_10)
        accuracy_90_10 = metrics.accuracy_score(Y_test_10, Y_pred_90_10)
        print(f'90/10 {k}K Accuracy: {accuracy_90_10}')
        ks[f'k{k}'].append(accuracy_90_10)

print('>>> Generating DataFrame')
columns = ['base', 'training_test', '1k', '2k', '3k', '4k', '5k']
data = []
dataframe_bases = []
training_tests = []

for base in bases:
    for _ in range(4):
        dataframe_bases.append(base)

dataframe_bases.append('mean')
dataframe_bases.append('standard')

data.append(dataframe_bases)

for _ in range(4):
    training_tests.append(training_test[0])
    training_tests.append(training_test[1])
    training_tests.append(training_test[2])
    training_tests.append(training_test[3])

training_tests.append('')
training_tests.append('')

data.append(training_tests)

for result in ks.values():
    result.append(mean(result))
    result.append(std(result))
    data.append(result)

df = pd.DataFrame()

columns.reverse()
data.reverse()

for col, row in zip(columns, data):
    df.insert(loc=0, column=col, value=row)

print(df.head)

print('>>> Writing Excel')
df.to_excel('./sheets/knn_results.xlsx', sheet_name='kNN Results')
