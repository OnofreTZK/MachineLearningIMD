from numpy import mean
from numpy import std
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold # Import mdfold function
from sklearn.model_selection import train_test_split # Import split percentage function
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Read the databases
print('>>> Reading the CSVs')
df_16_hog = pd.read_csv('./csv/sixteen_pixel_hog.csv', sep = ';')
df_16_pca = pd.read_csv('./csv/sixteen_pixel_pca.csv', sep = ';')
df_16_fs = pd.read_csv('./csv/sixteen_pixel_feature_selection.csv', sep = ';')
df_20_hog = pd.read_csv('./csv/twenty_pixel_hog.csv', sep = ';')

datasets = [df_16_hog, df_16_pca, df_16_fs, df_20_hog]

print(df_16_hog.head)
print(df_16_pca.head)
print(df_16_fs.head)
print(df_20_hog.head)
sys.exit(0)

# Raw DataFrame structure
bases = ['base_16_hog', 'base_16_pca', 'base_16_fs', 'base_20_hog']
training_test = ['10-fold CV', '70/30', '80/20', '90/10']

mds = {
    'md3' : [],
    'md4' : [],
    'md5' : [],
    'md6' : [],
    'md7' : []
}

print('>>> Running Decision Tree with 20 configurations on each dataset')
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

    for md in range(3, 8):
        print(f'>>>>>> BASE: {base} | MD : {md}')
        # Create model object
        dt = DecisionTreeClassifier(criterion='entropy', max_depth=md)

        # Accuracy
        # 10 fold cv
        scores = cross_val_score(dt, X, Y.ravel(), scoring='accuracy', cv=kf)
        print(f'10-fold CV max depth = {md} Accuracy: {mean(scores)} ({std(scores)})')
        mds[f'md{md}'].append(mean(scores))
        #plt.figure(figsize=(12, 12))
        #plt.savefig(f'./trees/kfold_10_md_{md}.png')

        # 70/30
        dt.fit(X_train_70, Y_train_70.ravel())
        Y_pred_70_30 = dt.predict(X_test_30)
        accuracy_70_30 = metrics.accuracy_score(Y_test_30, Y_pred_70_30)
        print(f'70/30 max depth = {md} Accuracy: {accuracy_70_30}')
        mds[f'md{md}'].append(accuracy_70_30)

        # 80/20
        dt.fit(X_train_80, Y_train_80.ravel())
        Y_pred_80_20 = dt.predict(X_test_20)
        accuracy_80_20 = metrics.accuracy_score(Y_test_20, Y_pred_80_20)
        print(f'80/20 max depth = {md} Accuracy: {accuracy_80_20}')
        mds[f'md{md}'].append(accuracy_80_20)

        # 90/10
        dt.fit(X_train_90, Y_train_90.ravel())
        Y_pred_90_10 = dt.predict(X_test_10)
        accuracy_90_10 = metrics.accuracy_score(Y_test_10, Y_pred_90_10)
        print(f'90/10 max_depth = {md} Accuracy: {accuracy_90_10}')
        mds[f'md{md}'].append(accuracy_90_10)


