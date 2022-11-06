from numpy import mean
from numpy import std
from statistics import median
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
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
activation_funcs = ['identity', 'logistic', 'tanh', 'relu']

tests_results = []

## Auxiliar Functions
def for_each_config(model, X, Y, config_description):
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
    # 10 fold cv
    scores = cross_val_score(model, X, Y.ravel(), scoring='accuracy', cv=kf)
    aux_str = f'10-fold {config_description} Accuracy: {mean(scores)} ({std(scores)})'
    tests_results.append(aux_str)
    print(aux_str)

    # 70/30
    model.fit(X_train_70, Y_train_70.ravel())
    Y_pred_70_30 = model.predict(X_test_30)
    accuracy_70_30 = metrics.accuracy_score(Y_test_30, Y_pred_70_30)
    aux_str = f'70/30 {config_description} Accuracy: {accuracy_70_30}'
    tests_results.append(aux_str)
    print(aux_str)

    # 80/20
    model.fit(X_train_80, Y_train_80.ravel())
    Y_pred_80_20 = model.predict(X_test_20)
    accuracy_80_20 = metrics.accuracy_score(Y_test_20, Y_pred_80_20)
    aux_str = f'80/20 {config_description} Accuracy: {accuracy_80_20}'
    tests_results.append(aux_str)
    print(aux_str)

    # 90/10
    model.fit(X_train_90, Y_train_90.ravel())
    Y_pred_90_10 = model.predict(X_test_10)
    accuracy_90_10 = metrics.accuracy_score(Y_test_10, Y_pred_90_10)
    aux_str = f'90/10 {config_description} Accuracy: {accuracy_90_10}'
    tests_results.append(aux_str)
    print(aux_str)

# TEST RUNS
# 1
def const_A():
    '''Running MLP with A = (number of attributes + classes) / 2'''
    print('>>> Running MLP with A = (number of attributes + classes) / 2')

    ds = datasets[0]

    A = int(((len(ds.columns) - 1) + 5)/2)

    # Features and classes
    features = ds.columns.drop('class')
    X = ds.loc[:, features].values
    Y = ds.loc[:, ['class']].values

    for func in activation_funcs:
        print(f'>>>>>> BASE: {bases[0]} | FUNC: {func} | NEURONS: {A}')
        mlp = MLPClassifier(hidden_layer_sizes=A,
                            activation=func,
                            solver='adam',
                            max_iter=500)

        for_each_config(mlp, X, Y, f'const A {func}')


    for result in tests_results:
            print(result)

# 2
def different_neurons():
    '''Running MLP with different number of neurons:
    A  = attr + class / 2
    O  = class
    T  = attr
    AO = Median value between A and O
    AT = Median value between A and T

    this one requires to plot a graph where:
    axis X is the number of neurons and axis Y is the number of accuracy'''
    print('>>> Running MLP with different number of neurons')

    ds = datasets[0]

    A  = int(((len(ds.columns) - 1) + 5)/2)
    O  = 5
    T  = len(ds.columns) - 1
    AO = int(median([A, O]))
    AT = int(median([A, T]))

    neurons = [A, O, T, AO, AT]

    # Features and classes
    features = ds.columns.drop('class')
    X = ds.loc[:, features].values
    Y = ds.loc[:, ['class']].values

    for neuron in neurons:
        print(f'>>>>>> BASE: {bases[0]} | FUNC: relu | NEURONS: {neuron}')
        mlp = MLPClassifier(hidden_layer_sizes=neuron,
                            activation='relu',
                            solver='adam',
                            max_iter=500)

        for_each_config(mlp, X, Y, f'neurons {neuron}')


    for result in tests_results:
            print(result)




#print('>>> Running MLP with n configurations on each dataset')
#for ds, base in zip(datasets, bases):
#    # Features and classes
#    features = ds.columns.drop('class')
#    X = ds.loc[:, features].values
#    Y = ds.loc[:, ['class']].values
#
#    # KFold 10
#    kf = KFold(n_splits=10, random_state=1, shuffle=True)
#
#    # Split percentage
#    ## 70/30
#    X_train_70, X_test_30, Y_train_70, Y_test_30 = train_test_split(X,
#                                                                    Y,
#                                                                    test_size=0.3,
#                                                                    random_state=1)
#
#    ## 80/20
#    X_train_80, X_test_20, Y_train_80, Y_test_20 = train_test_split(X,
#                                                                    Y,
#                                                                    test_size=0.2,
#                                                                    random_state=1)
#
#    ## 90/10
#    X_train_90, X_test_10, Y_train_90, Y_test_10 = train_test_split(X,
#                                                                    Y,
#                                                                    test_size=0.1,
#                                                                    random_state=1)
#
#    for func in activation_funcs:
#        print(f'>>>>>> BASE: {base} | K : {k}')
#        # Create model object
#        mlp = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
#
#        # Accuracy
#        # 10 fold cv
#        scores = cross_val_score(mlp, X, Y.ravel(), scoring='accuracy', cv=kf)
#        print(f'10-fold CV {k}K Accuracy: {mean(scores)} ({std(scores)})')
#        ks[f'k{k}'].append(mean(scores))
#
#        # 70/30
#        mlp.fit(X_train_70, Y_train_70.ravel())
#        Y_pred_70_30 = mlp.predict(X_test_30)
#        accuracy_70_30 = metrics.accuracy_score(Y_test_30, Y_pred_70_30)
#        print(f'70/30 {k}K Accuracy: {accuracy_70_30}')
#        ks[f'k{k}'].append(accuracy_70_30)
#
#        # 80/20
#        mlp.fit(X_train_80, Y_train_80.ravel())
#        Y_pred_80_20 = mlp.predict(X_test_20)
#        accuracy_80_20 = metrics.accuracy_score(Y_test_20, Y_pred_80_20)
#        print(f'80/20 {k}K Accuracy: {accuracy_80_20}')
#        ks[f'k{k}'].append(accuracy_80_20)
#
#        # 90/10
#        mlp.fit(X_train_90, Y_train_90.ravel())
#        Y_pred_90_10 = mlp.predict(X_test_10)
#        accuracy_90_10 = metrics.accuracy_score(Y_test_10, Y_pred_90_10)
#        print(f'90/10 {k}K Accuracy: {accuracy_90_10}')
#        ks[f'k{k}'].append(accuracy_90_10)
#
#print('>>> Generating DataFrame')
#columns = ['base', 'training_test', '1k', '2k', '3k', '4k', '5k']
#data = []
#dataframe_bases = []
#training_tests = []
#
#for base in bases:
#    for _ in range(4):
#        dataframe_bases.append(base)
#
#dataframe_bases.append('mean')
#dataframe_bases.append('standard')
#
#data.append(dataframe_bases)
#
#for _ in range(4):
#    training_tests.append(training_test[0])
#    training_tests.append(training_test[1])
#    training_tests.append(training_test[2])
#    training_tests.append(training_test[3])
#
#training_tests.append('')
#training_tests.append('')
#
#data.append(training_tests)
#
#for result in ks.values():
#    result.append(mean(result))
#    result.append(std(result))
#    data.append(result)
#
#df = pd.DataFrame()
#
#columns.reverse()
#data.reverse()
#
#for col, row in zip(columns, data):
#    df.insert(loc=0, column=col, value=row)
#
#print(df.head)
#
#print('>>> Writing Excel')
#df.to_excel('./sheets/mlp_results.xlsx', sheet_name='Multilayer Perceptron Results')

# DONE const A to discover the best activation func to my base
# const_A()
# Best function to my base: relu

# DONE const activation func to discover the best number of neurons
# different_neurons()
# Best number of neurons: 884
