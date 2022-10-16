from skimage.transform import resize
from skimage.io import imread, imshow
from skimage import exposure, data
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from io import BytesIO
from os import listdir, mkdir

import numpy as np
import matplotlib.pyplot as plt
import PIL
import pandas as pd

print('>>> Reading csv and creating dataframes')
df_sixteen = pd.read_csv('./csv/sixteen_pixel_hog.csv', sep = ';')
df_twenty = pd.read_csv('./csv/twenty_pixel_hog.csv', sep = ';')

features_16 = df_sixteen.loc[:, (df_sixteen.columns.drop('class'))].values
features_20 = df_twenty.loc[:, (df_twenty.columns.drop('class'))].values

class_16 = df_sixteen.loc[:, ['class']].values
class_20 = df_twenty.loc[:, ['class']].values

#print('>>> Thresholding')
#X_16_threshold = .5
#X_20_threshold = .5
#
#selection_16 = VarianceThreshold(threshold=X_16_threshold)
#selection_20 = VarianceThreshold(threshold=X_20_threshold)
#
#sel_16 = selection_16.fit_transform(features_16)
#sel_20 = selection_20.fit_transform(features_20)
#
#print(sel_16.shape)
#print(sel_20.shape)

n_estimator_16 = 1000
n_estimator_20 = 1000

clf_16 = ExtraTreesClassifier(n_estimators=n_estimator_16)
clf_20 = ExtraTreesClassifier(n_estimators=n_estimator_20)

clf_16.fit(features_16, class_16.ravel())
clf_20.fit(features_20, class_20.ravel())

model_16 = SelectFromModel(clf_16, prefit=True)
model_20 = SelectFromModel(clf_20, prefit=True)

X_16_selected = model_16.transform(features_16)
X_20_selected = model_20.transform(features_20)

print(X_16_selected.shape)
print(X_20_selected.shape)
