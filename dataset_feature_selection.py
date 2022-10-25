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

features_16 = df_sixteen.loc[:, (df_sixteen.columns.drop('class'))].values

class_16 = df_sixteen.loc[:, ['class']].values

# Feature Selection 
print('>>> Applying feature selection with tree classifier')
n_estimator_16 = 1000

clf_16 = ExtraTreesClassifier(n_estimators=n_estimator_16)

clf_16.fit(features_16, class_16.ravel())

model_16 = SelectFromModel(clf_16, prefit=True)

X_16_selected = model_16.transform(features_16)

# Generating CSV
print('>>> Generating Dataframe')
feature_selection_16_df = pd.DataFrame(data = X_16_selected,
                                       columns = [f'feature_{n}' for n in range(len(X_16_selected[0]))])

feature_selection_16_df = pd.concat([feature_selection_16_df, df_sixteen[['class']]], axis = 1)

print('>>> Generating CSV')
feature_selection_16_df.to_csv('./csv/sixteen_pixel_feature_selection.csv',
                               sep = ';',
                               index = None)

