from skimage.transform import resize
from skimage.io import imread, imshow
from skimage import exposure, data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from io import BytesIO
from os import listdir, mkdir

import numpy as np
import matplotlib.pyplot as plt
import PIL
import pandas as pd

print('>>> Reading csv and creating dataframes')
df_sixteen = pd.read_csv('./csv/sixteen_pixel_hog.csv', sep = ';')

df_twenty = pd.read_csv('./csv/twenty_pixel_hog.csv', sep = ';')

#print((df_sixteen.loc[df_sixteen['class']=='ragdoll']).count())
features_16 = df_sixteen.columns.drop('class')
features_20 = df_twenty.columns.drop('class')

X_16 = StandardScaler().fit_transform(df_sixteen.loc[:, features_16].values)
X_20 = StandardScaler().fit_transform(df_twenty.loc[:, features_20].values)

Y_16 = df_sixteen.loc[:, ['class']].values
Y_20 = df_twenty.loc[:, ['class']].values
