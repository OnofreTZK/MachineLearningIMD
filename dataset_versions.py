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
# Standardizing
features_16 = df_sixteen.columns.drop('class')
features_20 = df_twenty.columns.drop('class')

print('>>> Standardizing the attributes')
X_16_scaled = StandardScaler().fit_transform(df_sixteen.loc[:, features_16].values)
X_20_scaled = StandardScaler().fit_transform(df_twenty.loc[:, features_20].values)

Y_16_classes = df_sixteen.loc[:, ['class']].values
Y_20_classes = df_twenty.loc[:, ['class']].values

# Applying PCA
pca_16_components = 65
pca_20_components = 30

pca_16 = PCA(n_components=pca_16_components)
pca_20 = PCA(n_components=pca_20_components)

print('>>> Applying PCA')
applied_pca_16 = pca_16.fit_transform(X_16_scaled)
applied_pca_20 = pca_20.fit_transform(X_20_scaled)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Experimenting zone !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# reference: https://towardsdatascience.com/how-to-select-the-best-number-of-principal-components-for-the-dataset-287e64b14c6d#:~:text=If%20our%20sole%20intention%20of,variables%20in%20the%20original%20dataset.
# I want to find the bestest n_components number
def plot_pca_visu(fig_name, pca, n_components):

    exp_var = pca.explained_variance_ratio_ * 100
    cum_exp_var = np.cumsum(exp_var)

    plt.bar(range(0, n_components), exp_var, align='center',
            label='Individual explained variance')

    plt.step(range(0, n_components), cum_exp_var, where='mid',
             label='Cumulative explained variance', color='red')

    plt.ylabel('Explained variance percentage')
    plt.xlabel('Principal component index')
    #plt.xticks(ticks=[1, 2, 3, 4])
    plt.legend(loc='best')
    plt.tight_layout()

    plt.savefig(f'barplot_{fig_name}.png')

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

plot_pca_visu('pca_16', pca_16, pca_16_components)
plot_pca_visu('pca_20', pca_20, pca_20_components)

# After some runs I decided to use 65 components for 16 pixels and 30 components for 20 pixels.

# Generate PCA csvs
print('>>> Generating Dataframes')
pca_16_df = pd.DataFrame(data = applied_pca_16,
                         columns = [f'principal_component_{n}' for n in range(pca_16_components)])

pca_20_df = pd.DataFrame(data = applied_pca_20,
                         columns = [f'principal_component_{n}' for n in range(pca_20_components)])

pca_16_with_class_df = pd.concat([pca_16_df, df_sixteen[['class']]], axis = 1)
pca_20_with_class_df = pd.concat([pca_20_df, df_twenty[['class']]], axis = 1)

print('>>> Generating csv')
pca_16_with_class_df.to_csv('./csv/sixteen_pixel_pca.csv',
                            sep = ';',
                            index = None)

pca_20_with_class_df.to_csv('./csv/twenty_pixel_pca.csv',
                            sep = ';',
                            index = None)
