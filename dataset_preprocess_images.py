from skimage.transform import resize
from skimage.io import imread, imshow
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage import exposure, data
from io import BytesIO
from os import listdir, mkdir

import numpy as np
import matplotlib.pyplot as plt
import PIL
import pandas as pd

# Group 7 races
class1 = 'staffordshire_bull_terrier'
class2 = 'wheaten_terrier'
class3 = 'samoyed'
class4 = 'Bombay'
class5 = 'Ragdoll'

def print_img_info(imgarr_arr, filename, flag):
    '''Function to print and plot a single image'''

    if flag:
        imshow(imgarr_arr[0])
        plt.savefig(f'./{filename}.png')
        print(imgarr_arr[0].shape)

# Read images
dataset_filename_list = listdir('./dataset')

print(f'>>> Number of images: {len(dataset_filename_list)}')

dataset = []

for img_name in dataset_filename_list:
    img = PIL.Image.open(f'./dataset/{img_name}')
    dataset.append(np.array(img.convert('RGB')))

print_img_info(dataset, 'normal_test', False)

# Resize images
print('>>> Resizing images')
resized_dataset = list(map(lambda imgarr : resize(imgarr, (128, 128)), dataset))

print_img_info(resized_dataset, 'resized_test', False)

# Applying grayscale
print('>>> Applying grayscale')
grayscale_dataset = list(map(rgb2gray, resized_dataset))

print_img_info(grayscale_dataset, 'grayscale_test', False)

# HOG
# 16 x 16
sixteen_pixel_fd = []
sixteen_pixel_images = []

print('>>> Calculating hog 16x16')
for img in grayscale_dataset:
    fd, hog_img = hog(img, orientations=9, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), visualize=True)
    sixteen_pixel_fd.append(fd)
    sixteen_pixel_images.append(hog_img)

#imshow(sixteen_pixel_images[0][1])
#plt.savefig('./hog16_test.png')
#print(sixteen_pixel_images[0][0].shape)

# 20 x 20
twenty_pixel_fd = []
twenty_pixel_images = []

print('>>> Calculating hog 20x20')
for img in grayscale_dataset:
    fd, hog_img = hog(img, orientations=9, pixels_per_cell=(20, 20),
                      cells_per_block=(2, 2), visualize=True)
    twenty_pixel_fd.append(fd)
    twenty_pixel_images.append(hog_img)

#imshow(twenty_pixel_images[0][1])
#plt.savefig('./hog20_test.png')
#print(twenty_pixel_images[0][0].shape)

# Creating list of classes
print('>>> Generating dataframes')
classes = []
for name in dataset_filename_list:

    race_class = ''

    if class1 in name:
        race_class = class1
    elif class2 in name:
        race_class = class2
    elif class3 in name:
        race_class = class3
    elif class4 in name:
        race_class = class4
    else:
        race_class = class5

    classes.append(race_class.lower())

# 16x16 dataframe and csv
dset_16 = {}

for fd_idx in range(0, len(sixteen_pixel_fd[0])):
    key = f'fd_{fd_idx}'
    value = []
    for img_idx in range(0, len(sixteen_pixel_fd)):
        value.append(sixteen_pixel_fd[img_idx][fd_idx])

    dset_16.update({key: value})

dset_16.update({'class' : classes})

dataframe_sixteen = pd.DataFrame.from_dict(dset_16)
# }}}

# 20x20 dataframe and csv
dset_20 = {}

for fd_idx in range(0, len(twenty_pixel_fd[0])):
    key = f'fd_{fd_idx}'
    value = []
    for img_idx in range(0, len(twenty_pixel_fd)):
        value.append(twenty_pixel_fd[img_idx][fd_idx])

    dset_20.update({key: value})

dset_20.update({'class' : classes})

dataframe_twenty = pd.DataFrame.from_dict(dset_20)
# }}}

# Writing the csvs
print('>>> Generating csv')
dataframe_sixteen.to_csv('./csv/sixteen_pixel_hog.csv',
                         sep = ';',
                         index = None)

dataframe_twenty.to_csv('./csv/twenty_pixel_hog.csv',
                         sep = ';',
                         index = None)
