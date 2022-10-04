from skimage.transform import resize
from skimage.io import imread, imshow
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage import exposure, data
from sklearn.decomposition import PCA
from io import BytesIO
from os import listdir, mkdir

import numpy as np
import matplotlib.pyplot as plt
import PIL

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
sixteen_pixel_images = []

print('>>> Calculating hog 16x16')
for img in grayscale_dataset:
    fd, hog_img = hog(img, orientations=9, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), visualize=True)
    sixteen_pixel_images.append((fd, hog_img))

#imshow(sixteen_pixel_images[0][1])
#plt.savefig('./hog16_test.png')
#print(sixteen_pixel_images[0][0].shape)

# 20 x 20
twenty_pixel_images = []

print('>>> Calculating hog 20x20')
for img in grayscale_dataset:
    fd, hog_img = hog(img, orientations=9, pixels_per_cell=(20, 20),
                      cells_per_block=(2, 2), visualize=True)
    twenty_pixel_images.append((fd, hog_img))

#imshow(twenty_pixel_images[0][1])
#plt.savefig('./hog20_test.png')
#print(twenty_pixel_images[0][0].shape)
