from skimage.transform import resize
from skimage.io import imread, imshow
from skimage.feature import hog
from skimage import exposure, data
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

# Read images
dataset_filename_list = listdir('./dataset')

dataset = []

for img_name in dataset_filename_list:
    img = PIL.Image.open(f'./dataset/{img_name}')
    dataset.append(np.array(img))

#imshow(dataset[0])
#plt.savefig('./normal_test.png')
#print(dataset[0].shape)

# Resize images
resized_dataset = list(map((lambda imgarr : resize(imgarr, (128, 128))), dataset))

#imshow(resized_dataset[0])
#plt.savefig('./resized_test.png')
#print(resized_dataset[0].shape)

# HOG

