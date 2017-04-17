
# coding: utf-8

# # Data wrangling

# Make your folder structure like this:
# 
# ```
# .
# ├── test
# ├── train
# │   ├── inputs
# │   └── targets
# ├── train_all
# │   ├── inputs
# │   └── targets
# └── valid
#     ├── inputs
#     └── targets
# ```
# 
# TODO: Use os.makedirs to create the structure for the notebook user

# ## Dependencies

# In[1]:

import shutil

from glob import glob
import pandas as pd
import tifffile as tiff
import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def plots(imgs, figsize=(12, 12), rows=1, cols=1, interp=None, titles=None, cmap=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(cmap, list):
        cmap = [cmap] * len(imgs)
    if not isinstance(interp, list):
        interp = [interp] * len(imgs)
    fig = plt.figure(figsize=figsize)
    for i in range(len(imgs)):
        sp = fig.add_subplot(rows, cols, i+1)
        if titles:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(imgs[i], interpolation=interp[i], cmap=cmap[i])
        plt.axis('off')
        
def in_interval(x, a, b):
    return x >= a and x <= b


# ## Split the dataset into train_all and test

# In[ ]:

INPUT_DIR = 'input/'


# In[2]:

filenames3 = glob(INPUT_DIR+'three_band/*.tif')
filenames16 = glob(INPUT_DIR+'sixteen_band/*.tif')
wkt_csv = pd.read_csv(INPUT_DIR+'train_wkt_v4.csv')

print(len(filenames3))  # 450 - 425 test == 25 train
print(filenames3[0])
print(len(filenames16)) # 1350 / 3 image types == 450
print(filenames16[0])
print(len(wkt_csv))  # 250 / 10 classes == 25 train
wkt_csv.head()


# In[3]:

train_IDs = list(set(wkt_csv.ImageId))

print(len(train_IDs))
print(train_IDs[0])


# In[4]:

filenames3_IDs = [f.split('/')[-1].split('.')[0] for f in filenames3]
train_filenames3 = [f for f, ID in zip(filenames3, filenames3_IDs) if ID in train_IDs]

print(len(filenames3_IDs))
print(len(train_filenames3))


# In[5]:

filenames16_IDs = [f.split('/')[-1].split('.')[0][:-2] for f in filenames16]
train_filenames16 = [f for f, ID in zip(filenames16, filenames16_IDs) if ID in train_IDs]

print(len(filenames16_IDs))
print(len(train_filenames16))  # 75 / 3 image types == 25 train


# In[14]:

train_filenames[0].split('/')[-1]


# In[12]:

train_filenames[-1]


# In[6]:

train_filenames = train_filenames3 + train_filenames16

print(len(train_filenames))


# In[16]:

for f in train_filenames: shutil.move(f, 'train_all/inputs/'+f.split('/')[-1])

print(len(glob('train_all/inputs/*.tif')))


# In[34]:

sorted(glob('train_all/inputs/*.tif'))[:8]


# In[24]:

print(len(filenames3))
print(len(filenames16))

filenames3 = glob(INPUT_DIR+'three_band/*.tif')
filenames16 = glob(INPUT_DIR+'sixteen_band/*.tif')

print(len(filenames3))
print(len(filenames16))


# In[32]:

for f in filenames3+filenames16: shutil.move(f, 'test/'+f.split('/')[-1])

print(len(glob('test/*.tif')))
sorted(glob('test/*.tif'))[:8]

