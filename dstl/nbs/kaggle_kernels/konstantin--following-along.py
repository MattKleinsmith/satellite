
# coding: utf-8

# In[1]:

from collections import defaultdict
import csv
import sys

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

csv.field_size_limit(sys.maxsize);


# In[184]:

import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

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


# In[3]:

grid_sizes = pd.read_csv('input/grid_sizes.csv')
grid_sizes.head()


# In[73]:

grid_sizes.shape


# In[88]:

get_ipython().system('ls ./input/three_band | wc -l')
get_ipython().system('ls ./input/sixteen_band/ | wc -l')
get_ipython().system('ls ./input/train_geojson_v3/ | wc -l')


# 450 * 3 == 1350
# 
# 
# The 3 corresponds to "A", "M", and "P". I don't know what these letters mean.
# 
# - Panchromatic: 0.31m 
# - Multispectral: 1.24 m
# - SWIR: Delivered at 7.5m
# 
# There's a "P" and an "M", but no "A".

# In[74]:

train_wkt = pd.read_csv('input/train_wkt_v4.csv')
train_wkt.head()


# In[75]:

train_wkt.shape


# There seems to be only 25 images with labels. 250 / 10 classes per image == 25 images.
# 
# If this is true, what do we do with the other 425 unlabeled images?

# In[5]:

polygon = train_wkt[3:4]
polygon


# In[6]:

n = 11
polygon = train_wkt[n:n+1]['MultipolygonWKT'].values[0]
polygon_values = polygon.split(',')
print(len(polygon_values), '\n')
for val in polygon_values[:10]: print(val)
shapely.wkt.loads(polygon)


# In[7]:

IM_ID = '6120_2_2'
POLY_TYPE = '1'  # buildings

# Load grid size
x_max = y_min = None
for _im_id, _x, _y in csv.reader(open('input/grid_sizes.csv')):
    if _im_id == IM_ID:
        x_max, y_min = float(_x), float(_y)
        break

# Load train poly with shapely
train_polygons = None
for _im_id, _poly_type, _poly in csv.reader(open('input/train_wkt_v4.csv')):
    if _im_id == IM_ID and _poly_type == POLY_TYPE:
        train_polygons = shapely.wkt.loads(_poly)
        break

# Read image with tiff
im_rgb = tiff.imread('input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
im_size = im_rgb.shape[:2]


# In[8]:

train_polygons


# In[9]:

def get_scalers():
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min

x_scaler, y_scaler = get_scalers()

train_polygons_scaled = shapely.affinity.scale(
    train_polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

train_polygons_scaled


# In[10]:

type(train_polygons_scaled)


# In[11]:

len(train_polygons_scaled)


# In[12]:

train_polygons_scaled[0]


# In[13]:

train_polygons_scaled[0].exterior


# In[14]:

# Why are there five points if the polygon only has four vertices?
np.array(train_polygons_scaled[0].exterior.coords)
# It seems there's an extra line, superimposed on another, making it invisible.
# Ah, I see now. The first point and the last point are identical.


# In[15]:

points = np.array(train_polygons_scaled[0].exterior.coords)
len(points)


# In[16]:

plt.scatter(points[:, 0], points[:, 1])


# In[17]:

train_polygons_scaled[0].centroid


# In[128]:

train_polygons_scaled[0:1000]


# In[19]:

polygons = train_polygons_scaled
int_coords = lambda x: np.array(x).round().astype(np.int32)
exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
interiors = [int_coords(pi.coords) for poly in polygons
             for pi in poly.interiors]
len(interiors), type(interiors)


# In[20]:

len(exteriors)


# In[21]:

len(interiors)


# In[22]:

interiors
# Only five interiors? I'm not sure what an interior is.
# A few cells down I removed the interiors from the mask generator.
## It has no qualitative effect on the mask.


# In[23]:

plt.scatter(interiors[0][:, 0], interiors[0][:, 1])


# In[24]:

def mask_for_polygons_no_interiors(polygons):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    #cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

train_mask = mask_for_polygons_no_interiors(train_polygons_scaled)
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(train_mask, cmap="gray");


# In[25]:

def mask_for_polygons(polygons):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

train_mask = mask_for_polygons(train_polygons_scaled)

fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(train_mask, cmap="gray");


# In[129]:

im_rgb.shape


# In[48]:

tiff.imshow(im_rgb);


# In[43]:

matrix = im_rgb
w, h, d = matrix.shape
matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
# Get 2nd and 98th percentile
mins = np.percentile(matrix, 1, axis=0)
maxs = np.percentile(matrix, 99, axis=0) - mins
matrix = (matrix - mins[None, :]) / maxs[None, :]
matrix = np.reshape(matrix, [w, h, d])
matrix = matrix.clip(0, 1)


# In[44]:

mins


# In[45]:

maxs


# In[46]:

tiff.imshow(matrix);


# In[47]:

matrix = im_rgb
w, h, d = matrix.shape
matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
# Get 2nd and 98th percentile
mins = np.percentile(matrix, 1, axis=0)
maxs = np.percentile(matrix, 99, axis=0) - mins
matrix = (matrix - maxs[None, :]) / mins[None, :]
matrix = np.reshape(matrix, [w, h, d])
matrix = matrix.clip(0, 1)

tiff.imshow(matrix);


# In[51]:

matrix = im_rgb
w, h, d = matrix.shape
matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
mins = np.percentile(matrix, 10, axis=0)
maxs = np.percentile(matrix, 90, axis=0) - mins
matrix = (matrix - mins[None, :]) / maxs[None, :]
matrix = np.reshape(matrix, [w, h, d])
matrix = matrix.clip(0, 1)

tiff.imshow(matrix);


# In[52]:

matrix = im_rgb
w, h, d = matrix.shape
matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
mins = np.percentile(matrix, 20, axis=0)
maxs = np.percentile(matrix, 70, axis=0) - mins
matrix = (matrix - mins[None, :]) / maxs[None, :]
matrix = np.reshape(matrix, [w, h, d])
matrix = matrix.clip(0, 1)

tiff.imshow(matrix);


# In[26]:

def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


# In[28]:

tiff.imshow(scale_percentile(im_rgb));


# # Widget

# In[29]:

# Stdlib imports
from io import BytesIO

# Third-party libraries
from IPython.display import Image as IImage
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import matplotlib as mpl
from skimage import data, filters, io, img_as_float
import numpy as np

def arr2img(arr):
    """Display a 2- or 3-d numpy array as an image."""
    if arr.ndim == 2:
        format, cmap = 'png', mpl.cm.gray
    elif arr.ndim == 3:
        format, cmap = 'jpg', None
    else:
        raise ValueError("Only 2- or 3-d arrays can be displayed as images.")
    # Don't let matplotlib autoscale the color range so we can control overall luminosity
    vmax = 255 if arr.dtype == 'uint8' else 1.0
    with BytesIO() as buffer:
        mpl.image.imsave(buffer, arr, format=format, cmap=cmap, vmin=0, vmax=vmax)
        out = buffer.getvalue()
    return IImage(out)

def crop_image(img, x1, x2, y1, y2):
    """image: numpy array"""
    cropped_img = scale_percentile(img[y1:y2, x1:x2])
    return arr2img(cropped_img)

x1_slider = widgets.IntSlider(min=0, max=3348, value=1786)
x2_slider = widgets.IntSlider(min=0, max=3348, value=2473)
y1_slider = widgets.IntSlider(min=0, max=3403, value=2441)
y2_slider = widgets.IntSlider(min=0, max=3403, value=3403)
band_slider = None

interact(crop_image, img=fixed(im_rgb), x1=x1_slider, x2=x2_slider, y1=y1_slider, y2=y2_slider);

# 2900:3200,2000:2300


# # Moving on

# In[57]:

fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(train_mask, cmap="gray");


# In[61]:

fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(train_mask[2900:3200,2000:2300], cmap="gray");


# In[64]:

tiff.imshow(scale_percentile(im_rgb[2900:3200,2000:2300]));


# # Logistic regression classifier

# In[91]:

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score

xs = im_rgb.reshape(-1, 3).astype(np.float32)
ys = train_mask.reshape(-1)
pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))

print('training...')
# do not care about overfitting here
pipeline.fit(xs, ys)
pred_ys = pipeline.predict_proba(xs)[:, 1]
print('average precision', average_precision_score(ys, pred_ys))
pred_mask = pred_ys.reshape(train_mask.shape)


# In[94]:

q = pipeline.predict_proba(xs)


# In[95]:

q.shape


# In[96]:

q[0:10]


# In[125]:

fig, ax = plt.subplots(figsize=(20, 10)); plt.axis('off')
fig.add_subplot(1, 2, 1)
plt.imshow(pred_mask[2900:3200,2000:2300], cmap="gray");
fig.add_subplot(1, 2, 2)
plt.imshow(train_mask[2900:3200,2000:2300], cmap="gray");


# # What are these images? Which are test images? How do we know?

# We can look at the WKT file to find the image IDs of the train images.

# In[62]:

sorted(list(set(train_wkt.ImageId)))


# Ah, maybe there are 25 train images and 425 test images.

# # Exploring on my own for a second

# In[1]:

from collections import defaultdict
import csv
import sys

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

csv.field_size_limit(sys.maxsize);

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

def get_scalers():
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min

def mask_for_polygons(polygons):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

###############################################################################

IM_ID = '6120_2_2'
POLY_TYPE = '1'  # buildings

# Load grid size
x_max = y_min = None
for _im_id, _x, _y in csv.reader(open('input/grid_sizes.csv')):
    if _im_id == IM_ID:
        x_max, y_min = float(_x), float(_y)
        break

# Load train poly with shapely
train_polygons = None
for _im_id, _poly_type, _poly in csv.reader(open('input/train_wkt_v4.csv')):
    if _im_id == IM_ID and _poly_type == POLY_TYPE:
        train_polygons = shapely.wkt.loads(_poly)
        break

# Read image with tiff
im_rgb = tiff.imread('input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
im_size = im_rgb.shape[:2]

x_scaler, y_scaler = get_scalers()

train_polygons_scaled = shapely.affinity.scale(train_polygons,
                                               xfact=x_scaler,
                                               yfact=y_scaler,
                                               origin=(0, 0, 0))

train_mask = mask_for_polygons(train_polygons_scaled)

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score

xs = im_rgb.reshape(-1, 3).astype(np.float32)
ys = train_mask.reshape(-1)
pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))

print('training...')
# do not care about overfitting here
pipeline.fit(xs, ys)
pred_ys = pipeline.predict_proba(xs)[:, 1]
print('average precision', average_precision_score(ys, pred_ys))
pred_mask = pred_ys.reshape(train_mask.shape)

fig, ax = plt.subplots(figsize=(20, 10)); plt.axis('off')
fig.add_subplot(1, 2, 1)
plt.imshow(pred_mask[2900:3200,2000:2300], cmap="gray");
fig.add_subplot(1, 2, 2)
plt.imshow(train_mask[2900:3200,2000:2300], cmap="gray");


# # Plotting 19 bands

# In[181]:

from collections import defaultdict
import csv
import sys

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

csv.field_size_limit(sys.maxsize);

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

def plots(imgs, figsize=(12, 12), rows=1, cols=1, interp=None, titles=None, cmap=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig = plt.figure(figsize=figsize)
    for i in range(len(imgs)):
        sp = fig.add_subplot(rows, cols, i+1)
        if titles:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(imgs[i], interpolation=interp, cmap=cmap)
        plt.axis('off')


# In[5]:

IM_ID = '6120_2_2'
y1, y2, x1, x2 = 2900, 3200, 2000, 2300
im_rgb = tiff.imread('input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])

fig, ax = plt.subplots(figsize=(20, 10)); plt.axis('off')
fig.add_subplot(1, 2, 1)
plt.imshow(scale_percentile(im_rgb[y1:y2, x1:x2]));
fig.add_subplot(1, 2, 2)
plt.imshow(scale_percentile(im_rgb[y1:y2, x1:x2]));


# In[11]:

get_ipython().system('ls input/sixteen_band/ | head')


# In[2]:

IM_ID = '6120_2_2'


# ## RGB

# In[67]:

img = tiff.imread('input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
print(img.shape)
imgs = [img[:, :, i] for i in range(3)]
plots(imgs, figsize=(10, 10*3), rows=3, cols=1, cmap="gray")


# In[80]:

img = tiff.imread('input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
print(img.shape)
imgs = [img[:, :, i] for i in range(3)]
imgs_scaled = [scale_percentile(np.expand_dims(img[:, :, i], -1))[:, :, 0] for i in range(3)]
imgs_unsqueezed = [[j, k] for j, k in zip(imgs, imgs_scaled)]
imgs = []
for pair in imgs_unsqueezed:
    imgs += pair
plots(imgs, figsize=(12, 20), rows=3, cols=2, cmap="gray")


# ## SWIR, 7.5 m, 1195 - 2365 nm

# In[48]:

img = tiff.imread('input/sixteen_band/{}_A.tif'.format(IM_ID)).transpose([1, 2, 0])
print(img.shape)
imgs = [img[:, :, i] for i in range(8)]
plots(imgs, figsize=(16, 8), rows=2, cols=4, cmap="gray")


# In[82]:

img = tiff.imread('input/sixteen_band/{}_A.tif'.format(IM_ID)).transpose([1, 2, 0])
print(img.shape)
imgs = [scale_percentile(np.expand_dims(img[:, :, i], -1))[:, :, 0] for i in range(8)]
plots(imgs, figsize=(16, 8), rows=2, cols=4, cmap="gray")


# ## Multispectral, 1.24 m, 400 - 1040 nm

# In[61]:

img = tiff.imread('input/sixteen_band/{}_M.tif'.format(IM_ID)).transpose([1, 2, 0])
print(img.shape)
imgs = [img[:, :, i] for i in range(8)]
plots(imgs, figsize=(16, 32), rows=4, cols=2, cmap="gray")


# In[4]:

img = tiff.imread('input/sixteen_band/{}_M.tif'.format(IM_ID)).transpose([1, 2, 0])
print(img.shape)
imgs = [scale_percentile(np.expand_dims(img[:, :, i], -1))[:, :, 0] for i in range(4)]
plots(imgs, figsize=(16, 32), rows=4, cols=2, cmap="gray")


# In[5]:

img = tiff.imread('input/sixteen_band/{}_M.tif'.format(IM_ID)).transpose([1, 2, 0])
print(img.shape)
imgs = [scale_percentile(np.expand_dims(img[:, :, i], -1))[:, :, 0] for i in range(4, 8)]
plots(imgs, figsize=(16, 32), rows=4, cols=2, cmap="gray")


# ## Panchromatic, 31 cm, 450 - 800 nm

# In[55]:

img = tiff.imread('input/sixteen_band/{}_P.tif'.format(IM_ID))
print(img.shape)
plots([img], figsize=(12, 12), rows=1, cols=1, cmap="gray")


# In[71]:

img = tiff.imread('input/sixteen_band/{}_P.tif'.format(IM_ID))
print(img.shape)
img = scale_percentile(np.expand_dims(img, -1))
print(img.shape)
img = img[:, :, 0]
print(img.shape)
plots([img], figsize=(12, 12), rows=1, cols=1, cmap="gray")


# # PyTorch

# In[1]:

from collections import defaultdict
import csv
import sys

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff
import pandas as pd

import torch
from torch import nn
from torch.autograd import Variable

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


csv.field_size_limit(sys.maxsize);


def get_scalers():
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


def mask_for_polygons(polygons):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

# Load data
# Define model
# Define loss function
# Define optimizer
# Use the optimizer to train the model with respect to the loss function
## Clear the gradients
## Get the graidents
## Update the weights
## Repeat


# In[2]:

# Load data
INPUT_DIR = "input/"
RGB_DIR = INPUT_DIR + "three_band/"
EXT = ".tif"
N = 10

train_wkt = pd.read_csv(INPUT_DIR + 'train_wkt_v4.csv')
all_img_IDs = list(set(train_wkt.ImageId))
np.random.seed(8675309)
np.random.shuffle(all_img_IDs)
img_IDs = all_img_IDs[:N]
img_IDs_val = all_img_IDs[N:]
filenames = [RGB_DIR + img_id + EXT for img_id in img_IDs]
filenames_val = [RGB_DIR + img_id + EXT for img_id in img_IDs_val]


# In[13]:

imgs_uint16 = [tiff.imread(f) for f in filenames]


# In[16]:

imgs_float32 = [np.float32(img) for img in imgs_uint16]


# In[17]:

len(imgs_float32)


# In[21]:

[img.shape for img in imgs_float32]


# - New problem: The images have different shapes.
#   - Approach 1: Zero padding
#   - Approach 2: Reflection padding
#   - Approach 3: Center cropping
#     - What exactly does this mean?
#   - Approach 4: Crop largest side to smallest side
# - Problem properties:
#   - How we preprocess the input is how we'll need to preprocess the output.
# - Others' approaches:
#   - Sergey Mushinskiy's: Use patches, e.g. of size 160 x 160
#   
# I want to process an entire image in a forward pass, so that the network sees more information when training. With patches, the network doesn't know which patches were near each other, and so if a building is split between two patches, the network won't know it was one building.
# 
# Response: Let the process-the-entire-image improvement hypothesis be something you test later. Get the basics going. The first goal is to assess the best solutions of this competition and see how far they have to go to be viable in real situations.

# In[185]:

X = Variable(torch.from_numpy(np.float32(filenames)))
X_val = Variable(torch.from_numpy(np.array(filenames_val)))


# In[ ]:

X = Variable(torch.)
Y = get_masks


# In[43]:

# Define model
model = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Softmax2d()
        )


# In[ ]:

# Define loss function
loss_fn = torch.nn.MSELoss()
# Define optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
# Use the optimizer to train the model with respect to the loss function
batch_size = 4
n_epochs = 1
n_batches = N // batch_size + 1
for i in range(n_epochs):
    for k in range(n_batches):
        start, end = k * batch_size, (k + 1) * batch_size
        
## Clear the gradients
## Get the graidents
## Update the weights
## Repeat


# # Playing with contours

# In[67]:

import numpy as np
import cv2

im = tiff.imread('input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])


# In[68]:

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
plots([imgray], cmap='gray', figsize=(12, 12))


# In[ ]:

mask = mask * 255 / mask.max()
print(mask.min(), mask.max())
mask[np.where(mask < thresh)] = 0
mask[np.where(mask >= thresh)] = 1
plots([mask], cmap='gray', figsize=(w, w))


# In[103]:

import numpy as np
import cv2

w = 12
thresh = 127
im = tiff.imread('input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
mask2 = mask / mask.max() * 255
mask2.max()


# In[123]:

def get_mask(im, thresh, w=12):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgray = imgray / imgray.max() * 255
    mask[np.where(imgray < thresh)] = 1
    mask[np.where(imgray >= thresh)] = 0
    plots([mask], cmap='gray', figsize=(w, w))
    return mask


# In[118]:

mask = get_mask(im, thresh=127/2)


# In[119]:

mask = get_mask(im, thresh=127)


# In[146]:

import numpy as np
import cv2

im = tiff.imread('input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
imgray = cv2.cvtColor(np.float32(scale_percentile(im)), cv2.COLOR_BGR2GRAY)
imgray = imgray / imgray.max() * 255
plt.hist(imgray.reshape(np.prod(imgray.shape)))


# In[147]:

plots([imgray], cmap='gray')


# In[148]:

plots([get_mask(imgray)], cmap='gray')


# In[152]:

def get_mask_scaled(im, thresh, w=12):
    imgray = cv2.cvtColor(np.float32(scale_percentile(im)), cv2.COLOR_BGR2GRAY)
    imgray = imgray / imgray.max() * 255
    mask[np.where(imgray < thresh)] = 1
    mask[np.where(imgray >= thresh)] = 0
    plots([mask], cmap='gray', figsize=(w, w))
    return mask

im = tiff.imread('input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
mask = get_mask_scaled(im, 127)


# In[153]:

mask = get_mask(im, 127)


# In[154]:

mask = get_mask(im, 127/2)


# In[178]:

imgray = cv2.cvtColor(np.float32(scale_percentile(im)), cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(np.float32(imgray), 0.5, 1, 0)
_, contours, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# In[191]:

plots([scale_percentile(im), imgray, mask], rows=2, cols=2, cmap=[None]+['gray']*2, figsize=(16,16))


# In[215]:

h, w, d = im.shape
zeros = np.zeros((h, w))
res = cv2.drawContours(scale_percentile(im), contours, -1, (255, 255, 255), 3)
plots(res, figsize=(16, 16))


# # Contours

# In[75]:

get_ipython().magic('matplotlib inline')

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile as tiff

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
        
def scale_and_clip(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


# In[83]:

IM_ID = '6120_2_2'
im = tiff.imread('input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])

plt.hist(im.reshape(im.size)); plt.show()
tiff.imshow(im); plt.axis('off'); plt.show()


# In[77]:

im_scaled = scale_and_clip(im)

plt.hist(im_scaled.reshape(im_scaled.size)); plt.show()
plots(im_scaled)


# In[78]:

imgray = cv2.cvtColor(np.float32(im), cv2.COLOR_BGR2GRAY)

plt.hist(im.reshape(im.size)); plt.show()
plots(imgray, cmap='gray')


# In[79]:

imgray_scaled = cv2.cvtColor(np.float32(im_scaled), cv2.COLOR_BGR2GRAY)

plt.hist(im_scaled.reshape(im_scaled.size)); plt.show()
plots(imgray_scaled, cmap='gray')


# In[80]:

_, mask = cv2.threshold(np.float32(imgray), int(np.mean([imgray.max(), imgray.min()])), int(imgray.max()), int(imgray.min()))

plt.hist(im.reshape(im.size)); plt.show()
plots(mask, cmap='gray')


# In[81]:

_, mask = cv2.threshold(np.float32(imgray_scaled), 0.5, 1, 0)
plt.hist(im_scaled.reshape(im_scaled.size)); plt.show()
plots(mask, cmap='gray')


# In[117]:

_, contours, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
result = cv2.drawContours(np.zeros(im.shape), contours, -1, (1, 1, 1), 2)

plots((result - result.min()) / result.max())


# In[119]:

result = cv2.drawContours(np.copy(im_scaled), contours, -1, (0, 0, 1), 2)
plots((result - result.min()) / result.max())

