
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


INPUT_DIR = 'input/'



filenames3 = glob(INPUT_DIR+'three_band/*.tif')
filenames16 = glob(INPUT_DIR+'sixteen_band/*.tif')
wkt_csv = pd.read_csv(INPUT_DIR+'train_wkt_v4.csv')

print(len(filenames3))  # 450 - 425 test == 25 train
print(filenames3[0])
print(len(filenames16)) # 1350 / 3 image types == 450
print(filenames16[0])
print(len(wkt_csv))  # 250 / 10 classes == 25 train
wkt_csv.head()



train_IDs = list(set(wkt_csv.ImageId))

print(len(train_IDs))
print(train_IDs[0])



filenames3_IDs = [f.split('/')[-1].split('.')[0] for f in filenames3]
train_filenames3 = [f for f, ID in zip(filenames3, filenames3_IDs) if ID in train_IDs]

print(len(filenames3_IDs))
print(len(train_filenames3))



filenames16_IDs = [f.split('/')[-1].split('.')[0][:-2] for f in filenames16]
train_filenames16 = [f for f, ID in zip(filenames16, filenames16_IDs) if ID in train_IDs]

print(len(filenames16_IDs))
print(len(train_filenames16))  # 75 / 3 image types == 25 train



train_filenames[0].split('/')[-1]



train_filenames[-1]



train_filenames = train_filenames3 + train_filenames16

print(len(train_filenames))



for f in train_filenames: shutil.move(f, 'train_all/inputs/'+f.split('/')[-1])

print(len(glob('train_all/inputs/*.tif')))



sorted(glob('train_all/inputs/*.tif'))[:8]



print(len(filenames3))
print(len(filenames16))

filenames3 = glob(INPUT_DIR+'three_band/*.tif')
filenames16 = glob(INPUT_DIR+'sixteen_band/*.tif')

print(len(filenames3))
print(len(filenames16))



for f in filenames3+filenames16: shutil.move(f, 'test/'+f.split('/')[-1])

print(len(glob('test/*.tif')))
sorted(glob('test/*.tif'))[:8]


# ## Split train_all into train and valid

# We have 25 satellite images.
# 
# How should we split them into a train set and a validation set?
# 
# We could put 20% (i.e. 5 images) in the validation set.
# 
# However, it seems too possible that we could choose five images that don't contain buildings.
# 
# We could do what Kagglers did, and split the 25 images into subimages, and make our validation set from there; this way the validation set would run through all the images.
# 
# One mistake the Kagglers made was to train on their validation set. They didn't keep track of which subimages they used as validation. They merely randomly sampled 3,000 subimages for their validation set and 10,000 subimages for their training set.
# 
# 


IM_ID = '6120_2_0'
PATCH_LEN = 200


# ### Load image


im = tiff.imread('train_all/inputs/{}.tif'.format(IM_ID))
im = np.rollaxis(im, 0, 3)
im = (im - im.min()) / (im.max() - im.min())

plots(im, figsize=(6, 6))



im2 = im.copy()
n_patches = 4
cv2.rectangle(im2, (10, 15), (n_patches*PATCH_LEN, PATCH_LEN), (1, 0, 0), 10)
plots(im2, figsize=(6, 6))



imgs = [im[:PATCH_LEN, k*PATCH_LEN:(k+1)*PATCH_LEN, :] for k in range(n_patches)]

plots(imgs, cols=n_patches, figsize=(16, 4))


# ### Create validation set


FRACTION_VALID = 0.20
N_BARS = 4


# #### Naive validation sampling


h, w = im.shape[:2]
r_max = h - PATCH_LEN
c_max = w - PATCH_LEN

print(h, w)
print(r_max)
print(r_max)

im2 = im.copy()
cv2.rectangle(im2, (10, 15), (c_max, r_max), (1, 0, 0), 10)
plots(im2, figsize=(6, 6))



im3 = im2.copy()
patches = []
for _ in range(14*4):
    r = np.random.randint(0, r_max)
    c = np.random.randint(0, c_max)
    r2, c2 = r+PATCH_LEN, c+PATCH_LEN
    patches.append(im[r:r2, c:c2])
    cv2.rectangle(im3, (c, r), (c2, r2), (0, 1, 0), 10)
plots(im3, figsize=(6, 6))


# #### Create horizontal bars


h, w = im.shape[:2]
image_area = h * w
patches_per_row = int(w / PATCH_LEN)
patches_per_col = int(h / PATCH_LEN)
bar_area = image_area * FRACTION_VALID / N_BARS
patch_area = PATCH_LEN**2
patches_per_bar = int(bar_area / patch_area) - 1

assert patches_per_row >= patches_per_bar
assert patches_per_col >= patches_per_bar

print(h, w)
print(image_area)
print(patches_per_row)
print(patches_per_col)
print(bar_area)
print(patch_area)
print(patches_per_bar)



r_min_1 = PATCH_LEN*3
r_max_1 = h - PATCH_LEN*4
r_min_2 = r_min_1 - PATCH_LEN*2
r_max_2 = r_max_1 + PATCH_LEN*2
c_min = PATCH_LEN
c_max = c_min + PATCH_LEN

print(r_min_1, r_max_1)
print(r_min_2, r_max_2)
print(c_min, c_max)

im4 = im.copy()
cv2.rectangle(im4, (c_min, r_min_2), (c_max, r_max_2), (1, 1, 0), 10)
cv2.rectangle(im4, (c_min, r_min_1), (c_max, r_max_1), (1, 1, 1), 10)
plots(im4, figsize=(6, 6))



im5 = im4.copy()
bar_anchors = []
for _ in range(N_BARS//2):
    if bar_anchors:  # Only works with two bars right now
        r_bar = bar_anchors[0][0]
        r_above = np.random.randint(r_min_2, r_bar-2*PATCH_LEN)
        r_below = np.random.randint(r_bar+2*PATCH_LEN, r_max_2)
        r = np.random.choice([r_above, r_below])
    else:
        r = np.random.randint(r_min_1, r_max_1)
    c = np.random.randint(c_min, c_max)
    r2, c2 = r+PATCH_LEN, c+PATCH_LEN*patches_per_bar
    bar_anchors.append((r, c))
    cv2.rectangle(im5, (c, r), (c2, r2), (1, 0, 0), 10)
plots(im5, figsize=(6, 6))



im6 = im5.copy()
im10 = im.copy()
im10_full = im.copy()
patches_val = []
for r_bar, c_bar in bar_anchors:
    for i in range(patches_per_bar):
        c = c_bar + i*PATCH_LEN
        r2, c2 = r_bar+PATCH_LEN, c+PATCH_LEN
        patches_val.append(im[r_bar:r2, c:c2])
        cv2.rectangle(im6, (c, r_bar), (c2, r2), (0, 1, 0), 10)
        cv2.rectangle(im10, (c, r_bar), (c2, r2), (0, 1, 0), 10)
        cv2.rectangle(im10_full, (c, r_bar), (c2, r2), (0, 1, 0), -1)
plots(im6, figsize=(6, 6))



plots(patches_val, rows=N_BARS//2, cols=patches_per_bar, figsize=(15, N_BARS//2))


# #### Create vertical bars


c_min_1 = PATCH_LEN*3
c_max_1 = w - PATCH_LEN*4
c_min_2 = c_min_1 - PATCH_LEN*2
c_max_2 = c_max_1 + PATCH_LEN*2
r_min = PATCH_LEN
r_max = r_min + PATCH_LEN

print(c_min_1, c_max_1)
print(c_min_2, c_max_2)
print(r_min, r_max)

im7 = im6.copy()
cv2.rectangle(im7, (c_min_2, r_min), (c_max_2, r_max), (1, 1, 0), 10)
cv2.rectangle(im7, (c_min_1, r_min), (c_max_1, r_max), (1, 1, 1), 10)
plots(im7, figsize=(6, 6))



im8 = im7.copy()
for _ in range(N_BARS//2):
    if bar_anchors[2:]:  # Only works with two bars right now
        c_bar = bar_anchors[2:][0][1]
        c_left_of = np.random.randint(c_min_2, c_bar-2*PATCH_LEN)
        c_right_of = np.random.randint(c_bar+2*PATCH_LEN, c_max_2)
        c = np.random.choice([c_left_of, c_right_of])
    else:
        c = np.random.randint(c_min_1, c_max_1)
    r = np.random.randint(r_min, r_max)
    c2, r2 = c+PATCH_LEN, r+PATCH_LEN*(patches_per_bar)
    bar_anchors.append((r, c))
    cv2.rectangle(im8, (c, r), (c2, r2), (1, 0, 0), 10)
plots(im8, figsize=(6, 6))



im9 = im8.copy()
for r_bar, c_bar in bar_anchors[2:]:
    for i in range(patches_per_bar):
        r_i = r_bar + i*PATCH_LEN
        r2, c2 = r_i+PATCH_LEN, c_bar+PATCH_LEN
        patches_val.append(im[r_i:r2, c_bar:c2])
        cv2.rectangle(im9, (c_bar, r_i), (c2, r2), (0, 1, 0), 10)
        cv2.rectangle(im10, (c_bar, r_i), (c2, r2), (0, 1, 0), 10)
        cv2.rectangle(im10_full, (c_bar, r_i), (c2, r2), (0, 1, 0), -1)
plots(im9, figsize=(6, 6))


# #### Validation patches


plots(im10, figsize=(6, 6))



plots(patches_val, rows=N_BARS, cols=patches_per_bar, figsize=(15, N_BARS))



plots(patches_val[7], figsize=(6, 6))


# Is 200 x 200 a big enough patch size for images of this spatial resolution?

# ### Create training set


plots(im10, figsize=(6, 6))


# #### Calculate unallowed ranges for training patch coordinates

# ##### Start with one validation bar


bar_anchor = bar_anchors[1]

c_min = bar_anchor[1]
c_max = c_min + PATCH_LEN*patches_per_bar

r_min = bar_anchor[0]
r_max = bar_anchor[0] + PATCH_LEN

print(bar_anchor)
print(patches_per_bar)
print(c_min, c_max)
print(r_min, r_max)



im11_red = im10.copy()
im11 = im10.copy()
for _ in range(300):
    # sample
    r = np.random.randint(0, h - PATCH_LEN)
    c = np.random.randint(0, w - PATCH_LEN)
    patch_corners = [(r, c), (r, c+PATCH_LEN), (r+PATCH_LEN, c), (r+PATCH_LEN, c+PATCH_LEN)]

    # test
    accepted = True
    for r, c in patch_corners:
        if in_interval(r, r_min, r_max) and in_interval(c, c_min, c_max):
            #print(r, r_min, r_max)
            #print(c, c_min, c_max)
            accepted = False
            break
    string = "Accepted" if accepted else "Rejected"
    if accepted:
        cv2.rectangle(im11, patch_corners[0][::-1], patch_corners[-1][::-1], (0, 1, 1), 10)
        cv2.rectangle(im11_red, patch_corners[0][::-1], patch_corners[-1][::-1], (0, 1, 1), 10)
    else:
        pass
        cv2.rectangle(im11_red, patch_corners[0][::-1], patch_corners[-1][::-1], (1, 0, 0), 10)
plots(im11_red, figsize=(6, 6))



plots(im11, figsize=(6, 6))


# ##### Calculate it for all validation bars


x = [(1, 2), (3, 4)]
for i, p in enumerate(x):
    print(i, p)



x = [(1, 2), (3, 4)]
for i, (a, b) in enumerate(x):
    print(i, a, b)



im12_red = im10.copy()
im12 = im10.copy()
c_intervals = []
r_intervals = []
for i, (r_min, c_min) in enumerate(bar_anchors):
    # define the intervals
    if i < 2:
        r_max = r_min + PATCH_LEN
        c_max = c_min + PATCH_LEN*patches_per_bar
    else:
        r_max = r_min + PATCH_LEN*patches_per_bar
        c_max = c_min + PATCH_LEN
    r_intervals.append((r_min, r_max))
    c_intervals.append((c_min, c_max))

for _ in range(300):
    # sample
    r = np.random.randint(0, h - PATCH_LEN)
    c = np.random.randint(0, w - PATCH_LEN)
    patch_corners = [(r, c), (r, c+PATCH_LEN), (r+PATCH_LEN, c), (r+PATCH_LEN, c+PATCH_LEN)]

    # test
    accepted = True
    for r, c in patch_corners:
        for (r_min, r_max), (c_min, c_max) in zip(r_intervals, c_intervals):
            if in_interval(r, r_min, r_max) and in_interval(c, c_min, c_max):
                accepted = False
                break
        if not accepted:
            break
    string = "Accepted" if accepted else "Rejected"
    if accepted:
        cv2.rectangle(im12, patch_corners[0][::-1], patch_corners[-1][::-1], (0, 1, 1), 10)
        cv2.rectangle(im12_red, patch_corners[0][::-1], patch_corners[-1][::-1], (0, 1, 1), 10)
    else:
        pass
        cv2.rectangle(im12_red, patch_corners[0][::-1], patch_corners[-1][::-1], (1, 0, 0), 10)
plots(im12_red, figsize=(6, 6))



plots(im12, figsize=(6, 6))



im12_red = im10.copy()
im12 = im10.copy()
c_intervals = []
r_intervals = []
for i, (r_min, c_min) in enumerate(bar_anchors):
    # define the intervals
    if i < 2:
        r_max = r_min + PATCH_LEN
        c_max = c_min + PATCH_LEN*patches_per_bar
    else:
        r_max = r_min + PATCH_LEN*patches_per_bar
        c_max = c_min + PATCH_LEN
    r_intervals.append((r_min, r_max))
    c_intervals.append((c_min, c_max))

for _ in range(400):
    # sample
    r = np.random.randint(0, h - PATCH_LEN)
    c = np.random.randint(0, w - PATCH_LEN)
    patch_corners = [(r, c), (r, c+PATCH_LEN), (r+PATCH_LEN, c), (r+PATCH_LEN, c+PATCH_LEN)]

    # test
    accepted = True
    for r, c in patch_corners:
        for (r_min, r_max), (c_min, c_max) in zip(r_intervals, c_intervals):
            if in_interval(r, r_min, r_max) and in_interval(c, c_min, c_max):
                accepted = False
                break
        if not accepted:
            break
    string = "Accepted" if accepted else "Rejected"
    if accepted:
        cv2.rectangle(im12, patch_corners[0][::-1], patch_corners[-1][::-1], (0, 1, 1), 10)
        cv2.rectangle(im12_red, patch_corners[0][::-1], patch_corners[-1][::-1], (0, 1, 1), 10)
    else:
        pass
        cv2.rectangle(im12_red, patch_corners[0][::-1], patch_corners[-1][::-1], (1, 0, 0), 10)
plots(im12_red, figsize=(6, 6))



plots(im12, figsize=(6, 6))



im12_full = im10_full.copy()
im12_red = im10.copy()
im12 = im10.copy()
c_intervals = []
r_intervals = []
for i, (r_min, c_min) in enumerate(bar_anchors):
    # define the intervals
    if i < 2:
        r_max = r_min + PATCH_LEN
        c_max = c_min + PATCH_LEN*patches_per_bar
    else:
        r_max = r_min + PATCH_LEN*patches_per_bar
        c_max = c_min + PATCH_LEN
    r_intervals.append((r_min, r_max))
    c_intervals.append((c_min, c_max))

for _ in range(1000):
    # sample
    r = np.random.randint(0, h - PATCH_LEN)
    c = np.random.randint(0, w - PATCH_LEN)
    patch_corners = [(r, c), (r, c+PATCH_LEN), (r+PATCH_LEN, c), (r+PATCH_LEN, c+PATCH_LEN)]

    # test
    accepted = True
    for r, c in patch_corners:
        for (r_min, r_max), (c_min, c_max) in zip(r_intervals, c_intervals):
            if in_interval(r, r_min, r_max) and in_interval(c, c_min, c_max):
                accepted = False
                break
        if not accepted:
            break
    string = "Accepted" if accepted else "Rejected"
    if accepted:
        cv2.rectangle(im12, patch_corners[0][::-1], patch_corners[-1][::-1], (0, 1, 1), 10)
        cv2.rectangle(im12_full, patch_corners[0][::-1], patch_corners[-1][::-1], (0, 1, 1), -1)
        cv2.rectangle(im12_red, patch_corners[0][::-1], patch_corners[-1][::-1], (0, 1, 1), 10)
    else:
        pass
        cv2.rectangle(im12_red, patch_corners[0][::-1], patch_corners[-1][::-1], (1, 0, 0), 10)
plots(im12_red, figsize=(6, 6))



plots(im12, figsize=(6, 6))



plots(im12_full, figsize=(6, 6))

