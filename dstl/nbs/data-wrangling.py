
# coding: utf-8

# # Data wrangling

# ## Readme

# ```
# .
# ├── data
# │   ├── meta
# │   │   ├── train_wkt_v4.csv
# │   │   ├── grid_sizes.csv
# │   │   └── sample_submission.csv
# │   ├── big
# │   │   ├── inputs
# │   │   │   ├── test
# │   │   │   └── train_all
# │   │   │       ├── 6010_1_2.tif
# │   │   │       ├── 6010_1_2_A.tif
# │   │   │       ├── 6010_1_2_M.tif
# │   │   │       ├── 6010_1_2_P.tif
# │   │   │       └── ...
# │   │   └── targets
# │   │       └── masks_train_all
# │   └── tiles
# │       └── 224x224
# │           └── stride_full
# │               ├── inputs
# │               │   ├── test
# │               │   ├── train
# │               │   ├── train_all
# │               │   └── valid
# │               ├── results
# │               │   ├── kaggle_submissions
# │               │   ├── masks_test_predictions
# │               │   ├── masks_train_predictions
# │               │   └── masks_valid_predictions
# │               └── targets
# │                   ├── masks_train
# │                   ├── masks_train_all
# │                   └── masks_valid
# ├── lib
# │   └── weights
# └── nbs
#     └── this notebook
# ```
# 
# ---
# 
# 
# See **kaggle-data-to-our-format--incomplete-but-useful.py** for hints on setting up our data folder format (i.e. putting A, M, P, and RGB in the same folder). TODO: Automate this for new users.

# ## Globals


ROOT = '../'
TRAIN_BIG = ROOT + 'data/big/inputs/train_all/'
WKT_PATH = ROOT + 'data/meta/train_wkt_v4.csv'
GRID_PATH = ROOT + 'data/meta/grid_sizes.csv'
TILE_LEN = 224
IM_ID = '6120_2_0'
CLASS_ID = 1  # Buildings

GRID_COLOR = (0, 0.8, 0)
GRID_THICK = 10

WKTS = pd.read_csv(WKT_PATH)
GRIDS = pd.read_csv(GRID_PATH, names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


# ## Dependencies


from glob import glob

import pandas as pd
import tifffile as tiff
import numpy as np
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


def plots(imgs, figsize=(12, 12), rows=1, cols=1,
          interp=None, titles=None, cmap='gray'):
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(cmap, list):
        if imgs[0].ndim == 2:
            cmap = 'gray'
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


def plot(im, f=6, r=1, c=1):
    fs = f if isinstance(f, tuple) else (f, f)
    plots(im, figsize=fs, rows=r, cols=c)


def id2im(im_id):
    path = TRAIN_BIG+im_id+'.tif'
    im = tiff.imread(path)
    im = np.rollaxis(im, 0, 3)  # Channels last for tf and plt
    im = (im - im.min()) / (im.max() - im.min())
    return im


def load_grid_size(im_id):
    x_max, y_min = GRIDS[GRIDS.ImageId == im_id].iloc[0, 1:]
    return x_max, y_min


def load_polygons(im_id, class_id):
    """Loads a wkt and converts it to a Shapely.MultiPolygon."""
    wkt_row = WKTS[WKTS.ImageId == im_id]
    wkt = wkt_row[wkt_row.ClassType == class_id].MultipolygonWKT.values[0]
    polygons = shapely.wkt.loads(wkt)
    return polygons


def get_scalers(im_id, h, w):
    x_max, y_min = load_grid_size(im_id)
    h2 = h * (h / (h + 1))
    w2 = w * (w / (w + 1))
    return w2 / x_max, h2 / y_min


def scale_polygons(mp, x, y):
    return shapely.affinity.scale(mp, xfact=x, yfact=y, origin=(0, 0, 0))


def get_polygons(im_id, class_id):
    """Returns a scaled Shapely.MultiPolygon."""
    polygons = load_polygons(im_id, class_id)
    x_scaler, y_scaler = get_scalers(im_id, h, w)
    polygons_scaled = scale_polygons(polygons, x_scaler, y_scaler)
    return polygons_scaled


def get_int_coords(x):
    return np.array(x).round().astype(np.int32)


def get_exteriors(polygons):
    return [get_int_coords(poly.exterior.coords) for poly in polygons]


def get_interiors(polygons):
    return [get_int_coords(pi.coords) for poly in polygons
            for pi in poly.interiors]


def polygons2mask(polygons, h, w):
    mask = np.zeros((h, w), np.uint8)
    if not polygons: return mask
    cv2.fillPoly(mask, get_exteriors(polygons), 1)
    cv2.fillPoly(mask, get_interiors(polygons), 0)  # This line seems to do nothing.
    return mask


def look_good(matrices):
    def _scale_percentile(matrix):
        w, h, d = matrix.shape
        matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
        # Get 2nd and 98th percentile
        mins = np.percentile(matrix, 1, axis=0)
        maxs = np.percentile(matrix, 99, axis=0) - mins
        matrix = (matrix - mins[None, :]) / maxs[None, :]
        matrix = np.reshape(matrix, [w, h, d])
        matrix = matrix.clip(0, 1)
        return matrix
    if not isinstance(matrices, list):
        return _scale_percentile(matrices)
    return [_scale_percentile(m) for m in matrices]


# ## Load image


im = id2im(IM_ID)

plot(im)



h, w = im.shape[:2]

print(h, w)
print(im.min(), im.max())


# ## Create masks


polygons = get_polygons(IM_ID, CLASS_ID)
mask = polygons2mask(polygons, h, w)

plot([im, mask], c=2, f=16)


# ## Create tiles

# ### Create tiles for image

# - Load the image
# - Count the number of tiles per image dimension, ignore the remainder
# - Get the anchor point of each tile (i.e. the top-left corner of the tile)
# - Repeat for all images


tiles_per_col = h // TILE_LEN
tiles_per_row = w // TILE_LEN

print(tiles_per_col, tiles_per_row)



anchor_points = [(r*TILE_LEN, c*TILE_LEN) for r in range(tiles_per_col)
                                          for c in range(tiles_per_row)]

im2 = im.copy()
for r, c in anchor_points:
    r2, c2 = (r+TILE_LEN, c+TILE_LEN)
    cv2.rectangle(im2, (c, r), (c2, r2), GRID_COLOR, GRID_THICK)
plot(im2)



tiles = [im[r:r+TILE_LEN, c:c+TILE_LEN] for r, c in anchor_points]

print(len(tiles))
plot(tiles[:tiles_per_row], f=(tiles_per_row, 1), r=1, c=tiles_per_row)


# ### Create tiles for mask


mask_anchor_points = [(r*TILE_LEN, c*TILE_LEN) for r in range(tiles_per_col)
                                               for c in range(tiles_per_row)]

mask2 = mask.copy()
mask2 = cv2.cvtColor(mask2*255, cv2.COLOR_GRAY2RGB)/255
for r, c in mask_anchor_points:
    r2, c2 = (r+TILE_LEN, c+TILE_LEN)
    cv2.rectangle(mask2, (c, r), (c2, r2), GRID_COLOR, GRID_THICK)
plot(mask2)



mask_tiles = [mask[r:r+TILE_LEN, c:c+TILE_LEN] for r, c in mask_anchor_points]

print(len(mask_tiles))
plot(mask_tiles[:tiles_per_row], f=(tiles_per_row, 1), r=1, c=tiles_per_row)


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

