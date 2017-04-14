
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
# │   │   │   ├── 6010_1_2.tif
# │   │   │   ├── 6010_1_2_A.tif
# │   │   │   ├── 6010_1_2_M.tif
# │   │   │   ├── 6010_1_2_P.tif
# │   │   │   └── ...
# │   │   ├── targets
# │   │   │   ├── 6010_1_2.png
# │   │   │   ├── 6010_1_2_A.png
# │   │   │   ├── 6010_1_2_M.png
# │   │   │   ├── 6010_1_2_P.png
# │   │   │   └── ...
# │   │   └── test
# │   └── tiles
# │       └── 224x224
# │           └── stride_full
# │               ├── inputs
# │               │   ├── train
# │               │   ├── train_all
# │               │   └── valid
# │               ├── targets
# │               │   ├── masks_train
# │               │   ├── masks_train_all
# │               │   └── masks_valid
# │               ├── test
# │               └── results
# │                   ├── kaggle_submissions
# │                   ├── masks_test_predictions
# │                   ├── masks_train_predictions
# │                   └── masks_valid_predictions
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

# ## Dependencies


from glob import glob

import pandas as pd
import numpy as np
import tifffile as tiff
from PIL import Image
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## Globals


ROOT = '../'
INPUTS_BIG = ROOT + 'data/big/inputs/'
TARGETS_BIG = ROOT + 'data/big/targets/'
INPUTS_TILES = ROOT + 'data/tiles/224x224/stride_full/inputs/train_all/'
TARGETS_TILES = ROOT + 'data/tiles/224x224/stride_full/targets/masks_train_all/'
WKT_PATH = ROOT + 'data/meta/train_wkt_v4.csv'
GRID_PATH = ROOT + 'data/meta/grid_sizes.csv'
TILE_LEN = 224
IM_ID = '6120_2_0'
CLASS_ID = 1  # Buildings

GRID_COLOR = (0, 0.8, 0)
GRID_THICK = 10

WKTS = pd.read_csv(WKT_PATH)
GRIDS = pd.read_csv(GRID_PATH, names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


# ## Functions


###################
## Visualization ##
###################

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

##############################################################################

def id2im(im_id):
    path = INPUTS_BIG + im_id + '.tif'
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
    cv2.fillPoly(mask, get_exteriors(polygons), 255)
    cv2.fillPoly(mask, get_interiors(polygons), 0)  # This line does nothing?
    return mask


def saveim(im, ext, folder, im_id, i=''):
    if i != '': i = '_%03d' % i  # "001.png" instead of "1.png"
    path = folder + im_id + i + ext
    if ext == '.png':
        Image.fromarray(im).save(path)
    elif ext == '.tif':
        tiff.imsave(path, im)
    else:
        raise Exception('Unsupported file type')


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
mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2RGB)/255
for r, c in mask_anchor_points:
    r2, c2 = (r+TILE_LEN, c+TILE_LEN)
    cv2.rectangle(mask2, (c, r), (c2, r2), GRID_COLOR, GRID_THICK)
plot(mask2)



mask_tiles = [mask[r:r+TILE_LEN, c:c+TILE_LEN] for r, c in mask_anchor_points]

print(len(mask_tiles))
plot(mask_tiles[:tiles_per_row], f=(tiles_per_row, 1), r=1, c=tiles_per_row)


# ## Save


saveim(mask, '.png', TARGETS_BIG, IM_ID)

for i, mask_tile in enumerate(mask_tiles):
    saveim(mask_tile, '.png', TARGETS_TILES, IM_ID, i)
    
for i, tile in enumerate(tiles):
    saveim(tile, '.tif', INPUTS_TILES, IM_ID, i)

