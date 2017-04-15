
# coding: utf-8

# # Data wrangling

# ## Directory tree

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
# │               │   └── valid
# │               ├── targets
# │               │   ├── train
# │               │   └── valid
# │               ├── indices
# │               ├── test
# │               └── results
# │                   ├── kaggle_submissions
# │                   ├── test
# │                   ├── train
# │                   └── valid
# ├── lib
# │   └── weights
# └── nbs
#     └── this notebook
# ```
# 
# ---
# TODO: Automate the conversion from the kaggle directory tree to this directory tree.

# ## Dependencies


from glob import glob
from time import time

import pandas as pd
import numpy as np
import tifffile as tiff
from PIL import Image
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity


# ## Globals


##############################
## Globals you might change ##
##############################

ROOT = '../'
TILE_LEN = 224
CLASS_ID = 1  # Buildings
VALID_PORTION = 0.20

STRIDE = 'full'  # 'full', 'half', 'fourth'  # only 'full' works right now

#######################################
## Globals you probably won't change ##
#######################################

DATA = ROOT + 'data/'
BIG = DATA + 'big/'
TILES = DATA + 'tiles/{0}x{0}/stride_{1}/'.format(TILE_LEN, STRIDE)
META = DATA + 'meta/'
INPUTS_BIG = BIG + 'inputs/'
TARGETS_BIG = BIG + 'targets/'
INPUTS_TILES = TILES + 'inputs/'
TARGETS_TILES = TILES + 'targets/'
INDICES_TILES = TILES + 'indices/'
INPUTS_TILES_TRN = INPUTS_TILES + 'train/'
TARGETS_TILES_TRN = TARGETS_TILES + 'train/'
INPUTS_TILES_VAL = INPUTS_TILES + 'valid/'
TARGETS_TILES_VAL = TARGETS_TILES + 'valid/'
WKT_PATH = META + 'train_wkt_v4.csv'
GRID_PATH = META + 'grid_sizes.csv'

WKTS = pd.read_csv(WKT_PATH)
GRIDS = pd.read_csv(GRID_PATH, names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


# ## Functions


def loadpng(path):
    return np.array(Image.open(path))


def loadtif(path, roll=False, unit_norm=False):
    im = tiff.imread(path)
    if roll:
        im = np.rollaxis(im, 0, 3)  # Channels last for tf and plt
    if unit_norm:
        im = (im - im.min()) / (im.max() - im.min())  # min: 0.0, max: 1.0
    return im


def id2im(im_id):
    path = INPUTS_BIG + im_id + '.tif'
    im = loadtif(path, roll=True, unit_norm=True)
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
        

def savetiles(tiles, ext, folder, im_id):
    for i, tile in enumerate(tiles):
        saveim(tile, ext, folder, im_id, i)


# ## Pipeline


start_time = time()
image_IDs = list(set(WKTS.ImageId))
assert len(image_IDs) == 25
for i, im_id in enumerate(image_IDs):
    start_time_i = time()
    # Load the image
    im = id2im(im_id)
    h, w = im.shape[:2]

    assert im.dtype == np.float64
    assert im.ndim == 3
    assert h > 3000
    assert w > 3000
    assert im.shape[-1] == 3
    assert im.min() == 0.0
    assert im.max() == 1.0

    # Get mask
    polygons = get_polygons(im_id, CLASS_ID)
    mask = polygons2mask(polygons, h, w)

    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    assert mask.shape[0] > 3000
    assert mask.shape[1] > 3000
    assert mask.min() == 0
    assert mask.max() == 255 or mask.max() == 0

    saveim(mask, '.png', TARGETS_BIG, im_id)

    # Get tiles
    tiles_per_col = h // TILE_LEN
    tiles_per_row = w // TILE_LEN

    anchor_points = [(r*TILE_LEN, c*TILE_LEN)
                     for r in range(tiles_per_col)
                     for c in range(tiles_per_row)]

    tiles = [im[r:r+TILE_LEN, c:c+TILE_LEN]
             for r, c in anchor_points]

    mask_anchor_points = [(r*TILE_LEN, c*TILE_LEN)
                          for r in range(tiles_per_col)
                          for c in range(tiles_per_row)]

    mask_tiles = [mask[r:r+TILE_LEN, c:c+TILE_LEN]
                  for r, c in mask_anchor_points]

    # Split tiles
    n_tiles = len(tiles)
    indices = range(n_tiles)
    n_val = int(n_tiles * VALID_PORTION)
    indices_val = sorted(np.random.choice(indices, n_val, replace=False))
    indices_trn = [i for i in indices if i not in indices_val]

    assert len(indices_trn) + len(indices_val) == len(indices)
    assert len(indices_val) == int(len(indices) * VALID_PORTION)

    tiles_trn = [tiles[i] for i in indices_trn]
    mask_tiles_trn = [mask_tiles[i] for i in indices_trn]
    tiles_val = [tiles[i] for i in indices_val]
    mask_tiles_val = [mask_tiles[i] for i in indices_val]
    np.save(INDICES_TILES+'%s_trn.npy' % im_id, indices_trn)
    np.save(INDICES_TILES+'%s_val.npy' % im_id, indices_val)

    savetiles(tiles_trn, '.tif', INPUTS_TILES_TRN, im_id)
    savetiles(mask_tiles_trn, '.png', TARGETS_TILES_TRN, im_id)
    savetiles(tiles_val, '.tif', INPUTS_TILES_VAL, im_id)
    savetiles(mask_tiles_val, '.png', TARGETS_TILES_VAL, im_id)
    print("Time this loop (%s): %s" % (i, (time() - start_time_i)))
print("Total time: %s" % (time() - start_time))



# Test a set of files
imgs = [loadtif(INPUTS_BIG+im_id+'.tif', roll=True, unit_norm=True),
        loadtif(INPUTS_TILES_TRN+im_id+'_000.tif'),
        loadpng(TARGETS_BIG+im_id+'.png'),
        loadpng(TARGETS_TILES_TRN+im_id+'_000.png')]

assert np.array_equal(imgs[0][0][0], imgs[1][0][0])  # Top-left pixel
assert np.array_equal(imgs[2][0][0], imgs[3][0][0])  # Top-left pixel

