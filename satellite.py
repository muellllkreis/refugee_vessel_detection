import rasterio
from rasterio.plot import show
from rasterio import plot
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from shapely.geometry import Polygon
from shapely import speedups
import rasterio.features
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import fiona
from scipy import ndimage, misc, signal

import pprint

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

downscale_factor = 1/2
speedups.disable()

## TCI downscaled to 20m ppx
with rasterio.open("../tci.jp2", tiled=True, blockxsize=256, blockysize=256, num_threads='all_cpus') as dataset:
    windows_d1 = []
    windows_d1.append(dataset.read(window=Window(2500, 3000, 2500, 1500),
                            out_shape=(
                                        dataset.count,
                                        int(1500 * downscale_factor),
                                        int(2500 * downscale_factor)
                                       ),
                            resampling=Resampling.bilinear))
    windows_d1.append(dataset.read(window=Window(6500, 7000, 2500, 1500),
                            out_shape=(
                                        dataset.count,
                                        int(1500 * downscale_factor),
                                        int(2500 * downscale_factor)
                                       ),
                            resampling=Resampling.bilinear))


## MIR image with 20m ppx
with rasterio.open("../swir.jp2", tiled=True, blockxsize=256, blockysize=256, num_threads='all_cpus') as dataset2:
    windows_d2 = []
    windows_d2.append(dataset2.read(window=Window(1250, 1500, 1250, 750)))
    windows_d2.append(dataset2.read(window=Window(3250, 3500, 1250, 750)))

## Calculate MNDWI, resample to 10m ppx
mndwi = []
water = []
result = []
zoomed = []
for i in range(len(windows_d1)):
    print(windows_d1[i].shape, windows_d2[i].shape)
    mndwi.append((np.float32(windows_d1[i][1]) - np.float32(windows_d2[i])) / (windows_d1[i][1] + windows_d2[i]))
    water.append(np.where(mndwi[i] > -0.4, 1, 0))
    result.append(ndimage.maximum_filter(water[i], size=5))
    result[i] = ndimage.minimum_filter(result[i], size=5)
    zoomed.append(ndimage.zoom(result[i], (1, 2, 2), order=0))

## Open full TCI image
with rasterio.open("../tci.jp2", tiled=True, blockxsize=256, blockysize=256, num_threads='all_cpus') as src:
    windows_src = []
    windows_src.append(src.read(window=Window(2500, 3000, 2500, 1500)))
    windows_src.append(src.read(window=Window(6500, 7000, 2500, 1500)))

filtered_src = []

## Apply MNDWI mask
for i in range(len(windows_src)):
    filtered_src.append(np.where(zoomed[i] == 1, windows_src[i], 0))


## Plot results (TCI vs. masked image)
##fig, ((ax1, bx1), (ax2, bx2)) = plt.subplots(nrows=2, ncols=2)
##show(windows_src[0], ax=ax1, title='route 1 true color')
##show(windows_src[1], ax=ax2, title='route 2 true color')
##show(filtered_src[0], ax=bx1, title='route 1 water filter')
##show(filtered_src[1], ax=bx2, title='route 2 water filter')
##plt.show()

## Vessel detection
## Vessel Index VI = NIR - Red Channel
## Open NIR image
with rasterio.open("../nir.jp2", tiled=True, blockxsize=256, blockysize=256, num_threads='all_cpus') as nir:
    windows_nir = []
    windows_nir.append(nir.read(window=Window(2500, 3000, 2500, 1500)))
    windows_nir.append(nir.read(window=Window(6500, 7000, 2500, 1500)))

with rasterio.open("../red.jp2", tiled=True, blockxsize=256, blockysize=256, num_threads='all_cpus') as red:
    windows_red = []
    windows_red.append(red.read(window=Window(2500, 3000, 2500, 1500)))
    windows_red.append(red.read(window=Window(6500, 7000, 2500, 1500)))

with rasterio.open("../blue.jp2", tiled=True, blockxsize=256, blockysize=256, num_threads='all_cpus') as blue:
    windows_blue = []
    windows_blue.append(blue.read(window=Window(2500, 3000, 2500, 1500)))
    windows_blue.append(blue.read(window=Window(6500, 7000, 2500, 1500)))

with rasterio.open("../green.jp2", tiled=True, blockxsize=256, blockysize=256, num_threads='all_cpus') as green:
    windows_green = []
    windows_green.append(green.read(window=Window(2500, 3000, 2500, 1500)))
    windows_green.append(green.read(window=Window(6500, 7000, 2500, 1500)))

vi = []
vessels_src = []
red_src = []
blue_src = []
green_src = []
nir_src = []

## Apply water mask to all 10m channels (BGR, NIR)
for i in range(len(windows_d1)):
    blue_src.append(np.where(filtered_src[i] > 0, windows_blue[i], 0))
    green_src.append(np.where(filtered_src[i] > 0, windows_green [i], 0))
    red_src.append(np.where(filtered_src[i] > 0, windows_red[i], 0))
    nir_src.append(np.where(filtered_src[i] > 0, windows_nir[i], 0))
    
    #vi.append(windows_nir[i] - filtered_src[i][2])
    ## Calculate Vessel Index
    vi.append(nir_src[i].astype(int) - red_src[i].astype(int))
    # Max value of spectral array instead of actual value - needs to be checked
    vessels_src.append(np.where(vi[i] > 0, 6215, 0))
    #vessels_src.append(np.where(vi[i] > 0, filtered_src[i], 0))

## Get spectral array: Sum of all 10m channels
spectral_array = [red_src[0][0].astype(int), blue_src[0][0].astype(int), green_src[0][0].astype(int), nir_src[0][0].astype(int)]
spectral_sum = sum(spectral_array)

print(red_src[0][0][0,0], " ", blue_src[0][0][0,0], " ", green_src[0][0][0,0], " ", nir_src[0][0][0,0])
    
print("Spectral Sum")
print(spectral_sum)

## Add VI values to spectral array
spectral_sum = np.where(spectral_sum < vessels_src[0], vessels_src[0], spectral_sum)
## Perform binomial logistic regression (fit results to scale of 0 to 1)
rescale = np.interp(spectral_sum, (spectral_sum.min(), spectral_sum.max()), (0, 1))

print("Rescaling")
print(rescale)
print(np.amax(spectral_sum))
print(mad(rescale))

## Calculate mean and mean absolute deviation - will serve as threshold for sea features
meanval = np.mean(rescale)
dev = mad(rescale)

## Use mean to create binary mask of vessels
vessels = np.where((rescale > meanval + dev), 1, 0)

## Convert to float before plotting (float uses 0..1 RGB values instead of int 0..255 scale)
vessels = vessels.astype('float32')
vessel_mask = vessels.astype('bool')
vessels = ndimage.median_filter(vessels, size=3)

shapes = rasterio.features.shapes(vessels)

print("Shapes")

## Plot results (TCI vs. masked image)
fig, ((ax1, bx1, cx1), (ax2, bx2, cx2)) = plt.subplots(nrows=2, ncols=3)
show(windows_src[0], ax=ax1, title='route 1 true color')
for shape in shapes:
    ##print(shape)
    coords = shape[0]['coordinates']
    pol = Polygon(coords[0])
    # filter areas that are too big (e.g. islands, sandbanks, ...)
    if(pol.area > 100):
        continue
    # Check if shape is close to coast (we don't care about vessels close to coasts
    # and can rule out coast fragments like this
    bounding_center = (int((pol.bounds[0] + pol.bounds[2])/2), int((pol.bounds[1] + pol.bounds[3])/2))
    cols_nsew = []
    #print(bounding_center[1], " < ", filtered_src[1][0].shape[0], " and ", bounding_center[1], " >= 0 and ", bounding_center[0] , " < " , filtered_src[1][0].shape[1], " and ", bounding_center[0], " >= 0") 
    if(bounding_center[1] + 50 < filtered_src[0][0].shape[0] and bounding_center[1] - 50 >= 0 and bounding_center[0] < filtered_src[0][0].shape[1] and bounding_center[0] >= 0):
        cols_nsew.append(filtered_src[0][0][(bounding_center[1] + 50), (bounding_center[0])])
        cols_nsew.append(filtered_src[0][0][(bounding_center[1] - 50), (bounding_center[0])])

    if(bounding_center[0] + 50 < filtered_src[0][0].shape[1] and bounding_center[0] - 50 >= 0 and bounding_center[1] < filtered_src[0][0].shape[0] and bounding_center[1] >= 0):
        cols_nsew.append(filtered_src[0][0][(bounding_center[1]), (bounding_center[0] + 50)])
        cols_nsew.append(filtered_src[0][0][(bounding_center[1]), (bounding_center[0] - 50)])
    if(not 0 in cols_nsew):
        x = [i for i,j in coords[0]]
        y = [j for i,j in coords[0]]
        ax1.plot(x,y)
        #a = [bounding_center[0] + 50, bounding_center[0] - 50, bounding_center[0], bounding_center[0]]
        #b = [bounding_center[1], bounding_center[1], bounding_center[1] + 50, bounding_center[1] - 50]
        #ax1.plot(a,b)

show(windows_src[1], ax=ax2, title='route 2 true color')
show(filtered_src[0], ax=bx1, title='route 1 water filter')
show(filtered_src[1], ax=bx2, title='route 2 water filter')
show(vessels, ax=cx1, title='route 1 vessel index')
show(vessels_src[1], ax=cx2, title='route 2 vessel index')
bool_arr = vessels_src[0] == filtered_src[0]
print(np.all(bool_arr))
print(filtered_src[0][1][1120, 1711])
plt.show()

##print("NIR, RED")
##print(windows_nir[0].size, windows_red[0].size)
##print(windows_nir[0], windows_red[0])
##
##fig = plt.figure()
##ax = plot.show(windows_nir[0])
##fig.show()
#plot.show(windows_src[0][2])
#plot.show(windows_nir[0] - windows_src[0][2])
##print("NIR - RED")
##test = windows_nir[0].astype(int) - windows_red[0].astype(int)
##print(windows_nir[0].astype(int) - windows_red[0].astype(int))
##print("Test:")
##print(test[0][908, 401])
##print(np.amin(test))

