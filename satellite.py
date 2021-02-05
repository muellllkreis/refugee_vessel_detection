import rasterio
from rasterio.plot import show
from rasterio import plot
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from shapely.geometry import Polygon, Point, MultiPoint
from shapely import speedups
import rasterio.features
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import numpy as np
import fiona
from scipy import ndimage, misc, signal
import ais_parse as ais

import pprint

def mad(data, axis=None):
    return np.nanmean(np.absolute(data - np.nanmean(data, axis)), axis)

downscale_factor = 1/2
speedups.disable()

test = rasterio.open("../tci.jp2")
print(test.bounds)
print("------")
print(test.transform)
print("------")
print(test.crs)
print(type(test))

transforms = []

vessel_list = ais.get_ais_info()

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
    transforms.append(rasterio.windows.transform(Window(2500, 3000, 2500, 1500), dataset.transform))
    windows_d1.append(dataset.read(window=Window(6500, 7000, 2500, 1500),
                            out_shape=(
                                        dataset.count,
                                        int(1500 * downscale_factor),
                                        int(2500 * downscale_factor)
                                       ),
                            resampling=Resampling.bilinear))
    transforms.append(rasterio.windows.transform(Window(6500, 7000, 2500, 1500), dataset.transform))


print("-------------")
print("Window Transform")
print(transforms[0])
print(type(dataset))
print(type(windows_d1[0]))

##for i, j in np.ndindex(windows_d1[0].shape[1:]):
##    coordinatex,coordinatey = rasterio.transform.xy(transforms[0], i, j, offset='center')
##    print ("long of result is " + str(coordinatex))
##    print ("lat of result is " + str(coordinatey))

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


windows_hsv = []
for i in range(len(windows_d1)):
    hsv = colors.rgb_to_hsv(windows_src[i].transpose()).transpose()
    windows_hsv.append(np.where(filtered_src[i] > 0, hsv, 0))

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
    green_src.append(np.where(filtered_src[i] > 0, windows_green[i], 0))
    red_src.append(np.where(filtered_src[i] > 0, windows_red[i], 0))
    nir_src.append(np.where(filtered_src[i] > 0, windows_nir[i], 0))
    
    ## Calculate Vessel Index
    vi.append(nir_src[i].astype(int) - red_src[i].astype(int))
    # Max value of spectral array instead of actual value - needs to be checked
    vessels_src.append(np.where(vi[i] > 0, 6215, 0))

all_shapes = []
all_vessels = []

## Get spectral array: Sum of all 10m channels
for i in range(len(windows_src)):    
    spectral_array = [red_src[i][0].astype(int), blue_src[i][0].astype(int), green_src[i][0].astype(int), nir_src[i][0].astype(int)]
    spectral_sum = sum(spectral_array)

    print(red_src[i][0][0,0], " ", blue_src[i][0][0,0], " ", green_src[i][0][0,0], " ", nir_src[i][0][0,0])
        
    print("Spectral Sum")
    print(spectral_sum)

    ## Add VI values to spectral array
    spectral_sum = np.where(spectral_sum < vessels_src[i], vessels_src[i], spectral_sum)
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
    all_vessels.append(ndimage.median_filter(vessels, size=3))

    all_shapes.append(rasterio.features.shapes(all_vessels[i]))

## Plot results (TCI vs. masked image)
fig, ((ax1, bx1, cx1), (ax2, bx2, cx2)) = plt.subplots(nrows=2, ncols=3)

ax_dict = {0: ax1, 1: ax2}

for i in range(len(windows_src)):
    windows_hsv[i][windows_hsv[i] == 0] = np.nan
    mean_hsv = np.nanmean(windows_hsv[i][1])
    dev_hsv = mad(windows_hsv[i][1])

    print("MEAN HSV:", mean_hsv)
    print("DEV HSV:", dev_hsv)
    
    axis = ax_dict[i]
    show(windows_src[i], ax=axis, title='route ' + str(i) + ' true color', transform=transforms[i])
    for shape in all_shapes[i]:
        coords = shape[0]['coordinates']
        pol = Polygon(coords[0])
        # filter areas that are too big (e.g. islands, sandbanks, ...)
        if(pol.area > 100):
            continue
        # Check if shape is close to coast (we don't care about vessels close to coasts
        # and can rule out coast fragments like this
        bounding_center = (int((pol.bounds[0] + pol.bounds[2])/2), int((pol.bounds[1] + pol.bounds[3])/2))
        cols_nsew = []
        if(bounding_center[1] + 50 < filtered_src[0][0].shape[0] and bounding_center[1] - 50 >= 0 and bounding_center[0] < filtered_src[i][0].shape[1] and bounding_center[0] >= 0):
            cols_nsew.append(filtered_src[i][0][(bounding_center[1] + 50), (bounding_center[0])])
            cols_nsew.append(filtered_src[i][0][(bounding_center[1] - 50), (bounding_center[0])])

        if(bounding_center[0] + 50 < filtered_src[0][0].shape[1] and bounding_center[0] - 50 >= 0 and bounding_center[1] < filtered_src[i][0].shape[0] and bounding_center[1] >= 0):
            cols_nsew.append(filtered_src[i][0][(bounding_center[1]), (bounding_center[0] + 50)])
            cols_nsew.append(filtered_src[i][0][(bounding_center[1]), (bounding_center[0] - 50)])
        if(not 0 in cols_nsew):
            x = [i for i,j in coords[0]]
            y = [j for i,j in coords[0]]
            tr_x, tr_y = rasterio.transform.xy(transforms[i], y, x, offset='center')

            # Get translated representation of shape to test against AIS data
            translated_coords = []
            for n in range(len(tr_x)):
                translated_coords.append((tr_x[n], tr_y[n]))

            testpol = Polygon(translated_coords)

            ### test hsv color of polygon
            print("--------------Shape--------------")
            print(coords[0][0])
            hsv_sum = 0
            for xval, yval in coords[0]:
                #print(xval, ", ", yval)
                hsv_sum += windows_hsv[i][1][int(yval) - 1, int(xval) - 1]
                #print(windows_hsv[i][1][int(yval) - 1, int(xval) - 1])
            print("---------------------------------")
            hsv_avg = hsv_sum / len(coords[0])
            print(hsv_avg)
            if(hsv_avg < mean_hsv + 0.05 and hsv_avg > mean_hsv - 0.05) or (np.isnan(hsv_avg)):
                print("Likely water")
                continue
            else:
                print("Likely something else")
            ###
            
            is_registered = False
            for vessel in vessel_list:
                if testpol.contains(Point(vessel.long, vessel.lat)):
                    is_registered = True
                    axis.text(vessel.long + 5, vessel.lat + 5, vessel.name, color='white', fontsize=5)
                    axis.plot(tr_x, tr_y, color='green')
            if not is_registered:
                axis.plot(tr_x, tr_y, color='red')

print("CRS:")
print(dataset.crs)

print(windows_src[1].shape)

#show(windows_hsv[0], ax=ax1, title='route 1 hsv', transform=transforms[0])
#show(windows_src[1], ax=ax2, title='route 2 true color', transform=transforms[1])
show(filtered_src[0], ax=bx1, title='route 1 water filter')
show(filtered_src[1], ax=bx2, title='route 2 water filter')
show(all_vessels[0], ax=cx1, title='route 1 vessel index')
show(all_vessels[1], ax=cx2, title='route 2 vessel index')
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

