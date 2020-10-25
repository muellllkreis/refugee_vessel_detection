import rasterio
from rasterio.plot import show
from rasterio import plot
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import ndimage, misc, signal

downscale_factor = 1/2

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
    water.append(np.where(mndwi[i] > -0.8, 1, 0))
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

vi = []
vessels_src = []
for i in range(len(windows_d1)):
    vi.append(windows_nir[i] - filtered_src[i][2])
    vessels_src.append(np.where(vi[i] > 0, filtered_src[i], 0))

print(windows_nir[0], windows_src[0][2])
plot.show(windows_nir[0] - windows_src[0][2])
print(windows_nir[0] - windows_src[0][2])

## Plot results (TCI vs. masked image)
fig, ((ax1, bx1, cx1), (ax2, bx2, cx2)) = plt.subplots(nrows=2, ncols=3)
show(windows_src[0], ax=ax1, title='route 1 true color')
show(windows_src[1], ax=ax2, title='route 2 true color')
show(filtered_src[0], ax=bx1, title='route 1 water filter')
show(filtered_src[1], ax=bx2, title='route 2 water filter')
show(vessels_src[0], ax=cx1, title='route 1 vessel index')
show(vessels_src[1], ax=cx2, title='route 2 vessel index')
plt.show()
