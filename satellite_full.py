import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio import plot
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling
from rasterio import mask
import rasterio.features
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shapely.geometry import shape, mapping, Polygon
import fiona
from fiona.crs import from_epsg
from scipy import ndimage, misc, signal

#to display RGB
dataset = rasterio.open('tci.jp2')
dataset2 = rasterio.open('channel11.jp2')

downscale_factor = 1/2

tci_mask = dataset.read(
        [1,2,3],         
        out_shape=(
            dataset.count,
            int(dataset.height * downscale_factor),
            int(dataset.width * downscale_factor)
        ),
        resampling=Resampling.bilinear)

transform = dataset.transform * dataset.transform.scale(
    (dataset.width / tci_mask.shape[-1]),
    (dataset.height / tci_mask.shape[-2])
)

mir = dataset2.read()

print(dataset.width)
print(dataset2.width)
print(dataset.bounds)

print("Make filter...")

mndwi = (np.float32(tci_mask[1]) - np.float32(mir)) / (tci_mask[1] + mir)

water = np.where(mndwi > -0.8, 1, 0)

#show_hist(
#    mndwi, bins=50, lw=0.0, stacked=False, alpha=0.3,
#    histtype='stepfilled', title="Histogram")

result = ndimage.maximum_filter(water, size=5)
result = ndimage.minimum_filter(result, size=5)

print("Made filter.")
print("Rescaling...")

zoomed = ndimage.zoom(result, (1, 2, 2), order=0)

print("Rescaling successful.")
print(result.shape)
print(zoomed.shape)

print("Read full TCI image...")
src = dataset.read([1,2,3])
print("Full image read.")
print("Apply filter to image.")
src = np.where(zoomed == 1, src, 0)
print("Filter applied.")
print("Everything is ready.")

print("Plotting...")

plot.show(zoomed, cmap='Blues')
plot.show(src)
