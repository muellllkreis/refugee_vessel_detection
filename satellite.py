import math
import rasterio
import matplotlib.pyplot as plt

image_file = "sample.jp2"
sat_data = rasterio.open(image_file)

width_in_projected_units = sat_data.bounds.right - sat_data.bounds.left
height_in_projected_units = sat_data.bounds.top - sat_data.bounds.bottom

print("Width: {}, Height: {}".format(width_in_projected_units, height_in_projected_units))
print("Rows: {}, Columns: {}".format(sat_data.height, sat_data.width))

# Upper left pixel
row_min = 0
col_min = 0

# Lower right pixel.  Rows and columns are zero indexing.
row_max = sat_data.height - 1
col_max = sat_data.width - 1

# Transform coordinates with the dataset's affine transformation.
topleft = sat_data.transform * (row_min, col_min)
botright = sat_data.transform * (row_max, col_max)

print("Top left corner coordinates: {}".format(topleft))
print("Bottom right corner coordinates: {}".format(botright))

print(sat_data.count)

# sequence of band indexes
print(sat_data.indexes)

b, g, r = sat_data.read()

fig = plt.imshow(b)
plt.show()
