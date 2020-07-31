import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

well_data = pd.read_csv('../data/bcogc_well_comp_info.csv')
geo_maps = pd.read_csv('../data/raster_stack.csv')

fig, (ax, ax2, ax3, ax4) = plt.subplots(ncols=4)
ax.set_title("First Order Residual")
ax.tripcolor(geo_maps["x"], geo_maps["y"], geo_maps["first_order_residual"])
ax2.set_title("Isotherm")
ax2.tripcolor(geo_maps["x"], geo_maps["y"], geo_maps["isotherm"])
ax3.set_title("Third Order Residual")
ax3.tripcolor(geo_maps["x"], geo_maps["y"], geo_maps["third_order_residual"])
ax4.set_title("Third Order Residual")
ax4.tripcolor(geo_maps["x"], geo_maps["y"], geo_maps["third_order_residual"])
plt.show()