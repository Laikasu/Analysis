import tkinter as tk
from tkinter import filedialog
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from analysis import ISCATDataProcessor
import numpy as np
import sys

# Command line argument allowed, otherwise a file dialog will appear
filename = sys.argv[1] if len(sys.argv) > 1 else None
data = ISCATDataProcessor(filename)


fig, ax = plt.subplots()

roi_size_x, roi_size_y = data.roi_size()
images = data.images

image = plt.imshow(images[0], 'gray', extent=(0, roi_size_x, 0, roi_size_y))
fig.colorbar(image)
scalebar = AnchoredSizeBar(ax.transData, 10, '10 um', 'lower center', pad=0.1, frameon=False, color='white', sep=5)
ax.add_artist(scalebar)

def update(frame):
    image.set_data(images[frame])
    return image,

anim = FuncAnimation(fig, update, len(images))
plt.show()


# Save
filepath = filedialog.asksaveasfilename(
                title='Save animation',
                filetypes=[('Animation', '.gif')],
                initialdir=os.path.dirname(data.filepath)
            )

if filepath is not None:
    anim.save(filepath)