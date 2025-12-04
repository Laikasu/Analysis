import tkinter as tk
from tkinter import filedialog
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import analysis as al
import numpy as np

filepath = '/home/casperdely/Measurements/2025-10-09_focus_sweep_pfs_avg/40nm_focus.npy'
# filepath = filedialog.askopenfilename(
#     title='Select data',
#     filetypes=[('Raw data', '.npy')]
# )
print(filepath)

data, metadata = al.load_measurement(filepath)


fig, ax = plt.subplots()

pxsize_image = metadata['Camera.pixel_size [um]']
magnification = metadata['Setup.magnification']
pxsize_object = pxsize_image/magnification

height, width = np.shape(data[0])

image = plt.imshow(data[0], 'gray', extent=(0, width*pxsize_object, 0, height*pxsize_object))
fig.colorbar(image)
scalebar = AnchoredSizeBar(ax.transData, 10, '10 um', 'lower center', pad=0.1, frameon=False, color='white', sep=5)
ax.add_artist(scalebar)

def update(frame):
    image.set_data(data[frame])
    return image,

anim = FuncAnimation(fig, update, len(data))
plt.show()
anim.save(os.path.splitext(filepath)[0] + '.gif')