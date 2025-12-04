import tkinter as tk
from tkinter import filedialog
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import analysis as al

#filepath = '/home/casper/Documents/University/MEP/Data/2025-04-28/spectrum_100nmAuNS_CTAB.tsv'
filepath = filedialog.askopenfilename(
    title='Select data',
    filetypes=[('Raw data', '.npy')]
)

data, metadata = al.load_measurement(filepath)


fig, ax = plt.subplots()

image = plt.imshow(data[0], 'gray')
fig.colorbar(image)
scalebar = AnchoredSizeBar(ax.transData, 0.2, '200 nm', 'lower center', pad=0.1, frameon=False, color='white', sep=5)

def update(frame):
    image.set_data(data[frame])
    return image,

anim = FuncAnimation(fig, update, len(data))
plt.show()
anim.save(os.path.splitext(filepath)[0] + '.gif')