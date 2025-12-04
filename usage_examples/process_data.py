import tkinter as tk
from tkinter import filedialog
import os

import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import analysis as al

filepath = filedialog.askopenfilename(
    title='Select data',
    filetypes=[('Raw data', '.npy')]
)

data, metadata = al.load_measurement(filepath)

name, ext = os.path.splitext(filepath)
tiff.imwrite(f'{name}_processed.tif', al.float_to_mono(data))