import tkinter as tk
from tkinter import filedialog
import os

import matplotlib.pyplot as plt

import analysis as al
import numpy as np

filepath = filedialog.askopenfilename(
    title='Select data',
    filetypes=[('Raw data', '.npy')]
)

data, metadata = al.load_measurement(filepath)
peaks = al.identify_peaks(filepath)

wavelendata = metadata['Laser.wavelength [nm]']
wavelens = np.linspace(wavelendata['Start'], wavelendata['Stop'], wavelendata['Number'])

fig, ax = plt.subplots(figsize=(8, 5))

for peak in peaks:
    ax.plot(wavelens, data[:,*peak])

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Contrast')
plt.show()