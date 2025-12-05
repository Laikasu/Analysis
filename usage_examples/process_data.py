import os
import sys

from tkinter import filedialog

import tifffile as tiff

from analysis import ISCATDataProcessor

import analysis as al
import numpy as np

def float_to_mono(data, normalize=False):
    if normalize:
        d = (data - np.min(data))
        data = (d/np.max(d))*2-1
    else:
        data = np.clip(data, -1, 1)

    return ((data+1)*32767).astype(np.uint16)

# Command line argument allowed, otherwise a file dialog will appear
filename = sys.argv[1] if len(sys.argv) > 1 else None
data = ISCATDataProcessor(filename)


# Save
filepath = filedialog.asksaveasfilename(
                title='Save processed data',
                filetypes=[('Image stack', '.tif')],
                initialdir=os.path.dirname(data.filepath)
            )

if filepath is not None:
    tiff.imwrite(filepath, float_to_mono(data.images))