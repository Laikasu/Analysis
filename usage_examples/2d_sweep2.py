import sys
import matplotlib.pyplot as plt

import numpy as np
from analysis import ISCATDataProcessor
from matplotlib.animation import FuncAnimation

# Command line argument allowed, otherwise a file dialog will appear
filename = sys.argv[1] if len(sys.argv) > 1 else None
data = ISCATDataProcessor(filename)

images = data.images
peaks = data.peaks()
wavelens = data.wavelen()
defocus = data.defocus()
fig, ax = plt.subplots(figsize=(8, 5))

axis = 0

image = plt.imshow(images[0,0])

def update(frame):
    image.set_data(images[0,frame])
    return image,

anim = FuncAnimation(fig, update, images.shape[1])
plt.show()

# ax.legend()
# ax.set_xlabel('defocus (um)')
# ax.set_ylabel('Contrast')
# plt.show()