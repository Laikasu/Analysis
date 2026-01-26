import sys
import matplotlib.pyplot as plt

import numpy as np
from analysis import ISCATDataProcessor


# Command line argument allowed, otherwise a file dialog will appear
filename = sys.argv[1] if len(sys.argv) > 1 else None
data = ISCATDataProcessor(filename)

images = data.images
peaks = data.peaks()
wavelens = data.wavelen()
defocus = data.defocus()
fig, ax = plt.subplots(figsize=(8, 5))

avg = np.mean(np.array([images[0,:,:,*peak] for peak in peaks]), axis=0)
for i in range(len(avg[0])):
    ax.plot(defocus, avg[:,i], label={f'{wavelens[i]:.0f} nm'})

ax.legend()
ax.set_xlabel('defocus (um)')
ax.set_ylabel('Contrast')
plt.show()