import sys
import matplotlib.pyplot as plt

from analysis import ISCATDataProcessor


# Command line argument allowed, otherwise a file dialog will appear
filename = sys.argv[1] if len(sys.argv) > 1 else None
data = ISCATDataProcessor(filename)

images = data.images
peaks = data.peaks()
wavelens = data.wavelen()
fig, ax = plt.subplots(figsize=(8, 5))

for peak in peaks:
    ax.plot(wavelens, images[:,*peak])

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Contrast')
plt.show()