import numpy as np
from scipy.signal import convolve2d

def circular_mask(radius):
    """Create a circular binary mask with the given radius."""
    size = 2 * radius + 1
    y, x = np.ogrid[:size, :size]
    center = radius
    mask = ((x - center)**2 + (y - center)**2 <= (radius-1)**2) * ((x - center)**2 + (y - center)**2 >= (radius-2)**2)
    return mask.astype(float)

def convolve(image, radii):
    output = np.zeros_like(image)
    for radius in radii:
        mask = circular_mask(radius)
        mask/=np.sum(mask)
        output += convolve2d(image, mask, mode='same')**2
    return output

def symmetry(data:np.ndarray, radii):
    if not np.iterable(radii):
        radii = np.array([radii])
    
    *batch, h, w = data.shape
    processed = np.stack([convolve(img, radii) for img in data.reshape(-1, h, w)])

    return processed.reshape(*batch, h, w)
        