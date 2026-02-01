import numpy as np
from scipy.signal import convolve2d

def circular_mask(radius):
    """Create a circular binary mask with the given radius."""
    size = 2 * radius + 1
    y, x = np.ogrid[:size, :size]
    center = radius
    mask = ((x - center)**2 + (y - center)**2 <= (radius-1)**2) * ((x - center)**2 + (y - center)**2 >= (radius-2)**2)
    return mask.astype(float)/np.sum(mask)

def convolve(image, radii):
    output = np.zeros_like(image)
    image_sq = image**2
    for radius in radii:
        mask = circular_mask(radius)
        mean = convolve2d(image, mask, mode='same')
        mean_sq = convolve2d(image_sq, mask, mode='same')
        var = mean_sq - mean**2
        # Threshold mean to take out low mean, very low var => big
        mask = mean_sq>np.percentile(mean_sq, 90)
        output += np.where(mask, mean**2/var, 0)
    return output

def symmetry(data:np.ndarray, radii):
    if not np.iterable(radii):
        radii = np.array([radii])
    
    *batch, h, w = data.shape
    processed = np.stack([convolve(img, radii) for img in data.reshape(-1, h, w)])

    return processed.reshape(*batch, h, w)
        