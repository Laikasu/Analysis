import numpy as np
from numpy.typing import NDArray


def common_background(backgrounds):
    output_weights = np.zeros_like(backgrounds, dtype=np.float64)
    for i in range(len(backgrounds)):
        for j in range(len(backgrounds)):
            if (i < j):
                diff = np.divide(np.subtract(backgrounds[i], backgrounds[j], dtype=np.int32), backgrounds[j],dtype=np.float64)
                weight = np.exp(-np.abs(diff)/0.01)
                output_weights[i] = np.maximum(weight, output_weights[i])
                output_weights[j] = np.maximum(weight, output_weights[j])
    
    
    return np.average(backgrounds, axis=0, weights=output_weights).astype(backgrounds[0].dtype)

def background_subtracted(data, background, background_norm=True) -> NDArray:
    if background_norm:
        return np.divide(np.subtract(data, background,dtype=np.float64), background, dtype=np.float64)
    else:
        return np.subtract(data, background, dtype=np.int32)

def float_to_mono(data, normalize=False):
    if normalize:
        d = (data - np.min(data))
        data = (d/np.max(d))*2-1
    else:
        data = np.clip(data, -1, 1)

    return ((data+1)*32767).astype(np.uint16)