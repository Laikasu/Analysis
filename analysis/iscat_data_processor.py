"""
iscat_image_processor.py

Gives an object oriented abstracted way to analyze data
"""

import os

import yaml
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.filters import difference_of_gaussians
import hashlib

from tkinter import filedialog
from collections.abc import Mapping

from . import processing as pc
from .symmetry import symmetry
from .analysis import ask_threshold




class ISCATDataProcessor():
    def __init__(self, filepath=None):
        self._load_measurement(filepath)
        self._data = None
    
    def _load_measurement(self, filepath=None, reprocess=False):
        # Select file
        if filepath is None:
            filepath = filedialog.askopenfilename(
                title='Select Data',
                filetypes=[('Raw Data', '.npy')],
                initialdir='Measurements'
            )
        
        if not filepath:
            print('No file selected. Exiting')
            exit(0)
        
        self.filepath = filepath

        # Check the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'File {filepath} not found')
        
        if not os.path.splitext(filepath)[1] in ('.tif', '.npy'):
            raise ValueError('File must be a .tif or .npy file')
        

        # Load metadata
        metadatafile = os.path.splitext(filepath)[0] + '.yaml'
        if not os.path.exists(metadatafile):
            raise FileNotFoundError(f'Metadata file {metadatafile} not found')

        with open(metadatafile, 'r') as file:
            raw_metadata = yaml.safe_load(file)
        
        if not isinstance(raw_metadata, dict):
            raise TypeError("Expected YAML to contain a dictionary")

        self.metadata: dict = raw_metadata
        

        # If already processed, return that
        os.makedirs('Data', exist_ok=True)
        storage_path = os.path.join('Data', self._hash_file(filepath))
        if os.path.exists(storage_path) and not reprocess:
            data = np.load(storage_path)
            self.images = data['images']
            self.background = data['background']
            return
        

        # Process
        print(f'Processing {os.path.basename(filepath)}')

        self._data = np.load(filepath).squeeze()
        # Average
        if 'Camera.averaging' in self.metadata.keys():
            count = self.metadata['Camera.averaging']
            image = np.average(self._data[:,:count],axis=1)
        else:
            image = self._data[:,0]

        # Background subtraction
        self.background = np.array([pc.common_background(im[-4:]) for im in self._data])
        self.images = pc.background_subtracted(image,self.background)
        print(f'Processed {os.path.basename(filepath)}')

        np.savez(storage_path, images=self.images, background=self.background)

    def peaks(self, use_symmetry=True, threshold=None, reprocess=False, margin=20):
        """Sees if the peaks have already been identified, if not identifies them and saves to file."""
        # Peak file path
        storage_path = os.path.join('Data', self._hash_file(self.filepath))
        name, ext = os.path.splitext(storage_path)
        peakfile = f'{name}_peaks.npy'

        # Check if already processed
        if os.path.exists(peakfile) and not reprocess:
            peaks = np.load(peakfile)
        else:
            peaks = self._find_peaks(self.images, use_symmetry=True, threshold=threshold)
            np.save(peakfile, peaks)

        # Filter marginal peaks
        count, rows, cols = self.images.shape
        indeces = []
        for i, peak in enumerate(peaks):
            if peak[0] < margin or peak[0] > rows - margin or peak[1] < margin or peak[1] > cols - margin:
                indeces.append(i)
        peaks = np.delete(peaks, indeces, axis=0)
        return peaks


    def _find_peaks(self, data, use_symmetry=True, threshold=None):
        """Identify peaks through circular convolution and return in list."""
        print('Looking for peaks...')

        processed = []
        filtered = difference_of_gaussians(data, 2, 4)

        # Use covolution with circle to find the psfs
        if use_symmetry:
            processed.append(symmetry(filtered,np.arange(4,10,2)))
        else:
            processed.append(filtered)
        
        processed = np.array(processed).squeeze()

        if threshold is None:
            threshold = ask_threshold(data, processed)
            if threshold is None:
                print('No threshold selected, exiting')
                exit(0)

        peaks = self._local_max_peaks(np.average(processed, axis=0), threshold)

        print(f'Found {peaks.shape[0]} peaks!')

        return peaks


    def _local_max_peaks(self, image, threshold = 1/20):
        """Find peaks based on a local maximum threshold technique"""

        # Find maxima
        thresholded = np.where(np.abs(image)>np.max(image)*threshold, 1, 0)
        local_max = ndimage.maximum_filter(thresholded, size=5)
        # Label objects
        labeled, num_features = ndimage.label(local_max)

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(image)
        # ax[1].imshow(labeled)
        # plt.show()
        # Remove unresolvable objects
        sizes = ndimage.sum(thresholded, labeled, index=np.arange(1, num_features + 1))
        remove_indeces = np.arange(1, num_features + 1)[sizes > 100]
        labeled[np.isin(labeled, remove_indeces)] = 0
        labeled, num_features = ndimage.label(labeled)

        

        peaks = np.array(ndimage.center_of_mass(image, labeled, range(1, num_features+1))).astype(int)
        return peaks

    def _hash_file(self, filepath):
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        
        hash = hasher.hexdigest()[:8]
        name, ext = os.path.splitext(os.path.basename(filepath))

        return f"{name}_{hash}.npz"
    
    def raw(self):
        if self._data is None:
            self._data = np.load(self.filepath).squeeze()
        return self._data
    
    def roi_size(self) -> tuple:
        """Return roi_size first x then y"""
        pxsize_image = self.metadata.get('Camera.pixel_size [um]', None)
        magnification = self.metadata.get('Setup.magnification', None)
        if pxsize_image == None:
            raise ValueError("Pixel Size value is missing in metadata")
        if magnification == None:
            raise ValueError("Magnification value is missing in metadata")

        pxsize_object = pxsize_image/magnification
        count, rows, cols = self.images.shape
        return (pxsize_object*cols, pxsize_object*rows)
    
    def wavelen(self) -> NDArray:
        wavelen = self.metadata.get('Laser.wavelength [nm]', None)
        if wavelen is None:
            raise ValueError("Wavelength value is missing in metadata")
        if isinstance(wavelen, Mapping):
            return np.linspace(
                wavelen['Start'],
                wavelen['Stop'],
                wavelen['Number'])
        return wavelen
    
    def defocus(self) -> NDArray:
        defocus = self.metadata.get('Setup.z_focus [um]', self.metadata.get('Setup.defocus [nm]', None))
        if defocus is None:
            raise ValueError("Defocus value is missing in metadata")
        if isinstance(defocus, Mapping):
            return np.linspace(
                defocus['Start'],
                defocus['Stop'],
                defocus['Number'])
        return defocus