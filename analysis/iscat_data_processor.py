"""
iscat_image_processor.py

Gives an object oriented abstracted way to analyze data
"""

import os

import yaml
import numpy as np
from numpy.typing import NDArray
import hashlib
import re

from tkinter import filedialog
from collections.abc import Mapping

from numpy.lib.stride_tricks import sliding_window_view
from skimage.registration import phase_cross_correlation
from scipy.optimize import curve_fit

from . import processing as pc
from .widgets import start_browser, start_peakfinder, peak_editor

sigmax = 3
sigmay = 3
def gaussian_2d(coords, A, mux, muy):
    x, y = coords
    return A * np.exp(-(((x - mux)**2)/(2*sigmax**2) + ((y - muy)**2)/(2*sigmay**2)))


class ISCATDataProcessor():
    def __init__(self, filepath=None, reprocess=False):
        self._load_measurement(filepath, reprocess)
        self._data = None
    
    def _load_measurement(self, filepath=None, reprocess=False):
        # Select file
        if filepath is None:
            filepath = filedialog.askopenfilename(
                title='Select Data',
                filetypes=[('Raw Data Files', '*.npy *.npz'), ('All Files', '*.*')],
                initialdir='Measurements'
            )
        
        if not filepath:
            print('No file selected. Exiting')
            exit(0)
        
        self.filepath = filepath
        print(f'Selected {self.filepath}')

        # Check the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'File {filepath} not found')
        
        if not os.path.splitext(filepath)[1] in ('.tif', '.npy', '.npz'):
            raise ValueError('File must be a .tif, .npy or .npz file')
        
        self._stack = False
        if os.path.splitext(filepath)[1] == '.npz':
            self._stack = True
        

        # Load metadata
        metadatafile = os.path.splitext(filepath)[0] + '.yaml'
        if not os.path.exists(metadatafile):
            metadatafile = re.sub(r"_\d(?=\.yaml$)", "", metadatafile)
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
            data = np.load(storage_path, mmap_mode='r')
            self.images: np.ndarray = data['images']
            self.background: np.ndarray = data['background']
            print('Loaded in data')
            return
        

        # Process
        print(f'Processing {os.path.basename(filepath)}')

        if self._stack:
            data = np.load(filepath, mmap_mode='r')
            self.images = []
            self.background = []

            size = len(data.files)
            for i, key in enumerate(data.files):
                print(f'Processing {i+1}/{size}')
                images, bgs = self.process(data[key])
                self.images.append(images)
                self.background.append(bgs)
            
            self.images = np.stack(self.images)
            self.background = np.stack(self.background)
        else:
            self._data = np.load(filepath).squeeze()
            self.images, self.background = self.process(self._data)
        
        print(f'Processed {os.path.basename(filepath)}')

        np.savez(storage_path, images=self.images, background=self.background)


    def process(self, data) -> tuple[NDArray, NDArray]:
        # Average
        if 'Camera.averaging' in self.metadata.keys():
            count = self.metadata['Camera.averaging']
            idx = [slice(None)] * data.ndim
            idx[-3] = slice(None, count)
            image = np.average(data[tuple(idx)],axis=-3)
        else:
            image = np.take(data, 0, axis=-3)

        # Background subtraction
        idx = [slice(None)] * data.ndim
        idx[-3] = slice(-4, None)
        bg = data[tuple(idx)]
        *batch, i, h, w = bg.shape
        reshaped =  bg.reshape(-1, i, h, w)

        background = np.array([pc.common_background(im[-4:]) for im in reshaped]).reshape(*batch, h, w)
        return (pc.background_subtracted(image, background),
                background)
    

    def peaks(self):
        peaks = self.peak_positions().astype(np.uint16)
        
        # Gaussian blur makes less sensitive to exact pixel, sort of averages around the peak
        shape = self.images.shape[:-2]
        idx = np.ogrid[*tuple(slice(0, s) for s in shape)]
        return self.images[*idx, peaks[...,0], peaks[...,1]]
        
    def _fit(self, size=7):
        """Fitting function. Current result is suboptimal"""
        # Fitting
        print('Starting fit')

        # Average over all peaks to obtain mean patch
        patches = np.mean(self.psf(size), axis=0)

        y, x = np.mgrid[0:size, 0:size]
        xdata = np.vstack([x.ravel(), y.ravel()])

        A = np.zeros(patches.shape[:-2], np.float64)
        
        # A, x0, y0
        popt = [0, size//2, size//2]
        for idx in np.ndindex(A.shape):
            patch = patches[idx]
            ydata = patch.ravel()
            p0 = popt
            lower_bounds = (-np.inf, 2,2)
            upper_bounds = (np.inf, 4,4)
            popt, pcov = curve_fit(gaussian_2d, xdata, ydata, p0, bounds=(lower_bounds, upper_bounds))
            A[idx]  = popt[0]

        # plt.ioff()
        print('Done!')
        return A
        
    
    def psf(self, size=7):
        """Gives the point spread function for each peak"""
        windows = sliding_window_view(self.images, (size, size), axis=(-2,-1))

        # Weird numpy stuff to index for any dim
        shape = self.images.shape[:-2]
        idx = np.ogrid[*tuple(slice(0, s) for s in shape)]

        peaks = self.peak_positions().astype(np.uint16)
        patches = windows[*idx, peaks[...,0]-size//2, peaks[...,1]-size//2]
        return patches


    def has_peakfile(self):
        storage_path = os.path.join('Data', self._hash_file(self.filepath))
        name, ext = os.path.splitext(storage_path)

        peakfile = f'{name}_peaks.npy'
        return os.path.exists(peakfile)


    def peak_positions(self, peakfile=None):
        """Sees if the peaks have already been identified, if not identifies them and saves to file."""
        # Peak file path
        if peakfile is None:
            storage_path = os.path.join('Data', self._hash_file(self.filepath))
            name, ext = os.path.splitext(storage_path)

            peakfile = f'{name}_peaks.npy'

        # Check if already processed
        if os.path.exists(peakfile):
            peaks = np.load(peakfile)
        else:
            peaks = self.find_peaks(peakfile)
                
        return peaks


    def find_peaks(self, peakfile=None):
        """Identify peaks through circular convolution and return in list."""
        
        print('Looking for peaks...')
        peaks = start_peakfinder(self)

        if peaks is None or (self._stack and len(peaks) < len(self.images)):
            print('No peaks selected, exiting')
            exit(0)

        print(f'Found peaks!')

        # Standard location
        if peakfile is None:
            storage_path = os.path.join('Data', self._hash_file(self.filepath))
            name, ext = os.path.splitext(storage_path)
            peakfile = f'{name}_peaks.npy'
        
        np.save(peakfile, peaks)

        return peaks

    def _hash_file(self, filepath):
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        
        hash = hasher.hexdigest()[:8]
        name, ext = os.path.splitext(os.path.basename(filepath))

        return f"{name}_{hash}.npz"
    

    # Functions that return parameters
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
        defocus = self.metadata.get('Setup.z_focus [um]', self.metadata.get('Setup.defocus [um]', None))
        if defocus is None:
            raise ValueError("Defocus value is missing in metadata")
        if isinstance(defocus, Mapping):
            return np.linspace(
                defocus['Start'],
                defocus['Stop'],
                defocus['Number'])
        return defocus
    
    def browse(self):
        return start_browser(self)
    
    def edit_peaks(self):
        return peak_editor(self)