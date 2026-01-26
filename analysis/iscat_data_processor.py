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

from . import processing as pc
from .widgets import start_browser, start_peakfinder, peak_editor




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
            self.images = data['images']
            self.background = data['background']
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


    def process(self, data):
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
    

    def peak_values(self, use_symmetry=True, reprocess=False, peakfile=None) -> NDArray | list:
        peaks = self.peaks(use_symmetry, reprocess, peakfile).astype(np.uint16)

        n, d, w = self.images.shape[:-2]
        n_idx = np.arange(n)[None, :, None, None]   # shape (7,1,1)
        d_idx = np.arange(d)[None, None, :, None] # shape (1,20,1)
        w_idx = np.arange(w)[None, None, None, :]   # shape (1,1,8)

        return self.images[n_idx, d_idx, w_idx, peaks[...,0], peaks[...,1]]
    
    def has_peakfile(self):
        storage_path = os.path.join('Data', self._hash_file(self.filepath))
        name, ext = os.path.splitext(storage_path)

        peakfile = f'{name}_peaks.npy'
        return os.path.exists(peakfile)
    
    def peaks(self, use_symmetry=True, reprocess=False, peakfile=None):
        """Sees if the peaks have already been identified, if not identifies them and saves to file."""
        # Peak file path
        if peakfile is None:
            storage_path = os.path.join('Data', self._hash_file(self.filepath))
            name, ext = os.path.splitext(storage_path)

            peakfile = f'{name}_peaks.npy'

        # Check if already processed
        if os.path.exists(peakfile) and not reprocess:
            peaks = np.load(peakfile)
        else:
            peaks = self._find_peaks(use_symmetry)
            np.save(peakfile, peaks)
                
        return peaks


    def _find_peaks(self, use_symmetry=True):
        """Identify peaks through circular convolution and return in list."""
        print('Looking for peaks...')
        peaks = start_peakfinder(self, use_symmetry=use_symmetry)
        if peaks is None or len(peaks) < len(self.images):
            print('No peaks selected, exiting')
            exit(0)

        print(f'Found peaks!')

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