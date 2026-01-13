"""
analysis.py

Provides functions used to load, process and analyze data
"""

import numpy as np

import tifffile as tiff
import yaml
import os
import hashlib

import re


from scipy.interpolate import interp1d

from tkinter import filedialog
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


from skimage.filters import difference_of_gaussians
from scipy.special import jv
from scipy import ndimage

from . import processing as pc
from .symmetry import symmetry

import matplotlib.pyplot as plt



def peakfinder(data, processed, title="Peak Finder", min_val=1, max_val=200, default=30, use_symmetry=True):
    # --- Hidden root ---
    root = tk.Tk()
    root.withdraw()  # hide the root window


    result = None

    # --- Dialog class ---
    class ThresholdDialog(tk.Toplevel):
        def __init__(self, parent):
            self.framedim = data.ndim - 2
            super().__init__(parent)
            self.title(title)
            # self.minsize(1000,800)
            #self.attributes("-fullscreen", True)
            self.grab_set()  # modal
            self.result = default

            filtered = difference_of_gaussians(data[*([0]*self.framedim)], 2, 4)
            # Use covolution with circle to find the psfs
            if use_symmetry:
                processed = symmetry(filtered,np.arange(4,10,2))
            else:
                processed = filtered

            self.peaks = local_max_peaks(processed, default/1000)
            self.image = data if data.ndim == 2 else data[(0,)*self.framedim]

            # --- Matplotlib figure ---
            self.fig, ax = plt.subplots(figsize=(8, 5))
            self.fig.tight_layout()
            ax.set_axis_off()
            self.imshow = ax.imshow(self.image, cmap='gray')

            self.sc = ax.scatter([], [], marker='+', color='red')
            
            self.canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            if data.ndim > 2:
                # Frame slider
                self.frame_slider = tk.Scale(self, from_=0, to=data.shape[0]-1, orient="horizontal",
                                    label="Frame", command=self.update_plot)
                self.frame_slider.set(0)
                self.frame_slider.pack(fill="x", padx=10, pady=5)
            
            if data.ndim > 3:
                self.frame_slider2 = tk.Scale(self, from_=0, to=data.shape[1]-1, orient="horizontal",
                                    label="Frame", command=self.update_plot)
                self.frame_slider2.set(0)
                self.frame_slider2.pack(fill="x", padx=10, pady=5)

            # Threshold slider
            self.threshold_slider = tk.Scale(self, from_=min_val, to=max_val, orient="horizontal",
                                   label="Threshold", command=self.update_plot)
            self.threshold_slider.set(default)
            self.threshold_slider.pack(fill="x", padx=10, pady=5)

            self.button_frame = tk.Frame(self)
            self.button_frame.pack(side="bottom", anchor="e", pady=10, padx=10)
            
            # --- OK button ---
            tk.Button(self.button_frame, text="OK", command=self.on_ok).pack(side='right',padx=5)

            # Cancel button
            tk.Button(self.button_frame, text="Cancel", command=self.on_cancel).pack(side='right',padx=5)

            

            # Close cancels
            self.protocol("WM_DELETE_WINDOW", self.on_cancel)

        def update_plot(self, val):
            threshold = self.threshold_slider.get()

            frame = []
            if self.framedim > 0:
                frame.append(self.frame_slider.get())
            
            if self.framedim > 1:
                frame.append(self.frame_slider2.get())

            self.image = data[*frame]
            self.imshow.set_data(self.image)

            filtered = difference_of_gaussians(data[*frame], 2, 4)
            # Use covolution with circle to find the psfs
            if use_symmetry:
                processed = symmetry(filtered,np.arange(4,10,2))
            else:
                processed = filtered

            self.peaks = local_max_peaks(processed, threshold/1000)

            if len(self.peaks) == 1:
                self.sc.set_offsets(self.peaks[0])
            elif len(self.peaks) > 1:
                self.sc.set_offsets(self.peaks[:, [1,0]])
            self.canvas.draw_idle()
            self.result = threshold/1000

        def on_ok(self):
            #self.fig.remove()
            self.destroy()
        
        def on_cancel(self):
            #self.fig.remove()
            self.peaks = None
            self.destroy()

    # --- Open dialog and wait ---
    dialog = ThresholdDialog(root)
    root.wait_window(dialog)
    result = dialog.peaks
    root.destroy()  # clean up hidden root

    return result

def norm(a):
    return a/np.min(a)

def create_calibration():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    calibrationfile = os.path.join(current_file_path, 'calibration.npy')
    print(calibrationfile)
    filepath = filedialog.askopenfilepath(
        title='Select laser wavelength sweep calibration',
        filetypes=[('Data', '.tif'), ]
    )
    if filepath:
        # Load data
        data = tiff.imread(filepath)
        # Normalized average over entire image
        bg = np.min(data, axis=(-1,-2,-3))
        intensity = np.average(data, axis=(-1,-2,-3))-bg
        # Normalize
        # Load metadata
        with open(filepath.removesuffix('.tif') + '.yaml', 'r') as file:
            metadata = yaml.load(file, yaml.Loader)

        wl = metadata['Laser.wavelength [nm]']
        wavelens = np.linspace(wl['Start'], wl['Stop'], wl['Number'])
        calibration = np.stack((wavelens, norm(intensity)))
        np.save(calibrationfile, calibration)
        return calibration


def laser_spectrum(wavelen):
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    calibrationfile = os.path.join(current_file_path, 'calibration.npy')
    print(calibrationfile)
    if os.path.exists(calibrationfile):
        calibration = np.load(calibrationfile)
    else:
        calibration = create_calibration()
    
    # normalized interpolized spectrum
    norm = interp1d(calibration[0], calibration[1])(wavelen)
    norm = norm/np.min(norm)
    return norm
        

def open_wavelen_sweep(filepath, normalize=False):
    print('Processing Data')
    if filepath.endswith('.npy'):
        data = np.load(filepath)
    elif filepath.endswith('.tif'):
        data = tiff.imread(filepath)
    else:
        raise ValueError('Error: filepath should end in npy or tif')
    
    metadataname = filepath.removesuffix('_raw.npy').removesuffix('.tif') + '.yaml'
    if os.path.exists(metadataname):
        with open(metadataname, 'r') as file:
            metadata: dict = yaml.load(file, yaml.Loader)
    else:
        metadataname = os.path.exists(re.sub(r"_\d(?=\.yaml$)", "", metadataname))
        if os.path.exists(metadataname):
            with open(metadataname, 'r') as file:
                metadata: dict = yaml.load(file, yaml.Loader)
        else:
            raise FileNotFoundError('Error: metadata file not found')
        
    wl = metadata['Laser.wavelength [nm]']
    wavelens = np.linspace(wl['Start'], wl['Stop'], wl['Number'])

    if data.shape[1] == 4:
        background = np.array([pc.common_background(images) for images in data])
        raw = data[:,1]
        #processed = pc.background_subtracted(raw, background, False)
        processed = pc.float_to_mono(pc.background_subtracted(raw, background), normalize)
    else:
        raw = data
        bg = np.min(data, axis=(-1,-2,-3))[:,np.newaxis, np.newaxis,np.newaxis]
        davgbg = np.average(data-bg, axis=(-1,-2,-3))
        #norm = laser_spectrum(wavelens)[:,np.newaxis, np.newaxis,np.newaxis]

        spectrum = norm(davgbg[:,np.newaxis, np.newaxis,np.newaxis])
        #spectrum = laser_spectrum(wavelens)[:,np.newaxis, np.newaxis,np.newaxis]
        processed = (data-bg)/spectrum
        background = bg
    
    return wavelens, raw, processed, background

#%%

def gaussian_2d(xy, x0, y0, A, sigma,b):
    x, y = xy
    return A*np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma**2)+b

def sinc(xy, x0, y0, A, sigma):
    x, y = xy
    return A*np.sinc(np.sqrt((x-x0)**2+(y-y0)**2)/2/sigma)

def airy(xy, x0, y0, A, sigma):
    x, y = xy
    r = np.sqrt((x-x0)**2+(y-y0)**2)/2/sigma
    return A*2*jv(1, r)/r

def peak_intensities(data, peaks):
    return np.array([data[:,peak[0], peak[1]] for peak in peaks]).T

def hash_file(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    
    hash = hasher.hexdigest()[:8]
    name, ext = os.path.splitext(os.path.basename(filepath))

    return f"{name}_{hash}{ext}"

def load_measurement(filepath=None, reprocess=False):
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

    metadata: dict = raw_metadata
    
    

    # If already processed, return that
    os.makedirs('Data', exist_ok=True)
    storage_path = os.path.join('Data', hash_file(filepath))
    if os.path.exists(storage_path) and not reprocess:
        return np.load(storage_path), metadata
    

    # Process
    print(f'Processing {os.path.basename(filepath)}')
    # Load raw data
    data = np.load(filepath).squeeze()
    # Average
    if 'Camera.averaging' in metadata.keys():
        print('Averaging frames')
        count = metadata['Camera.averaging']
        image = np.average(data[:,:count],axis=1)
    else:
        image = data[:,0]

    # Background subtraction
    background = np.array([pc.common_background(im[-4:]) for im in data])
    images = pc.background_subtracted(image,background)
    print(f'Processed {os.path.basename(filepath)}')

    np.save(storage_path, images)
    return images, metadata






def identify_peaks(filepath, use_symmetry=True, threshold=None, reprocess=False, margin=20):
    """Sees if the peaks have already been identified, if not identifies them and saves to file."""
    data, metadata = load_measurement(filepath, reprocess=False)
    # Peak file path
    storage_path = os.path.join('Data', hash_file(filepath))
    name, ext = os.path.splitext(storage_path)
    peakfile = f'{name}_peaks{ext}'

    # Check if already processed
    if os.path.exists(peakfile) and not reprocess:
        peaks = np.load(peakfile)
    else:
        peaks = find_peaks(data, use_symmetry=True, threshold=threshold)
        np.save(peakfile, peaks)

    # Filter marginal peaks
    count, rows, cols = data.shape
    indeces = []
    for i, peak in enumerate(peaks):
        if peak[0] < margin or peak[0] > rows - margin or peak[1] < margin or peak[1] > cols - margin:
            indeces.append(i)
    peaks = np.delete(peaks, indeces, axis=0)
    return peaks


def find_peaks(data, use_symmetry=True, threshold=None):
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

    peaks = local_max_peaks(np.average(processed, axis=0), threshold)

    print(f'Found {peaks.shape[0]} peaks!')

    return peaks


def local_max_peaks(image, threshold = 1/20):
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