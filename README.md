# Analysis

## Overview  
**Analysis** is a python package designed to load and analyse data collected using the experiment-control-software.

## Features  
- Effortlessly load in processed data
- Find peaks through a GUI

## Installation
```bash
pip install git+https://github.com/Laikasu/analysis.git
```

## Usage
An object is provided: ISCATDataProcessor, which contains useful functions and the data structures.
```
from analysis import ISCATDataProcessor
data = ISCATDataProcessor(filepath)

# GUIs
data.browse() # Browse data
data.find_peads() # Find peaks (re-do)
data.edit_peaks() # Edit determined peaks

# Data
raw_data = data.raw()
images = data.images # Processed images shape (sweep parameters, height, width)
background = data.background # Background shape (sweep parameters, height, width)
peaks = data.peaks() # Peak values shape (N, sweep parameters)
data.psf(size=31) # Gives images around peaks shape (N, sweep parameters, size, size)

# Metadata
wavelens = data.wavelen()
defocus = data.defocus()
roi_size = data.roi_size()
```

Examples can be found under usage_examples.