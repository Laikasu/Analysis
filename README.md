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
raw_data = data.raw()
peaks = data.peaks()
images = data.images
background = data.background
wavelens = data.wavelen()
defocus = data.defocus()
roi_size = data.roi_size()
```

Examples can be found under usage_examples.