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
Two functions are provided: load_measurement and identify_peaks.
```
import analysis as al
data, metadata = al.load_measurement(filepath)
peaks = al.identify_peaks(filepath)
```

The metadata is a dictionary. To see the available keys you can look at the metadata file.

```
import analysis as al
data, metadata = al.load_measurement(filepath)
peaks = al.identify_peaks(filepath)
```

Examples can be found under usage_examples.