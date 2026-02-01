import sys
from analysis import ISCATDataProcessor


# Command line argument allowed, otherwise a file dialog will appear
filename = sys.argv[1] if len(sys.argv) > 1 else None
data = ISCATDataProcessor(filename)
data.edit_peaks()