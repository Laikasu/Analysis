"""
widgets.py

Provides widgets to look at and select data.
"""

import numpy as np

from PySide6.QtWidgets import QWidget, QSlider, QLabel, QFormLayout, QVBoxLayout, QApplication, QDialogButtonBox, QCheckBox, QGroupBox, QPushButton
from PySide6.QtCore import Qt

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

from skimage.filters import difference_of_gaussians
from scipy.ndimage import gaussian_filter1d
from .processing import local_max_peaks
from .symmetry import symmetry

import os


class Browser(QWidget):
    def __init__(self, data, lock_stack=False):
        super().__init__()
        # --- Matplotlib figure ---
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_axis_off()
        self.data = data
        
        self.canvas = FigureCanvasQTAgg(self.fig)
        
        self.peaks_fromfile = None
        if self.data.has_peakfile():
            self.peaks_fromfile = data.peaks()

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.canvas, stretch=1)

        self.sliders = []
        # Stack (mostly media)
        if data._stack:
            self.stack_label = QLabel("Frame")
            stack_slider = QSlider(Qt.Orientation.Horizontal)
            stack_slider.setMinimum(1)
            stack_slider.setMaximum(len(data.images))
            stack_slider.valueChanged.connect(self.update_plot)
            self.sliders.append(stack_slider)
            self.main_layout.addWidget(self.stack_label)
            self.main_layout.addWidget(stack_slider)
            if lock_stack:
                stack_slider.setEnabled(False)



        # Defocus
        self.defocus = data.defocus()
        if isinstance(self.defocus, np.ndarray):
            # Frame slider
            self.defocus_label = QLabel(f"Defocus: {self.defocus[0]:.2f} µm")
            defocus_slider = QSlider(Qt.Orientation.Horizontal)
            defocus_slider.setMinimum(1)
            defocus_slider.setMaximum(len(self.defocus))
            defocus_slider.valueChanged.connect(self.update_defocus)
            self.sliders.append(defocus_slider)
            self.main_layout.addWidget(self.defocus_label)
            self.main_layout.addWidget(defocus_slider)
        
        # Wavelen
        self.wavelen = data.wavelen()
        if isinstance(self.wavelen, np.ndarray):
            self.wavelen_label = QLabel(f"Wavelen: {self.wavelen[0]:.0f} µm")
            wavelen_slider = QSlider(Qt.Orientation.Horizontal)
            wavelen_slider.setMinimum(1)
            wavelen_slider.setMaximum(len(self.wavelen))
            wavelen_slider.valueChanged.connect(self.update_wavelen)
            self.sliders.append(wavelen_slider)
            self.main_layout.addWidget(self.wavelen_label)
            self.main_layout.addWidget(wavelen_slider)
        
        self.setLayout(self.main_layout)
        
        # Initiate image
        frame = [slider.value()-1 for slider in self.sliders]
        self.image = data.images[*frame]
        self.imshow = self.ax.imshow(self.image, cmap='gray')
        self.sc = self.ax.scatter([], [], marker='+', color='red', linewidths=0.7, alpha=0.7)
        self.fig.tight_layout()

        if self.peaks_fromfile is not None:
            self.update_plot()
    
    def update_defocus(self, value):
        index = int(value)-1
        actual_defocus = self.defocus[index]
        self.defocus_label.setText(f"Defocus: {actual_defocus:.2f} µm")
        self.update_plot()
    
    def update_wavelen(self, value):
        index = int(value)-1
        actual_wavelen = self.wavelen[index]
        self.wavelen_label.setText(f"Wavelen: {actual_wavelen:.0f} µm")
        self.update_plot()
    
    def update_plot(self):
        frame = [slider.value()-1 for slider in self.sliders]
        self.image = self.data.images[*frame]
        if self.peaks_fromfile is not None:
            self.sc.set_offsets(self.peaks_fromfile[:,frame[0], frame[1], 0,::-1])
        self.imshow.set_data(self.image)
        self.canvas.draw_idle()




class PeakFinder(Browser):
    def __init__(self, data, use_symmetry=True, lock_stack=True):
        super().__init__(data, lock_stack=lock_stack)

        self.result: np.ndarray | None = None
        self.peaks = [np.empty((5,2))]
        self.drifts = []
        
        self.lock_stack = lock_stack
        self.use_symmetry = use_symmetry

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self._dragging_index = None
        self._starting_pos = np.array([0,0])

        fitgroup = QGroupBox('Fitting')
        groupboxlayout = QFormLayout()
        fitgroup.setLayout(groupboxlayout)

        # Thresholds
        threshold_label = QLabel("Lower threshold")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(10000)
        self.threshold_slider.setValue(1)
        self.threshold_slider.sliderReleased.connect(self.calculate_peaks)
        groupboxlayout.addRow(threshold_label, self.threshold_slider)

        upper_threshold_label = QLabel("Upper threshold")
        self.upper_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.upper_threshold_slider.setMinimum(1)
        self.upper_threshold_slider.setMaximum(10000)
        self.upper_threshold_slider.setValue(10000)
        self.upper_threshold_slider.sliderReleased.connect(self.calculate_peaks)
        groupboxlayout.addRow(upper_threshold_label, self.upper_threshold_slider)

        self.symmetry = QCheckBox()
        self.symmetry.setChecked(True)
        self.main_layout.addWidget(self.symmetry)
        self.symmetry.toggled.connect(self.calculate_peaks)
        groupboxlayout.addRow("Symmetry", self.symmetry)

        self.main_layout.addWidget(fitgroup)


        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply_peaks)
        buttons.rejected.connect(self.cancel)
        
        self.main_layout.addWidget(buttons)

    def calculate_peaks(self):
        filtered = difference_of_gaussians(self.image, 2, 4)
        # Use covolution with circle to find the psfs
        if self.symmetry.isChecked():
            processed = symmetry(filtered,np.arange(4,20,2))
        else:
            processed = filtered

        self.peaks[-1] = local_max_peaks(processed, self.threshold_slider.value()/10000, self.upper_threshold_slider.value()/10000)

        # Display
        self.update_peaks()
    
    def calculate_drift(self, dist=5.0):
        # Correct based on previous rather than first
        
        # Calculate peaks
        filtered = difference_of_gaussians(self.image, 2, 4)
        if self.use_symmetry:
            processed = symmetry(filtered,np.arange(4,20,2))
        else:
            processed = filtered

        new_peaks = local_max_peaks(processed, self.threshold_slider.value()/10000, self.upper_threshold_slider.value()/10000)

        d_max = dist
        # compute distance matrix
        D = cdist(self.peaks[-1], new_peaks)
        # penalize large distances
        D[D > d_max] = 1e6
        row_ind, col_ind = linear_sum_assignment(D)

        matches = []
        for i, j in zip(row_ind, col_ind):
            if D[i, j] < d_max:
                matches.append((i, j))

        displacement = np.mean([new_peaks[j] - self.peaks[-1][i] for i, j in matches],axis=0)
        
        return displacement
    
    def calculate_defocus_drift(self):
        frame = [slider.value()-1 for slider in self.sliders]
        # Calculate peaks
        drift = np.zeros((len(self.data.images[1]), 2), dtype=np.int16)
        for k, image in enumerate(self.data.images[frame[0], :,frame[2]]):
            
            filtered = difference_of_gaussians(image, 2, 4)
            if self.use_symmetry:
                processed = symmetry(filtered,np.arange(4,10,2))
            else:
                processed = filtered

            new_peaks = local_max_peaks(processed, self.threshold_slider.value()/10000, self.upper_threshold_slider.value()/10000)

            d_max = 10.0

            # compute distance matrix
            D = cdist(self.peaks[-1], new_peaks)

            # penalize large distances
            D[D > d_max] = 1e6

            row_ind, col_ind = linear_sum_assignment(D)

            matches = []

            for i, j in zip(row_ind, col_ind):
                if D[i, j] < d_max:
                    matches.append((i, j))
            
            drift[k] = np.mean([new_peaks[j] - self.peaks[-1][i] for i, j in matches],axis=0)

        return drift
    
    def update_peaks(self):
        if len(self.peaks[-1]) == 0:
            offsets = np.empty((0, 2))
        else:
            offsets = self.peaks[-1].reshape(-1, 2)[:, [1, 0]]
        self.sc.set_offsets(offsets)
        self.canvas.draw_idle()
    
    def cancel(self):
        self.result = None
        self.close()
    
    def apply_peaks(self):
        if not self.lock_stack:
            self.result = self.peaks[0]
            self.close()

        
        self.drifts.append(self.calculate_defocus_drift())

        # Loop
        i = self.sliders[0].value()

        if i == len(self.data.images):
            # Peaks: (P,2) drift (n, 2)
            peaks  = np.array(self.peaks)
            drifts = np.array(self.drifts)
            print(drifts.shape)
            n, d, _ = drifts.shape
            n, N, _ = peaks.shape
            self.result = peaks.reshape(n, N, 1, 1, 2) + drifts.reshape(n, 1, d, 1, 2)
            self.result = self.result.swapaxes(0, 1)
            self.close()
        
        self.sliders[0].setValue(i+1)
        displacement = self.calculate_drift(dist=10.0)
        self.peaks.append(self.peaks[-1] + displacement)
        self.update_peaks()
        

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        # Find the closest point
        xy = self.sc.get_offsets()
        x, y = xy[:,0], xy[:,1]
        distances = np.hypot(x - event.xdata, y - event.ydata)
        if len(distances) == 0:
            return
        min_idx = np.argmin(distances)

        # Shift all
        if event.button == 3:
            self._starting_pos: np.ndarray | None = np.array([int(event.ydata), int(event.xdata)])

        if distances[min_idx] < 5:  # threshold for picking a point
            if event.button == 2:
                self.peaks[-1] = np.delete(self.peaks[-1], min_idx, axis=0)
                self.update_peaks()
            if event.button == 1:
                self._dragging_index = min_idx
            
            
        
        if distances[min_idx] > 20 and event.button == 1:
            self.peaks[-1] = np.vstack([self.peaks[-1], [int(event.ydata), int(event.xdata)]])
            self.update_peaks()
            self._dragging_index = -1
        
       

        
    def on_motion(self, event):
        if self._dragging_index is not None and event.button == 1:
            self.peaks[-1][self._dragging_index] = [int(event.ydata), int(event.xdata)]
            self.update_peaks()
        if event.button == 3 and self._starting_pos is not None:
            print(self.peaks[-1][:,::-1])
            self.sc.set_offsets(self.peaks[-1][:,::-1] + np.array([int(event.xdata), int(event.ydata)]) - self._starting_pos[::-1])
            self.canvas.draw_idle()
        
    
    def on_release(self, event):
        if event.button == 3:
            self.peaks[-1] = self.peaks[-1] + np.array([int(event.ydata), int(event.xdata)]) - self._starting_pos
            self.update_peaks()
            self._starting_pos = None
        if event.button == 1:
            self._dragging_index = None






class PeakEditor(Browser):
    def __init__(self, data):
        super().__init__(data, lock_stack=False)

        self.sensitivity = 10

        self.result: np.ndarray | None = None
        self.drifts = []

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self._dragging_index = None
        self._starting_pos = np.array([0,0])

        self.smoothbutton = QPushButton('Smooth')
        self.smoothbutton.clicked.connect(self.smooth)
        self.main_layout.addWidget(self.smoothbutton)


        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply_peaks)
        buttons.rejected.connect(self.cancel)
        
        self.main_layout.addWidget(buttons)
    

    def update_peaks(self):
        frame = [slider.value()-1 for slider in self.sliders]
        if len(self.peaks_fromfile[:,frame[0], frame[1], 0]) == 0:
            offsets = np.empty((0, 2))
        else:
            offsets = self.peaks_fromfile[:,frame[0], frame[1], 0].reshape(-1, 2)[:, [1, 0]]
        
        self.sc.set_offsets(offsets)
        self.canvas.draw_idle()
    
    def cancel(self):
        self.close()
    
    def apply_peaks(self):
        storage_path = os.path.join('Data', self.data._hash_file(self.data.filepath))
        name, ext = os.path.splitext(storage_path)
        peakfile = f'{name}_peaks.npy'
        np.save(peakfile, self.peaks_fromfile)
        self.close()
    
    def smooth(self):
        N, n, f, w, p = self.peaks_fromfile.shape
        displacement = self.peaks_fromfile - self.peaks_fromfile[:,0,0,0].reshape(-1,1,1,1,2)
        d = np.mean(displacement, axis=0).squeeze().reshape(-1,2)

        d0 = d[:,0]
        d1 = d[:,1]
        disp = np.column_stack((gaussian_filter1d(d0, sigma=2, mode='reflect'), gaussian_filter1d(d1, sigma=2, mode='reflect'))).reshape(1, n, f, 1, 2)
        self.peaks_fromfile = self.peaks_fromfile[:,0,0,0].reshape(-1,1,1,1,2) + disp
        self.update_peaks()
        

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        # Find the closest point
        xy = self.sc.get_offsets()
        x, y = xy[:,0], xy[:,1]
        distances = np.hypot(x - event.xdata, y - event.ydata)
        if len(distances) == 0:
            return
        min_idx = np.argmin(distances)

        # Shift all
        if event.button == 3:
            self._starting_pos: np.ndarray | None = np.array([int(event.ydata), int(event.xdata)])

        if distances[min_idx] < 5:  # threshold for picking a point
            if event.button == 2:
                self.peaks_fromfile = np.delete(self.peaks_fromfile, min_idx, axis=0)
                self.update_peaks()
            if event.button == 1:
                self._dragging_index = min_idx
        
        
        # if distances[min_idx] > 10 and event.button == 1:
        #     self.peaks[-1] = np.vstack([self.peaks[-1], [int(event.ydata), int(event.xdata)]])
        #     self.update_peaks()
        #     self._dragging_index = -1

    
    def on_motion(self, event):
        frame = [slider.value()-1 for slider in self.sliders]
        if self._dragging_index is not None and event.button == 1:
            self.peaks_fromfile[self._dragging_index, frame[0], frame[1], 0] = [int(event.ydata), int(event.xdata)]
            self.update_peaks()
        if event.button == 3 and self._starting_pos is not None:
            self.sc.set_offsets(self.peaks_fromfile[:, frame[0], frame[1], 0, ::-1] + (np.array([int(event.xdata), int(event.ydata)]) - self._starting_pos[::-1])/self.sensitivity)
            self.canvas.draw_idle()
        
    
    def on_release(self, event):
        frame = [slider.value()-1 for slider in self.sliders]
        if event.button == 3:
            self.peaks_fromfile[:, frame[0], frame[1],0] += (np.array([int(event.ydata), int(event.xdata)]) - self._starting_pos)/self.sensitivity
            self.update_peaks()
            self._starting_pos = None
        if event.button == 1:
            self._dragging_index = None







app = QApplication()

def start_browser(data):
    app.setApplicationName("Browser")
    app.setApplicationDisplayName("Browser")
    app.setStyle("fusion")

    w = Browser(data)
    w.show()

    app.exec()


def start_peakfinder(data, use_symmetry=True, lock_stack=-1):
    app.setApplicationName("Peak Finder")
    app.setApplicationDisplayName("Peak Finder")
    app.setStyle("fusion")

    w = PeakFinder(data, use_symmetry, lock_stack=lock_stack)
    w.show()

    app.exec()
    return w.result

def peak_editor(data):
    app.setApplicationName("Peak Editor")
    app.setApplicationDisplayName("Peak Editor")
    app.setStyle("fusion")

    w = PeakEditor(data)
    w.show()

    app.exec()
    return 0