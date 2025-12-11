"""
Harmonic Layer - Minimal harmonic annotation tool
Based on base_layer.py
Focuses on F0 detection and harmonic manipulation
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import json

from layers.base_layer import BaseLayer

# try:
#     from yaaat import utils
# except ImportError:
#     import utils

from utils import utils 

class HarmonicLayer(BaseLayer):
    """Simplified harmonic annotator focusing on F0 and harmonic lines"""
    
    def __init__(self, root):
        
        # Initialize custom attributes before parent init (setup_ui -> setup_custom_controls)
        self.f0_line = None  # Horizontal line for F0
        self.harmonic_lines = []  # List of {freq: float, num: int, line: Line2D}
        self.selected_line = None  # Currently selected line for dragging
        self.drag_start_y = None
        
        # Display options
        self.show_ridges = tk.BooleanVar(value=True)
        self.show_valleys = tk.BooleanVar(value=False)


        # Contour display & settings
        self.show_contour = tk.BooleanVar(value=False)
        self.contour_method = tk.StringVar(value='raw')   # 'raw', 'smooth', 'poly', 'spline'
        self.contour_smoothness = tk.DoubleVar(value=5.0) # window size in frames
        self.harmonic_contours = {}  # {harmonic_num: [(time, freq), ...]}


        # Detection parameters
        self.prominence = tk.DoubleVar(value=5.0)
        self.freq_min = tk.IntVar(value=500)
        self.freq_max = tk.IntVar(value=8000)
        
        # Detected data
        self.mean_spectrum = None
        self.detected_f0 = None
        self.harmonic_ridges = {}  # {harmonic_num: [(time, freq), ...]}
        self.valley_ridges = {}    # {valley_name: [(time, freq), ...]}
        self.ridge_method = tk.StringVar(value='max')  # 'max', 'peaks', 'centroid'
        

        # Call parent init (calls setup_ui -> setup_custom_controls)
        super().__init__(root)
        
        # Update title if it's a root window
        if isinstance(root, tk.Tk):
            self.root.title("Harmonic Layer - YAAAT")
    


    def setup_custom_controls(self):
        """Add harmonic-specific controls to the panel"""
        # Detection parameters
        ttk.Label(self.control_panel, text="F0 Detection:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        # Frequency range
        freq_frame = ttk.Frame(self.control_panel)
        freq_frame.pack(fill=tk.X, pady=2)
        ttk.Label(freq_frame, text="Range (Hz):", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.freq_min, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(freq_frame, text="-", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.freq_max, width=5).pack(side=tk.LEFT, padx=2)
        
        # Prominence slider
        prom_frame = ttk.Frame(self.control_panel)
        prom_frame.pack(fill=tk.X, pady=2)
        ttk.Label(prom_frame, text="Prominence:", font=('', 8)).pack(side=tk.LEFT)
        ttk.Scale(prom_frame, from_=0.1, to=20, variable=self.prominence, 
                orient=tk.HORIZONTAL, command=self.on_prominence_change).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.prom_label = ttk.Label(prom_frame, text=f"{self.prominence.get():.1f}", font=('', 8), width=5)
        self.prom_label.pack(side=tk.LEFT)
        
        ttk.Button(self.control_panel, text="Detect F0", command=self.detect_f0).pack(fill=tk.X, pady=2)
        
        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # Harmonic controls - no dialogs!
        ttk.Label(self.control_panel, text="Add/Remove Harmonics:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        


        # Harmonic toggle buttons
        harm_toggle_frame = ttk.Frame(self.control_panel)
        harm_toggle_frame.pack(fill=tk.X, pady=2)
        ttk.Label(harm_toggle_frame, text="Toggle:", font=('', 8)).pack(side=tk.LEFT)

        # Create toggle buttons for harmonics 1-5
        self.harmonic_buttons = {}
        for h in [1, 2, 3, 4, 5]:
            btn = tk.Button(harm_toggle_frame, text=f"H{h}", width=4, 
                        relief=tk.RAISED, bg='lightgray',
                        command=lambda n=h: self.toggle_harmonic(n))
            btn.pack(side=tk.LEFT, padx=2)
            self.harmonic_buttons[h] = btn
        


        ttk.Button(self.control_panel, text="Clear All (except F0)", command=self.clear_all_harmonics).pack(fill=tk.X, pady=2)
        
        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        
        # Display options
        ttk.Label(self.control_panel, text="Display:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Checkbutton(self.control_panel, text="Show Ridges", variable=self.show_ridges,
                    command=lambda: self.update_display()).pack(anchor=tk.W)
        ttk.Checkbutton(self.control_panel, text="Show Valleys", variable=self.show_valleys,
                    command=lambda: self.update_display()).pack(anchor=tk.W)
        ttk.Checkbutton(self.control_panel, text="Show Contour", variable=self.show_contour,
                    command=self.on_show_contour_toggle).pack(anchor=tk.W)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)


        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # Ridge detection method
        ttk.Label(self.control_panel, text="Ridge Detection Method:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        ridge_method_frame = ttk.Frame(self.control_panel)
        ridge_method_frame.pack(fill=tk.X, pady=2)

        ttk.Radiobutton(ridge_method_frame, text="Max", variable=self.ridge_method, 
                        value='max', command=self.on_ridge_method_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(ridge_method_frame, text="Peaks", variable=self.ridge_method, 
                        value='peaks', command=self.on_ridge_method_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(ridge_method_frame, text="Centroid", variable=self.ridge_method, 
                        value='centroid', command=self.on_ridge_method_change).pack(side=tk.LEFT, padx=5)

        # Tolerance slider (only active for peaks method)
        tol_frame = ttk.Frame(self.control_panel)
        tol_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tol_frame, text="Peak Tolerance:", font=('', 8)).pack(side=tk.LEFT)
        self.peak_tolerance = tk.DoubleVar(value=0.1)
        self.tol_scale = ttk.Scale(tol_frame, from_=0.01, to=0.5, variable=self.peak_tolerance,orient=tk.HORIZONTAL, command=self.on_tolerance_change)
        self.tol_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.tol_label = ttk.Label(tol_frame, text="0.10", font=('', 8), width=5)
        self.tol_label.pack(side=tk.LEFT)

        # Contour extraction methods
        ttk.Label(self.control_panel, text="Contour Extraction:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(4, 2))

        contour_frame = ttk.Frame(self.control_panel)
        contour_frame.pack(fill=tk.X, pady=2)

        ttk.Radiobutton(contour_frame, text="Raw", variable=self.contour_method,
                        value='raw', command=self.on_contour_method_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(contour_frame, text="Smooth", variable=self.contour_method,
                        value='smooth', command=self.on_contour_method_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(contour_frame, text="PolyFit", variable=self.contour_method,
                        value='poly', command=self.on_contour_method_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(contour_frame, text="Spline", variable=self.contour_method,
                        value='spline', command=self.on_contour_method_change).pack(side=tk.LEFT, padx=5)

        # Smoothness slider (used for smooth/poly)
        contour_smooth_frame = ttk.Frame(self.control_panel)
        contour_smooth_frame.pack(fill=tk.X, pady=2)
        ttk.Label(contour_smooth_frame, text="Contour Smoothness:", font=('', 8)).pack(side=tk.LEFT)
        self.contour_smooth_scale = ttk.Scale(
            contour_smooth_frame, from_=1.0, to=20.0,
            variable=self.contour_smoothness,
            orient=tk.HORIZONTAL,
            command=self.on_contour_smoothness_change
        )
        self.contour_smooth_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Current harmonics list
        ttk.Label(self.control_panel, text="Active Harmonics:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.harmonics_listbox = tk.Listbox(self.control_panel, height=5, font=('', 8))
        self.harmonics_listbox.pack(fill=tk.X, pady=2)
        
        # Info display
        self.info_label = ttk.Label(self.control_panel, text="No F0 detected", wraplength=300, font=('', 8))
        self.info_label.pack(fill=tk.X, pady=2)


    
    def process_audio(self):
        """Auto-detect F0 when audio is loaded"""
        if self.y is not None:
            self.detect_f0()
    
    def detect_f0(self):
        """Detect fundamental frequency from mean spectrum"""
        if self.S_db is None:
            return
        
        # Compute mean spectrum
        self.mean_spectrum = np.mean(self.S_db, axis=1)
        
        # Find peaks in frequency range
        freq_mask = (self.freqs >= self.freq_min.get()) & (self.freqs <= self.freq_max.get())
        masked_spectrum = self.mean_spectrum.copy()
        masked_spectrum[~freq_mask] = -np.inf
        
        peaks, properties = find_peaks(masked_spectrum, 
                                     prominence=self.prominence.get(),
                                     distance=max(5, int(50 / (self.freqs[1] - self.freqs[0]))))
        
        if len(peaks) == 0:
            self.detected_f0 = None
            self.info_label.config(text="No peaks found")
            return
        
        # Use strongest peak as F0
        peak_mags = masked_spectrum[peaks]
        strongest_idx = peaks[np.argmax(peak_mags)]
        self.detected_f0 = self.freqs[strongest_idx]
        
        # Clear existing harmonics
        self.harmonic_lines = []
        
        # Add F0 as first harmonic
        self.harmonic_lines.append({
            'freq': self.detected_f0,
            'num': 1,
            'line': None
        })
        
        # Detect harmonic ridges if enabled
        if self.show_ridges.get():
            self.detect_harmonic_ridges()
        
        self.update_display()
        self.update_info()
        self.update_button_states()

        print(f"✓ Detected F0: {self.detected_f0:.1f} Hz")
    


    ################################################################
    ######## DETECTION LOGIC #######################################
    ################################################################


    def detect_harmonic_ridges(self):
        """Detect time-varying harmonic contours using selected method"""
        if self.detected_f0 is None or self.S_db is None:
            return
        
        method = self.ridge_method.get()
        
        # Dispatch to appropriate method
        if method == 'max':
            self.harmonic_ridges = self.detect_ridges_max()
        elif method == 'peaks':
            self.harmonic_ridges = self.detect_ridges_peaks()
        elif method == 'centroid':
            self.harmonic_ridges = self.detect_ridges_centroid()
        elif method == 'parabolic':
            self.harmonic_ridges = self.detect_ridges_parabolic()
        
        # Recompute contours when ridges change
        if self.show_contour.get():
            self.compute_contours()


    def detect_ridges_max(self):
        """Simple maximum search within frequency window"""
        ridges = {}
        
        for h in self.harmonic_lines:
            harm_num = h['num']
            expected_freq = h['freq']
            ridge = []
            tolerance = expected_freq * 0.1  # Fixed 10% tolerance
            
            for t_idx in range(self.S_db.shape[1]):
                spectrum = self.S_db[:, t_idx]
                
                # Search window
                freq_mask = (self.freqs >= expected_freq - tolerance) & \
                        (self.freqs <= expected_freq + tolerance)
                
                if np.any(freq_mask):
                    search_indices = np.where(freq_mask)[0]
                    local_max_idx = np.argmax(spectrum[freq_mask])
                    freq_idx = search_indices[local_max_idx]
                    ridge.append((self.times[t_idx], self.freqs[freq_idx]))
            
            if ridge:
                ridges[harm_num] = ridge
        
        return ridges

    def detect_ridges_peaks(self):
        """Peak detection with prominence and tolerance"""
        from scipy.signal import find_peaks
        ridges = {}
        
        for h in self.harmonic_lines:
            harm_num = h['num']
            expected_freq = h['freq']
            ridge = []
            tolerance = expected_freq * self.peak_tolerance.get()
            
            for t_idx in range(self.S_db.shape[1]):
                spectrum = self.S_db[:, t_idx]
                
                # Search window
                freq_mask = (self.freqs >= expected_freq - tolerance) & \
                        (self.freqs <= expected_freq + tolerance)
                
                if np.any(freq_mask):
                    search_indices = np.where(freq_mask)[0]
                    local_spectrum = spectrum[freq_mask]
                    
                    # Find peaks with prominence
                    peaks, _ = find_peaks(local_spectrum, 
                                        prominence=self.prominence.get())
                    
                    if len(peaks) > 0:
                        # Choose peak closest to expected frequency
                        peak_freqs = self.freqs[search_indices[peaks]]
                        closest_idx = np.argmin(np.abs(peak_freqs - expected_freq))
                        freq_idx = search_indices[peaks[closest_idx]]
                        ridge.append((self.times[t_idx], self.freqs[freq_idx]))
            
            if ridge:
                ridges[harm_num] = ridge
        
        return ridges


    def detect_ridges_centroid(self):
        """Spectral centroid within frequency window"""
        ridges = {}
        
        for h in self.harmonic_lines:
            harm_num = h['num']
            expected_freq = h['freq']
            ridge = []
            tolerance = expected_freq * 0.15  # Fixed 15% tolerance for centroid
            
            for t_idx in range(self.S_db.shape[1]):
                spectrum = self.S_db[:, t_idx]
                
                # Search window
                freq_mask = (self.freqs >= expected_freq - tolerance) & \
                        (self.freqs <= expected_freq + tolerance)
                
                if np.any(freq_mask):
                    search_indices = np.where(freq_mask)[0]
                    local_spectrum = spectrum[freq_mask]
                    
                    # Convert from dB to linear for centroid
                    linear_spectrum = 10 ** (local_spectrum / 20)
                    freq_values = self.freqs[search_indices]
                    
                    # Calculate weighted centroid
                    total_energy = np.sum(linear_spectrum)
                    if total_energy > 0:
                        centroid_freq = np.sum(freq_values * linear_spectrum) / total_energy
                        ridge.append((self.times[t_idx], centroid_freq))
            
            if ridge:
                ridges[harm_num] = ridge
        
        return ridges


    def detect_ridges_parabolic(self):
        """Parabolic interpolation around spectral peak"""
        ridges = {}
        
        for h in self.harmonic_lines:
            harm_num = h['num']
            expected_freq = h['freq']
            ridge = []
            tolerance = expected_freq * 0.1
            
            for t_idx in range(self.S_db.shape[1]):
                spectrum = self.S_db[:, t_idx]
                
                freq_mask = (self.freqs >= expected_freq - tolerance) & \
                        (self.freqs <= expected_freq + tolerance)
                
                if np.any(freq_mask):
                    search_indices = np.where(freq_mask)[0]
                    local_spectrum = spectrum[freq_mask]
                    
                    # Find peak
                    peak_idx = np.argmax(local_spectrum)
                    
                    # Parabolic interpolation if not at edge
                    if 0 < peak_idx < len(local_spectrum) - 1:
                        y1 = local_spectrum[peak_idx - 1]
                        y2 = local_spectrum[peak_idx]
                        y3 = local_spectrum[peak_idx + 1]
                        
                        # Parabolic peak offset
                        p = (y3 - y1) / (2 * (2 * y2 - y1 - y3))
                        
                        # Interpolated frequency
                        freq_idx = search_indices[peak_idx]
                        freq_step = self.freqs[1] - self.freqs[0]
                        interp_freq = self.freqs[freq_idx] + p * freq_step
                        
                        ridge.append((self.times[t_idx], interp_freq))
            
            if ridge:
                ridges[harm_num] = ridge
        
        return ridges





    # = # = # = # = # = # = # = # = # = # = # = # = 
    # Contour extraction API    # = # = # = # = # = 
    # = # = # = # = # = # = # = # = # = # = # = # = 

    def on_show_contour_toggle(self):
        """Callback when 'Show Contour' is toggled."""
        if self.show_contour.get():
            self.compute_contours()
        self.update_display()

    def on_contour_method_change(self):
        """Handle change of contour method radio buttons."""
        if self.show_contour.get():
            self.compute_contours()
            self.update_display()

    def on_contour_smoothness_change(self, value):
        """Handle contour smoothness slider change."""
        if self.show_contour.get():
            self.compute_contours()
            self.update_display()

    def compute_contours(self):
        """Compute harmonic contours from current harmonic_ridges."""
        self.harmonic_contours = {}
        if not self.harmonic_ridges:
            return

        method = self.contour_method.get()

        for harm_num, ridge in self.harmonic_ridges.items():
            if not ridge:
                continue

            times, freqs = zip(*ridge)
            times = np.asarray(times)
            freqs = np.asarray(freqs)

            if method == 'raw':
                contour_freqs = freqs
            elif method == 'smooth':
                contour_freqs = self._smooth_contour(freqs, int(self.contour_smoothness.get()))
            elif method == 'poly':
                contour_freqs = self._polyfit_contour(times, freqs, int(self.contour_smoothness.get()))
            elif method == 'spline':
                contour_freqs = self._spline_contour(times, freqs, int(self.contour_smoothness.get()))
            else:
                contour_freqs = freqs

            self.harmonic_contours[harm_num] = list(zip(times, contour_freqs))

    def _smooth_contour(self, freqs, window):
        """Simple moving-average smoothing over frequency contour."""
        window = max(1, int(window))
        if window == 1 or len(freqs) < 3:
            return freqs

        kernel = np.ones(window) / window
        padded = np.pad(freqs, (window // 2, window - 1 - window // 2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed

    def _polyfit_contour(self, times, freqs, order_hint):
        """Polynomial fit of frequency vs time.

        order_hint is mapped to a low polynomial order to avoid overfitting.
        """
        if len(times) < 3:
            return freqs

        # Map smoothness slider to poly order 1–3
        if order_hint < 5:
            order = 1
        elif order_hint < 10:
            order = 2
        else:
            order = 3

        # Normalize time to improve conditioning
        t0 = times.mean()
        ts = times - t0

        try:
            coeffs = np.polyfit(ts, freqs, order)
            poly = np.poly1d(coeffs)
            return poly(ts)
        except np.linalg.LinAlgError:
            # Fallback if fit fails
            return freqs


    def _spline_contour(self, times, freqs, smooth_hint):
        """Spline-based smoothing of frequency vs time.

        smooth_hint comes from the slider and is mapped to a spline
        smoothing parameter.
        """
        if len(times) < 3:
            return freqs

        # Normalize time to improve conditioning
        t0 = times.mean()
        ts = times - t0

        # Map smooth_hint (e.g. 1–20) to a reasonable smoothing factor.
        # Larger -> smoother. This is heuristic; tweak if needed.
        smooth_hint = max(1, int(smooth_hint))
        s = smooth_hint * np.var(freqs) * 0.1

        try:
            spline = UnivariateSpline(ts, freqs, s=s)
            return spline(ts)
        except Exception:
            # If spline fails for any reason, fall back to original data
            return freqs




    def on_ridge_method_change(self):
        """Handle ridge detection method change"""
        method = self.ridge_method.get()
        print(f"✓ Switched to {method} ridge detection")
        
        # Enable/disable tolerance slider based on method
        if hasattr(self, 'tol_scale'):
            if method == 'peaks':
                self.tol_scale.configure(state='normal')
            else:
                self.tol_scale.configure(state='disabled')
        
        # Re-detect with new method
        if self.harmonic_lines:
            self.detect_harmonic_ridges()
            if self.show_contour.get():
                self.compute_contours()
            self.update_display()








    def on_tolerance_change(self, value):
        """Handle tolerance slider change"""
        if hasattr(self, 'tol_label'):
            self.tol_label.config(text=f"{self.peak_tolerance.get():.2f}")
        
        # Re-detect if using peaks method
        if self.ridge_method.get() == 'peaks' and self.harmonic_lines:
            self.detect_harmonic_ridges()
            if self.show_contour.get():
                self.compute_contours()
            self.update_display()





    def toggle_harmonic(self, harm_num):
        """Toggle harmonic on/off"""
        if self.detected_f0 is None:
            messagebox.showinfo("No F0", "Detect F0 first")
            return
        
        # Check if harmonic exists
        exists = any(h['num'] == harm_num for h in self.harmonic_lines)
        
        if exists:
            # Remove it
            self.remove_harmonic(harm_num)
            # Update button appearance
            self.harmonic_buttons[harm_num].config(relief=tk.RAISED, bg='lightgray')
        else:
            # Add it
            self.add_harmonic(harm_num)
            # Update button appearance
            self.harmonic_buttons[harm_num].config(relief=tk.SUNKEN, bg='lightgreen')
        
        self.update_button_states()

    def update_button_states(self):
        """Update toggle button appearances based on active harmonics"""
        for harm_num, btn in self.harmonic_buttons.items():
            exists = any(h['num'] == harm_num for h in self.harmonic_lines)
            if exists:
                btn.config(relief=tk.SUNKEN, bg='lightgreen')
            else:
                btn.config(relief=tk.RAISED, bg='lightgray')




    def draw_custom_overlays(self):
        """Draw harmonic lines and ridges"""
        
        # Draw horizontal harmonic lines
        for h in self.harmonic_lines:
            color = self.get_harmonic_color(h['num'])
            h['line'] = self.ax.axhline(h['freq'], color=color, linewidth=1, 
                                       alpha=0.5, label=f"H{h['num']}")

            self.ax.text(
                self.ax.get_xlim()[1],          # right edge of plot
                h['freq'],                      # y-position
                f"{h['freq']:.1f} Hz",          # label text
                color=color,
                fontsize=7,
                va='bottom',
                ha='right',
                alpha=0.8,
            )
        
        # Draw harmonic ridges if enabled
        if self.show_ridges.get() and self.harmonic_ridges:
            for harm_num, ridge in self.harmonic_ridges.items():
                if ridge:
                    times, freqs = zip(*ridge)
                    color = self.get_harmonic_color(harm_num)
                    self.ax.plot(times, freqs, color=color, linewidth=1, 
                               alpha=0.5, linestyle='--')
        
        # Draw valley ridges if enabled  
        if self.show_valleys.get() and self.valley_ridges:
            valley_colors = ['cyan', 'magenta', 'lime']
            for i, (valley_name, ridge) in enumerate(self.valley_ridges.items()):
                if ridge:
                    times, freqs = zip(*ridge)
                    color = valley_colors[i % len(valley_colors)]
                    self.ax.plot(times, freqs, color=color, linewidth=1,
                               alpha=0.5, linestyle=':')

        # Draw harmonic contours on top
        if self.show_contour.get() and self.harmonic_contours:
            for harm_num, contour in self.harmonic_contours.items():
                if contour:
                    times, freqs = zip(*contour)
                    color = self.get_harmonic_color(harm_num)
                    self.ax.plot(times, freqs, color=color, alpha=0.5, linewidth=1)
        
        # Add global annotation overlays (from API)
        self.draw_shared_point_annotations()

    
    def get_harmonic_color(self, harm_num):
        """Get consistent color for harmonic number"""
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown']
        return colors[(harm_num - 1) % len(colors)]


    def on_custom_press(self, event):
        """Handle clicks on harmonic lines - start dragging immediately"""
        if event.button != 1:  # Only left click
            return False
        
        # Check if click is near a harmonic line
        clicked_line = None
        min_dist = 100  # Hz threshold (increased for easier clicking)
        
        for h in self.harmonic_lines:
            if event.ydata is not None:
                dist = abs(event.ydata - h['freq'])
                if dist < min_dist:
                    min_dist = dist
                    clicked_line = h
        
        if clicked_line:
            # Start dragging immediately
            self.start_drag_line(clicked_line)
            return True
        
        return False


    def show_harmonic_menu(self, event, harmonic):
        """Show context menu for harmonic line"""
        menu = tk.Menu(self.root, tearoff=0)
        
        menu.add_command(label=f"H{harmonic['num']}: {harmonic['freq']:.1f} Hz", 
                        state=tk.DISABLED)
        menu.add_separator()
        menu.add_command(label="Add Harmonic Above", 
                        command=lambda: self.add_harmonic(harmonic['num'] + 1))
        menu.add_command(label="Remove This Harmonic", 
                        command=lambda: self.remove_harmonic(harmonic['num']))
        menu.add_separator()
        menu.add_command(label="Start Dragging", 
                        command=lambda: self.start_drag_line(harmonic))
        
        # Convert matplotlib coords to tkinter
        x = self.canvas.winfo_rootx() + int(event.x)
        y = self.canvas.winfo_rooty() + int(self.canvas.winfo_height() - event.y)
        
        menu.post(x, y)
    
    def add_harmonic(self, position):
        """Add harmonic at specified position"""
        if self.detected_f0 is None:
            return
        
        # Check if harmonic already exists
        for h in self.harmonic_lines:
            if h['num'] == position:
                messagebox.showinfo("Exists", f"H{position} already exists")
                return
        
        # Add new harmonic
        new_freq = self.detected_f0 * position
        self.harmonic_lines.append({
            'freq': new_freq,
            'num': position,
            'line': None
        })
        
        # Sort by harmonic number
        self.harmonic_lines.sort(key=lambda h: h['num'])
        
        # Re-detect ridges
        if self.show_ridges.get():
            self.detect_harmonic_ridges()
        
        self.update_display()
        self.update_info()
        self.changes_made = True
        print(f"✓ Added H{position} at {new_freq:.1f} Hz")
    
    def remove_harmonic(self, harm_num):
        """Remove specified harmonic"""
        if harm_num == 1:
            messagebox.showinfo("Cannot Remove", "Cannot remove fundamental (H1)")
            return
        
        self.harmonic_lines = [h for h in self.harmonic_lines if h['num'] != harm_num]
        
        # Remove from ridges
        if harm_num in self.harmonic_ridges:
            del self.harmonic_ridges[harm_num]
        
        self.update_display()
        self.update_info()
        self.changes_made = True
        print(f"✓ Removed H{harm_num}")





    def quick_add_harmonic(self, harm_num):
        """Quick add a specific harmonic number"""
        if self.detected_f0 is None:
            messagebox.showinfo("No F0", "Detect F0 first")
            return
        
        self.add_harmonic(harm_num)
        self.update_harmonics_list()

    def remove_selected_harmonic(self):
        """Remove harmonic selected in spinbox"""
        try:
            harm_num = int(self.remove_spinbox.get())
            self.remove_harmonic(harm_num)
            self.update_harmonics_list()
        except ValueError:
            pass

    def clear_all_harmonics(self):
        """Clear all harmonics except fundamental"""
        if len(self.harmonic_lines) <= 1:
            messagebox.showinfo("No Harmonics", "No harmonics to clear (keeping F0)")
            return
        
        if messagebox.askyesno("Clear Harmonics", 
                            f"Remove {len(self.harmonic_lines)-1} harmonics?\n"
                            f"(keeping F0)"):
            # Keep only H1
            self.harmonic_lines = [h for h in self.harmonic_lines if h['num'] == 1]
            
            # Clear ridges except H1
            self.harmonic_ridges = {1: self.harmonic_ridges.get(1, [])}
            
            self.update_display()
            self.update_info()
            self.changes_made = True
            print("✓ Cleared all harmonics except F0")

    def update_harmonics_list(self):
        """Update the harmonics listbox"""
        if hasattr(self, 'harmonics_listbox'):
            self.harmonics_listbox.delete(0, tk.END)
            for h in sorted(self.harmonic_lines, key=lambda x: x['num']):
                self.harmonics_listbox.insert(tk.END, f"H{h['num']}: {h['freq']:.1f} Hz")





    def start_drag_line(self, harmonic):
        """Start dragging a harmonic line"""
        self.selected_line = harmonic
        self.drag_start_y = harmonic['freq']
        print(f"Started dragging H{harmonic['num']}")


    def on_custom_motion(self, event):
        """Handle dragging harmonic lines"""
        if self.selected_line is None:
            return False
        
        # Update line frequency only if we have valid y data
        if event.ydata is not None:
            self.selected_line['freq'] = event.ydata
            self.update_display()
        
        return True


    def on_custom_release(self, event):
        """Finish dragging harmonic line"""
        if self.selected_line is None:
            return False
        
        # Record the change
        old_freq = self.drag_start_y
        new_freq = self.selected_line['freq']
        
        if abs(new_freq - old_freq) > 1.0:
            self.changes_made = True
            print(f"✓ Moved H{self.selected_line['num']} from {old_freq:.1f} to {new_freq:.1f} Hz")
            

            if self.selected_line['num'] == 1:
                # Update detected F0 to the new frequency
                self.detected_f0 = new_freq

                # Scale all other harmonics to preserve harmonic ratios
                if old_freq > 0:
                    ratio = new_freq / old_freq
                    for h in self.harmonic_lines:
                        if h is self.selected_line:
                            continue  # H1 already updated
                        h['freq'] *= ratio


            # Re-detect ridges and contours at new position
            if self.show_ridges.get():
                self.detect_harmonic_ridges()
            
            if self.show_contour.get():
                self.compute_contours()
            
            self.update_display()
        
        self.selected_line = None
        self.drag_start_y = None
        self.update_info()
        
        return True
    
    def on_prominence_change(self, value):
        """Handle prominence slider change"""
        self.prom_label.config(text=f"{self.prominence.get():.1f}")





    def update_info(self):
        """Update info label with current state"""
        if self.detected_f0 is None:
            self.info_label.config(text="No F0 detected")
        else:
            n_harmonics = len(self.harmonic_lines)
            self.info_label.config(
                text=f"F0: {self.detected_f0:.1f} Hz\n"
                    f"Harmonics: {n_harmonics}\n"
                    f"Click line for options"
            )
        # Add this line:
        if hasattr(self, 'harmonics_listbox'):
            self.update_harmonics_list()
            self.update_button_states()




    def save_custom_data(self):
        """Save harmonic annotations"""
        if not self.annotation_dir:
            return
        
        annotation_file = self.get_annotation_file()
        
        # Prepare harmonic data for ML pipeline
        harmonic_data = []
        for h in self.harmonic_lines:
            harmonic_data.append({
                'harmonic_num': h['num'],
                'frequency': float(h['freq']),
                'is_manual': h['num'] != 1  # F0 is detected, others are manual
            })
        
        # Extract ridge data if available
        ridge_data = {}
        for harm_num, ridge in self.harmonic_ridges.items():
            ridge_data[str(harm_num)] = [(float(t), float(f)) for t, f in ridge]
        
        data = {
            'audio_file': str(self.audio_files[self.current_file_idx]),
            'detected_f0': float(self.detected_f0) if self.detected_f0 else None,
            'harmonics': harmonic_data,
            'ridges': ridge_data,
            'parameters': {
                'n_fft': self.n_fft.get(),
                'hop_length': self.hop_length.get(),
                'prominence': self.prominence.get(),
                'freq_range': [self.freq_min.get(), self.freq_max.get()],
                'ridge_method': self.ridge_method.get(),
                'peak_tolerance': self.peak_tolerance.get(),
                'show_ridges': bool(self.show_ridges.get()),
                'show_valleys': bool(self.show_valleys.get()),
                'show_contour': bool(self.show_contour.get()),
                'contour_method': self.contour_method.get(),
                'contour_smoothness': float(self.contour_smoothness.get())
            }
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved harmonic data to {annotation_file.name}")
        self.changes_made = False
        
        # Prepare training data for ML pipeline
        self.prepare_training_data()



    def load_custom_data(self):
        """Load harmonic annotations"""
        annotation_file = self.get_annotation_file()
        
        if not annotation_file.exists():
            return
        
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            # Restore harmonics
            self.harmonic_lines = []
            for h in data.get('harmonics', []):
                self.harmonic_lines.append({
                    'freq': h['frequency'],
                    'num': h['harmonic_num'],
                    'line': None
                })
            
            self.detected_f0 = data.get('detected_f0')
            
            # Restore ridges if available
            self.harmonic_ridges = {}
            for harm_num_str, ridge in data.get('ridges', {}).items():
                self.harmonic_ridges[int(harm_num_str)] = ridge
            

            # Restore UI / method state if present
            params = data.get('parameters', {})
            if 'ridge_method' in params:
                self.ridge_method.set(params['ridge_method'])
            if 'peak_tolerance' in params:
                self.peak_tolerance.set(params['peak_tolerance'])
            if 'show_ridges' in params:
                self.show_ridges.set(bool(params['show_ridges']))
            if 'show_valleys' in params:
                self.show_valleys.set(bool(params['show_valleys']))
            if 'show_contour' in params:
                self.show_contour.set(bool(params['show_contour']))
            if 'contour_method' in params:
                self.contour_method.set(params['contour_method'])
            if 'contour_smoothness' in params:
                self.contour_smoothness.set(params['contour_smoothness'])

            # If contours were enabled, recompute them from loaded ridges
            if self.show_contour.get() and self.harmonic_ridges:
                self.compute_contours()


            print(f"✓ Loaded {len(self.harmonic_lines)} harmonics")
            
        except Exception as e:
            print(f"Error loading annotations: {e}")
    


    def prepare_training_data(self):
        """Prepare data for harmonic learning pipeline"""
        if not self.detected_f0 or not self.S_db:
            return
        
        training_file = self.annotation_dir / "harmonic_training.pkl"
        
        # Extract features around each harmonic
        samples = []
        
        for h in self.harmonic_lines:
            if h['num'] == 1:  # Skip F0 as it's auto-detected
                continue
            
            # Get spectral context
            freq_idx = np.argmin(np.abs(self.freqs - h['freq']))
            context = 20  # bins
            
            f_start = max(0, freq_idx - context)
            f_end = min(len(self.freqs), freq_idx + context)
            
            spec_slice = self.S_db[f_start:f_end, :]
            
            samples.append({
                'spectrogram': spec_slice,
                'freqs': self.freqs[f_start:f_end],
                'target_freq': h['freq'],
                'harmonic_num': h['num'],
                'f0': self.detected_f0,
                'audio_file': str(self.audio_files[self.current_file_idx])
            })
        
        if samples:
            import pickle
            
            # Append to existing training data
            if training_file.exists():
                with open(training_file, 'rb') as f:
                    existing = pickle.load(f)
            else:
                existing = []
            
            existing.extend(samples)
            
            with open(training_file, 'wb') as f:
                pickle.dump(existing, f)
            
            print(f"✓ Added {len(samples)} training samples ({len(existing)} total)")
    
    def get_annotation_file(self):
        """Get annotation file path for current audio"""
        audio_file = self.audio_files[self.current_file_idx]
        return self.annotation_dir / f"{audio_file.stem}_harmonic_layer.json"


def main():
    """Entry point for harmonic layer annotator"""
    root = tk.Tk()
    app = HarmonicLayer(root)
    root.geometry("1400x800")
    root.mainloop()


if __name__ == "__main__":
    main()