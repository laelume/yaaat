"""
Harmonic Annotator - Interactive tool for correcting harmonic frequency detections
Part of YAAAT (Yet Another Audio Annotation Tool)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.collections

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pathlib import Path
import json
from natsort import natsorted
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import minimum_filter1d

import pysoniq

try: 
    from yaaat import utils
except ImportError: 
    import utils

# ============================================================================
# FUZZY VALLEY BACKEND (from your documents)
# ============================================================================

class FlexibleSpectralValleyTracker:
    def __init__(self):
        """Initialize without hardcoded parameters"""
        self.y = None
        self.sr = None
        self.magnitude = None
        self.log_magnitude = None
        self.freqs = None
        self.times = None
        self.hop_length = None
        
    def load_audio(self, filepath, sr=None):
        """Load audio file"""
        self.y, self.sr = pysoniq.load(str(filepath))
        if self.y.ndim > 1:
            self.y = np.mean(self.y, axis=1)
        return self.y, self.sr
    
    def compute_spectrogram(self, n_fft=2048, hop_length=None, window='hann'):
        """Compute spectrogram"""
        if hop_length is None:
            hop_length = n_fft // 4
        
        from scipy import signal
        
        if window == 'hann':
            win = np.hanning(n_fft)
        elif window == 'hamming':
            win = np.hamming(n_fft)
        elif window == 'blackman':
            win = np.blackman(n_fft)
        else:
            win = np.ones(n_fft)
        
        f, t, Zxx = signal.stft(self.y, 
                                self.sr,
                                nperseg=n_fft, 
                                noverlap=n_fft - hop_length,
                                window=win,
                                return_onesided=True,
                                boundary=None,
                                padded=False)
        
        self.magnitude = np.abs(Zxx)
        self.log_magnitude = 20.0 * np.log10(np.maximum(1e-10, self.magnitude))
        self.freqs = f
        self.times = t
        self.hop_length = hop_length
        
        return self.magnitude, self.log_magnitude
    
    def analyze_psd(self, nperseg=None, noverlap=None, window='hann'):
        """Compute PSD"""
        from scipy import signal
        
        if nperseg is None:
            nperseg = min(2048, len(self.y) // 8)
        if noverlap is None:
            noverlap = nperseg // 2
        
        freqs, psd = signal.welch(self.y, fs=self.sr, 
                                 nperseg=nperseg, 
                                 noverlap=noverlap,
                                 window=window, 
                                 scaling='density')
        
        psd_db = 10 * np.log10(psd + 1e-12)
        
        return freqs, psd, psd_db
    
    def detect_fundamental_and_harmonics_from_psd(self, freqs, psd_db, 
                                                 fmin=100, fmax=None,
                                                 prominence_threshold=5,
                                                 max_harmonics=10,
                                                 harmonic_tolerance=0.05):
        """Detect fundamental frequency and harmonics from PSD"""
        if fmax is None:
            fmax = self.sr / 2
        
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs_subset = freqs[freq_mask]
        psd_subset = psd_db[freq_mask]
        
        # Find peaks in PSD
        peaks, properties = find_peaks(psd_subset, 
                                      prominence=prominence_threshold,
                                      distance=max(5, int(50 / (freqs[1] - freqs[0]))))
        
        if len(peaks) == 0:
            return None, []
        
        peak_freqs = freqs_subset[peaks]
        peak_mags = psd_subset[peaks]
        
        # Sort by magnitude
        sorted_indices = np.argsort(peak_mags)[::-1]
        peak_freqs_sorted = peak_freqs[sorted_indices]
        
        # Find fundamental
        best_fundamental = None
        best_harmonic_series = []
        best_score = 0
        
        for f0_candidate in peak_freqs_sorted[:min(10, len(peak_freqs_sorted))]:
            if f0_candidate < fmin:
                continue
            
            harmonic_series = [f0_candidate]
            tolerance = f0_candidate * harmonic_tolerance
            
            for harmonic_num in range(2, max_harmonics + 1):
                expected_freq = f0_candidate * harmonic_num
                if expected_freq > fmax:
                    break
                
                freq_diffs = np.abs(peak_freqs - expected_freq)
                closest_idx = np.argmin(freq_diffs)
                
                if freq_diffs[closest_idx] < tolerance:
                    harmonic_series.append(peak_freqs[closest_idx])
            
            score = len(harmonic_series)
            if score > best_score:
                best_score = score
                best_fundamental = f0_candidate
                best_harmonic_series = harmonic_series
        
        return best_fundamental, best_harmonic_series
    
    def track_harmonics_with_template(self, harmonic_series, 
                                    fmin=None, fmax=None,
                                    freq_tolerance=0.08,
                                    prominence_factor=0.05, 
                                    curve_smoothing_window=7):
        """Track harmonics across time using PSD-derived template"""
        if fmin is None:
            fmin = min(harmonic_series) * 0.8
        if fmax is None:
            fmax = max(harmonic_series) * 1.2
        
        harmonic_tracks = []
        
        for t_idx in range(self.magnitude.shape[1]):
            spectrum = self.log_magnitude[:, t_idx]
            frame_harmonics = {
                'time': self.times[t_idx],
                'harmonics': []
            }
            
            for i, expected_freq in enumerate(harmonic_series):
                if expected_freq < fmin or expected_freq > fmax:
                    continue
                
                tolerance = expected_freq * freq_tolerance
                freq_range = (expected_freq - tolerance, expected_freq + tolerance)
                
                freq_mask = (self.freqs >= freq_range[0]) & (self.freqs <= freq_range[1])
                
                if not np.any(freq_mask):
                    continue
                
                freq_indices = np.where(freq_mask)[0]
                spectrum_slice = spectrum[freq_mask]
                
                # Find peaks in this frequency range
                spectrum_range = np.max(spectrum_slice) - np.min(spectrum_slice)
                if spectrum_range > 0:
                    prominence_threshold = spectrum_range * prominence_factor
                    
                    peaks, _ = find_peaks(spectrum_slice, prominence=prominence_threshold)
                    
                    if len(peaks) > 0:
                        strongest_peak_idx = peaks[np.argmax(spectrum_slice[peaks])]
                        actual_freq_idx = freq_indices[strongest_peak_idx]
                        
                        harmonic_data = {
                            'harmonic_number': i + 1,
                            'expected_frequency': expected_freq,
                            'actual_frequency': self.freqs[actual_freq_idx],
                            'magnitude': spectrum[actual_freq_idx],
                            'freq_idx': actual_freq_idx
                        }
                        frame_harmonics['harmonics'].append(harmonic_data)
            
            harmonic_tracks.append(frame_harmonics)
                    
        # Smooth harmonic trajectories
        if curve_smoothing_window >= 3:
            for harm_num in range(1, len(harmonic_series) + 1):
                harm_times = []
                harm_freqs = []
                harm_indices = []
                
                for t_idx, frame_data in enumerate(harmonic_tracks):
                    for harmonic in frame_data['harmonics']:
                        if harmonic['harmonic_number'] == harm_num:
                            harm_times.append(frame_data['time'])
                            harm_freqs.append(harmonic['actual_frequency'])
                            harm_indices.append(t_idx)
                            break
                
                if len(harm_freqs) >= curve_smoothing_window:
                    window_len = min(curve_smoothing_window, len(harm_freqs))
                    if window_len % 2 == 0:
                        window_len -= 1
                    if window_len >= 3:
                        harm_freqs_smooth = savgol_filter(harm_freqs, window_length=window_len, polyorder=2)
                        
                        # Update actual frequencies
                        for i, t_idx in enumerate(harm_indices):
                            for harmonic in harmonic_tracks[t_idx]['harmonics']:
                                if harmonic['harmonic_number'] == harm_num:
                                    harmonic['actual_frequency'] = harm_freqs_smooth[i]
                                    break
        
        return harmonic_tracks
    
    def find_valleys_between_harmonics(self, harmonic_tracks, 
                                     valley_margin=0.25,
                                     min_gap=50):
        """Find valleys between consecutive harmonics"""
        valley_tracks = {}
        
        for t_idx, frame_data in enumerate(harmonic_tracks):
            harmonics = frame_data['harmonics']
            
            if len(harmonics) < 2:
                continue
            
            harmonics_sorted = sorted(harmonics, key=lambda x: x['actual_frequency'])
            
            for i in range(len(harmonics_sorted) - 1):
                h1 = harmonics_sorted[i]
                h2 = harmonics_sorted[i + 1]
                
                h1_freq = h1['actual_frequency']
                h2_freq = h2['actual_frequency']
                
                if h2_freq - h1_freq < min_gap:
                    continue
                
                freq_gap = h2_freq - h1_freq
                margin = freq_gap * valley_margin
                
                search_start_freq = h1_freq + margin
                search_end_freq = h2_freq - margin
                
                if search_start_freq >= search_end_freq:
                    continue
                
                search_start_idx = np.searchsorted(self.freqs, search_start_freq)
                search_end_idx = np.searchsorted(self.freqs, search_end_freq)
                
                valley_spectrum = self.log_magnitude[search_start_idx:search_end_idx, t_idx]
                valley_local_idx = np.argmin(valley_spectrum)
                valley_freq_idx = search_start_idx + valley_local_idx
                
                pair_key = f"H{h1['harmonic_number']}-H{h2['harmonic_number']}"
                
                if pair_key not in valley_tracks:
                    valley_tracks[pair_key] = []
                
                valley_data = {
                    'time': frame_data['time'],
                    'frequency': self.freqs[valley_freq_idx],
                    'magnitude': valley_spectrum[valley_local_idx],
                    'freq_idx': valley_freq_idx,
                    'between_harmonics': (h1_freq, h2_freq),
                    'harmonic_numbers': (h1['harmonic_number'], h2['harmonic_number']),
                    'valley_depth': (h1['magnitude'] + h2['magnitude']) / 2 - valley_spectrum[valley_local_idx]
                }
                
                valley_tracks[pair_key].append(valley_data)
        
        return valley_tracks


# ============================================================================
# HARMONIC ANNOTATOR GUI
# ============================================================================

class HarmonicAnnotator:
    """Interactive tool for correcting harmonic frequency detections"""
    
    def __init__(self, root):
        self.root = root
        if isinstance(root, tk.Tk):
            self.root.title("Harmonic Annotator - YAAAT")
        
        # Audio and tracking data
        self.audio_files = []
        self.current_file_idx = 0
        self.tracker = None
        self.fundamental_freq = None
        self.harmonic_series = []
        self.harmonic_tracks = []
        self.valley_tracks = {}
        self.boundary_data = None
        
        # Annotation data
        self.harmonic_corrections = []  # {time_idx, harmonic_num, old_freq, new_freq}
        self.changes_made = False
        self.annotation_dir = None
        self.base_audio_dir = None
        
        # Detection parameters
        self.n_fft = tk.IntVar(value=512)
        self.hop_length = tk.IntVar(value=64)
        self.psd_nperseg = tk.IntVar(value=512)
        self.psd_noverlap = tk.IntVar(value=128)
        self.fmin = tk.IntVar(value=500)
        self.fmax = tk.IntVar(value=16000)
        self.max_harmonics = tk.IntVar(value=10)
        self.prominence_threshold = tk.DoubleVar(value=3.0)
        self.curve_smoothing_window = tk.IntVar(value=7)

        # Display parameters
        self.fmin_plot = tk.IntVar(value=100)
        self.fmax_plot = tk.IntVar(value=16000)
        
        # UI state
        self.dragging_harmonic = None  # (time_idx, harmonic_num, edge_type, base_freq)
        self.drag_start = None
        self.drag_rect = None
        self.changepoints = []  # List of (time, freq) tuples marking segment boundaries
        self.selected_region = None  # (region_idx, base_freq, click_freq) when dragging
        self.region_hover = None  # Region index under cursor
        self.zoom_stack = []
        self.spec_image = None
        
        # Playback state
        self.playback_gain = tk.DoubleVar(value=1.0)
        self.loop_enabled = False
        
        # Visualization options
        self.show_valleys = tk.BooleanVar(value=True)
        self.show_harmonics = tk.BooleanVar(value=True)
        
        self.setup_ui()
        self.root.after(100, self.auto_load_directory)
    
    def setup_ui(self):
        """Create the user interface"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== LEFT CONTROL PANEL =====
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Scrollable canvas
        canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scrollable_frame.bind("<Configure>", on_frame_configure)
        
        # Mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_to_mousewheel(widget):
            widget.bind("<MouseWheel>", on_mousewheel)
            for child in widget.winfo_children():
                bind_to_mousewheel(child)
        
        bind_to_mousewheel(scrollable_frame)
        
        # ===== HEADER =====
        ttk.Label(scrollable_frame, text="Harmonic Annotator", font=('', 10, 'bold')).pack(pady=(0, 2))
        ttk.Label(scrollable_frame, text="Correct harmonic frequency detections by dragging harmonic ridges", 
                  wraplength=400, font=('', 8, 'italic')).pack(padx=5, pady=(0, 3))
        
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # ===== INSTRUCTIONS =====
        ttk.Label(scrollable_frame, text="Instructions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        instructions = ttk.Label(scrollable_frame, 
                        text="• Click harmonic: drag full contour\n"
                             "• Ctrl+Click: add changepoint (teal)\n"
                             "• Shift+Drag: move region only\n"
                             "• Right-click: undo zoom", 
                        wraplength=400, font=('', 8))
        instructions.pack(padx=5, pady=(0, 5))
        
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # ===== FILE MANAGEMENT =====
        file_buttons_frame = ttk.Frame(scrollable_frame)
        file_buttons_frame.pack(fill=tk.X, pady=2)
        
        load_frame = ttk.Frame(file_buttons_frame)
        load_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        ttk.Label(load_frame, text="File Management:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Button(load_frame, text="Load Audio Directory", command=self.load_directory).pack(anchor=tk.W, pady=2)
        ttk.Button(load_frame, text="Load Test Audio", command=self.load_test_audio).pack(anchor=tk.W, pady=2)
        
        ttk.Separator(file_buttons_frame, orient=tk.VERTICAL).grid(row=0, column=1, sticky='ns', padx=10)
        
        # ===== PLAYBACK CONTROLS =====
        play_frame = ttk.Frame(file_buttons_frame)
        play_frame.grid(row=0, column=2, sticky='nsew', padx=(5, 0))
        
        ttk.Label(play_frame, text="Playback:", font=('', 9, 'bold')).pack(anchor=tk.CENTER, pady=(0, 2))
        
        controls_container = ttk.Frame(play_frame)
        controls_container.pack(anchor=tk.CENTER)
        
        buttons_row = ttk.Frame(controls_container)
        buttons_row.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(buttons_row, text="▶", command=self.play_audio, bg='lightgreen', 
                 font=('', 12, 'bold'), width=2, height=1).pack(side=tk.LEFT, padx=2)
        
        self.pause_button = tk.Button(buttons_row, text="⏸", command=self.pause_audio, 
                                      bg='yellow', font=('', 12, 'bold'), width=2, height=1)
        self.pause_button.pack(side=tk.LEFT, padx=2)
        
        tk.Button(buttons_row, text="⏹", command=self.stop_audio, bg='lightcoral', 
                 font=('', 12, 'bold'), width=2, height=1).pack(side=tk.LEFT, padx=2)
        
        self.loop_button = tk.Button(buttons_row, text="⟳", command=self.toggle_loop, 
                                     bg='lightblue', font=('', 12, 'bold'), width=2, height=1)
        self.loop_button.pack(side=tk.LEFT, padx=2)
        
        # Gain slider
        gain_frame = ttk.Frame(controls_container)
        gain_frame.pack(side=tk.LEFT)
        
        ttk.Label(gain_frame, text="Gain", font=('', 7)).pack()
        ttk.Scale(gain_frame, from_=2.0, to=0.0, variable=self.playback_gain,
                 orient=tk.VERTICAL, length=60, command=lambda v: self.update_gain_label()).pack()
        
        self.gain_label = ttk.Label(gain_frame, text="100%", font=('', 7))
        self.gain_label.pack()
        
        file_buttons_frame.columnconfigure(0, weight=1)
        file_buttons_frame.columnconfigure(2, weight=1)
        
        # File info
        self.file_label = ttk.Label(scrollable_frame, text="No files loaded", wraplength=400, font=('', 8))
        self.file_label.pack(fill=tk.X, pady=2)
        
        self.annotation_file_button = tk.Button(scrollable_frame, text="No annotation file", 
                                               font=('', 8), relief=tk.FLAT, fg='blue', cursor='hand2',
                                               command=self.open_annotation_location, bg='SystemButtonFace', anchor='w')
        self.annotation_file_button.pack(fill=tk.X, pady=2)
        
        self.save_dir_button = tk.Button(scrollable_frame, text="No save directory", 
                                        font=('', 8), relief=tk.FLAT, fg='blue', cursor='hand2',
                                        command=self.open_save_location, bg='SystemButtonFace', anchor='w')
        self.save_dir_button.pack(anchor=tk.W, pady=2)
        
        # Correction count
        self.correction_info = ttk.Label(scrollable_frame, text="Corrections: 0", font=('', 8))
        self.correction_info.pack(pady=2)
        
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # ===== DETECTION PARAMETERS =====
        ttk.Label(scrollable_frame, text="Detection Parameters:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        # Frequency range
        freq_frame = ttk.Frame(scrollable_frame)
        freq_frame.pack(fill=tk.X, pady=2)
        ttk.Label(freq_frame, text="Analysis range (Hz):", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.fmin, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(freq_frame, text="-", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.fmax, width=5).pack(side=tk.LEFT, padx=2)
        
        # Max harmonics
        harm_frame = ttk.Frame(scrollable_frame)
        harm_frame.pack(fill=tk.X, pady=2)
        ttk.Label(harm_frame, text="Max harmonics:", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(harm_frame, textvariable=self.max_harmonics, width=5).pack(side=tk.LEFT, padx=2)
        
        # Prominence
        prom_frame = ttk.Frame(scrollable_frame)
        prom_frame.pack(fill=tk.X, pady=2)
        ttk.Label(prom_frame, text="Prominence:", font=('', 8)).pack(side=tk.LEFT)
        ttk.Scale(prom_frame, from_=0.01, to=1.0, variable=self.prominence_threshold,
                 orient=tk.HORIZONTAL, command=lambda v: self.update_prominence_label()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.prom_label = ttk.Label(prom_frame, text=f"{self.prominence_threshold.get():.1f}", font=('', 8), width=5)
        self.prom_label.pack(side=tk.LEFT)
        

        

        # Smoothing
        smooth_frame = ttk.Frame(scrollable_frame)
        smooth_frame.pack(fill=tk.X, pady=2)
        ttk.Label(smooth_frame, text="Curve smoothing:", font=('', 8)).pack(side=tk.LEFT)
        ttk.Scale(smooth_frame, from_=0, to=51, variable=self.curve_smoothing_window, orient=tk.HORIZONTAL, command=lambda v: self.on_smoothing_change(v)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.smooth_label = ttk.Label(smooth_frame, text=f"{self.curve_smoothing_window.get()}", font=('', 8), width=5)
        self.smooth_label.pack(side=tk.LEFT)


        ttk.Button(scrollable_frame, text="Recompute Harmonics", command=self.redetect_harmonics).pack(fill=tk.X, pady=2)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        





        # ===== VISUALIZATION OPTIONS =====
        ttk.Label(scrollable_frame, text="Display:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        ttk.Checkbutton(scrollable_frame, text="Show Valleys", variable=self.show_valleys, 
                       command=self.update_display).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(scrollable_frame, text="Show Harmonics", variable=self.show_harmonics,
                       command=self.update_display).pack(anchor=tk.W, pady=2)
        
        # Display range
        disp_frame = ttk.Frame(scrollable_frame)
        disp_frame.pack(fill=tk.X, pady=2)
        ttk.Label(disp_frame, text="Display (Hz):", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(disp_frame, textvariable=self.fmin_plot, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(disp_frame, text="-", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(disp_frame, textvariable=self.fmax_plot, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(disp_frame, text="↻", width=2, command=self.update_display_range).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # ===== ACTIONS =====
        ttk.Label(scrollable_frame, text="Actions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        button_grid = ttk.Frame(scrollable_frame)
        button_grid.pack(pady=2)
        
        buttons = [
            ("Undo Last", self.undo_last_correction),
            ("Play Audio", self.play_audio),
            ("Clear All", self.clear_all_corrections),
            ("Next File", self.next_file),
            
            ("Reset Zoom", self.reset_zoom),
            ("Debug Info", self.print_debug_info),
            ("Save Anno", self.save_annotations),
            ("Prev File", self.previous_file),

            ("Clear Changepoints", self.clear_changepoints),  # NEW

        ]
        
        for i, (text, command) in enumerate(buttons):
            row = i // 4
            col = i % 4
            ttk.Button(button_grid, text=text, command=command, width=12).grid(
                row=row, column=col, padx=2, pady=2, sticky='ew')
        
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # ===== STATISTICS =====
        ttk.Label(scrollable_frame, text="Statistics:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.stats_label = ttk.Label(scrollable_frame, text="No harmonics detected", 
                                     justify=tk.LEFT, font=('', 8))
        self.stats_label.pack(fill=tk.X, pady=2)
        
        # ===== RIGHT SPECTROGRAM PANEL =====
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Navigation
        nav_frame = ttk.Frame(plot_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(nav_frame, text="◄ Previous", command=self.previous_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next ►", command=self.next_file).pack(side=tk.LEFT, padx=5)
        
        file_nav_frame = ttk.Frame(nav_frame)
        file_nav_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(file_nav_frame, text="File:", font=('', 9)).pack(side=tk.LEFT, padx=2)
        self.file_number_entry = ttk.Entry(file_nav_frame, width=6, justify=tk.CENTER)
        self.file_number_entry.pack(side=tk.LEFT, padx=2)
        self.file_total_label = ttk.Label(file_nav_frame, text="/ 0", font=('', 9))
        self.file_total_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(file_nav_frame, text="Go", command=self.jump_to_file, width=4).pack(side=tk.LEFT, padx=2)
        
        self.file_number_entry.bind('<Return>', lambda e: self.jump_to_file())
        
        ttk.Label(nav_frame, text="[Drag harmonic: correct | Right-click: reset zoom | Ctrl+Wheel: zoom]", 
                  font=('', 8, 'italic')).pack(side=tk.RIGHT, padx=10)
        
        # Spectrogram canvas
        self.fig = Figure(figsize=(10, 6))
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Zoom info
        self.zoom_info_label = ttk.Label(plot_frame, text="", font=('', 8), foreground='blue')
        self.zoom_info_label.pack(pady=(2, 0))
        
        # Bind events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Initialize empty plot
        self.ax.set_xlabel('Time (s)', fontsize=8)
        self.ax.set_ylabel('Frequency (Hz)', fontsize=8)
        self.ax.set_title('Load audio files to begin')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()
    
    # ===== FILE MANAGEMENT =====
    
    def load_directory(self):
        """Load all .wav files from a directory"""
        directory = filedialog.askdirectory(title="Select Audio Directory")
        if not directory:
            return
        
        self.audio_files = natsorted(Path(directory).rglob('*.wav'))
        self.base_audio_dir = Path(directory)
        
        if not self.audio_files:
            messagebox.showwarning("No Files", "No .wav files found")
            return
        
        # Ask for annotation save location
        response = messagebox.askyesnocancel(
            "Annotation Save Location",
            "Where to save harmonic corrections?\n\n"
            "Yes = Choose existing directory\n"
            "No = Create new directory\n"
            "Cancel = Use default location"
        )
        
        if response is True:
            save_dir = filedialog.askdirectory(title="Select Annotation Directory")
            if save_dir:
                self.annotation_dir = Path(save_dir)
            else:
                return
        elif response is False:
            save_dir = filedialog.askdirectory(title="Select Parent Directory")
            if save_dir:
                dataset_name = Path(directory).name
                self.annotation_dir = Path(save_dir) / f"{dataset_name}_harmonic_corrections"
                self.annotation_dir.mkdir(exist_ok=True)
            else:
                return
        else:
            dataset_name = Path(directory).name
            default_dir = Path.home() / "yaaat_annotations" / f"{dataset_name}_harmonics"
            default_dir.mkdir(parents=True, exist_ok=True)
            self.annotation_dir = default_dir
        
        self.annotation_dir.mkdir(exist_ok=True)
        self.save_dir_button.config(text=f"📁 {self.annotation_dir}")
        
        self.current_file_idx = 0
        self.load_current_file()
        
        print(f"✓ Loaded {len(self.audio_files)} files")
        utils.save_last_directory(self.base_audio_dir)
    
    def load_test_audio(self):
        """Load bundled test audio files"""
        test_audio_dir = Path(__file__).parent / 'test_files' / 'test_audio' / 'kiwi'
        
        if not test_audio_dir.exists():
            messagebox.showinfo("No Test Data", "Test audio files not found")
            return
        
        self.audio_files = natsorted(test_audio_dir.rglob('*.wav'))
        self.base_audio_dir = test_audio_dir
        
        if not self.audio_files:
            messagebox.showwarning("No Files", "No .wav files in test directory")
            return
        
        default_dir = Path.home() / "yaaat_annotations" / "test_audio_harmonics"
        default_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_dir = default_dir
        
        self.save_dir_button.config(text=f"📁 {self.annotation_dir}")
        
        self.current_file_idx = 0
        self.load_current_file()
        
        print(f"✓ Loaded {len(self.audio_files)} test files")
        utils.save_last_directory(self.base_audio_dir)
    
    def auto_load_directory(self):
        """Auto-load last directory on startup"""
        last_dir = utils.load_last_directory()
        if last_dir and last_dir.exists():
            print(f"Auto-loading: {last_dir}")
            self.audio_files = natsorted(last_dir.rglob('*.wav'))
            self.base_audio_dir = last_dir
            if self.audio_files:
                dataset_name = last_dir.name
                self.annotation_dir = Path.home() / "yaaat_annotations" / f"{dataset_name}_harmonics"
                self.annotation_dir.mkdir(parents=True, exist_ok=True)
                self.save_dir_button.config(text=f"📁 {self.annotation_dir}")
                self.current_file_idx = 0
                self.load_current_file()
                return
        
        self.load_test_audio()
    
    def load_current_file(self):
        """Load current audio file and detect harmonics"""
        if not self.audio_files:
            return
        
        audio_file = self.audio_files[self.current_file_idx]
        print(f"Loading {audio_file.name}...")
        
        # Initialize tracker
        self.tracker = FlexibleSpectralValleyTracker()
        y, sr = self.tracker.load_audio(audio_file, sr=None)
        
        # Compute spectrogram
        self.tracker.compute_spectrogram(
            n_fft=self.n_fft.get(),
            hop_length=self.hop_length.get(),
            window='hann'
        )
        
        # Detect harmonics
        self.detect_harmonics()
        
        # Clear corrections
        self.harmonic_corrections = []
        
        # Load existing annotations
        annotation_file = self.get_annotation_file()
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                self.harmonic_corrections = data.get('harmonic_corrections', [])
                print(f"✓ Loaded {len(self.harmonic_corrections)} corrections")
        
        self.changes_made = False
        self.zoom_stack = []
        self.spec_image = None
        self.update_display(recompute=True)
        self.update_progress()
        
        # Update annotation file button
        if annotation_file.exists() and self.harmonic_corrections:
            self.annotation_file_button.config(text=f"✓ {annotation_file.name}", foreground='green')
        else:
            self.annotation_file_button.config(text=f"→ {annotation_file.name}", foreground='blue')
    
    def get_annotation_file(self):
        """Get annotation file path for current audio file"""
        audio_file = self.audio_files[self.current_file_idx]
        relative_path = audio_file.relative_to(self.base_audio_dir).parent
        filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
        
        if filename_prefix and filename_prefix != '.':
            return self.annotation_dir / f"{filename_prefix}_{audio_file.stem}_harmonic_corrections.json"
        else:
            return self.annotation_dir / f"{audio_file.stem}_harmonic_corrections.json"
    
    # ===== HARMONIC DETECTION =====
    
    def detect_harmonics(self):
        """Detect harmonics using fuzzy valley method"""
        if self.tracker is None:
            return
        
        # Compute PSD
        psd_freqs, psd, psd_db = self.tracker.analyze_psd(
            nperseg=self.psd_nperseg.get(),
            noverlap=self.psd_noverlap.get(),
            window='hann'
        )
        
        # Detect fundamental and harmonics
        self.fundamental_freq, self.harmonic_series = \
            self.tracker.detect_fundamental_and_harmonics_from_psd(
                psd_freqs, psd_db,
                fmin=self.fmin.get(),
                fmax=self.fmax.get(),
                prominence_threshold=self.prominence_threshold.get(),
                max_harmonics=self.max_harmonics.get(),
                harmonic_tolerance=0.1
            )
        
        if self.fundamental_freq is None:
            print("⚠ No fundamental detected")
            self.harmonic_tracks = []
            self.valley_tracks = {}
            self.boundary_data = None
            return
        
        # Track harmonics
        self.harmonic_tracks = self.tracker.track_harmonics_with_template(
            self.harmonic_series,
            fmin=self.fmin.get(),
            fmax=self.fmax.get(), 
            curve_smoothing_window=self.curve_smoothing_window.get()
        )
        
        # Find valleys
        self.valley_tracks = self.tracker.find_valleys_between_harmonics(
            self.harmonic_tracks,
            valley_margin=0.2
        )
        
        # Compute boundary data (for visualization)
        self.compute_boundary_data()
        
        print(f"✓ Detected F0={self.fundamental_freq:.1f} Hz, {len(self.harmonic_series)} harmonics")
    
    def compute_boundary_data(self):
        """Compute valley boundaries for visualization"""
        valley_boundaries = []
        
        for pair_key, valley_data in self.valley_tracks.items():
            if valley_data:
                times = np.array([v['time'] for v in valley_data])
                freqs = np.array([v['frequency'] for v in valley_data])
                
                # Smooth
                if len(freqs) >= self.curve_smoothing_window.get():
                    window_len = min(self.curve_smoothing_window.get(), len(freqs))
                    if window_len % 2 == 0:
                        window_len -= 1
                    if window_len >= 3:
                        freqs_smooth = savgol_filter(freqs, window_length=window_len, polyorder=2)
                    else:
                        freqs_smooth = freqs

                else:
                    freqs_smooth = freqs
                
                # Interpolate to time grid
                valley_interp = np.interp(self.tracker.times, times, freqs_smooth)
                valley_boundaries.append(valley_interp)
        
        # Sort by frequency
        valley_boundaries = sorted(valley_boundaries, key=lambda x: np.mean(x))
        
        # Find dynamic lower boundary (below F0)
        f0_ridge_freqs = []
        f0_times = []
        
        for frame_data in self.harmonic_tracks:
            for harmonic in frame_data['harmonics']:
                if harmonic['harmonic_number'] == 1:
                    f0_ridge_freqs.append(harmonic['actual_frequency'])
                    f0_times.append(frame_data['time'])
                    break
        
        if f0_ridge_freqs:
            f0_ridge_interp = np.interp(self.tracker.times, f0_times, f0_ridge_freqs)
            
            lower_valley_data = []
            for t_idx in range(len(self.tracker.times)):
                search_start_idx = np.searchsorted(self.tracker.freqs, self.fmin_plot.get())
                search_end_idx = np.searchsorted(self.tracker.freqs, f0_ridge_interp[t_idx])
                
                if search_end_idx > search_start_idx:
                    valley_spectrum = self.tracker.log_magnitude[search_start_idx:search_end_idx, t_idx]
                    valley_local_idx = np.argmin(valley_spectrum)
                    valley_freq_idx = search_start_idx + valley_local_idx
                    lower_valley_data.append(self.tracker.freqs[valley_freq_idx])
                else:
                    lower_valley_data.append(self.fmin_plot.get())
            
            if len(lower_valley_data) >= self.curve_smoothing_window.get():
                window_len = min(self.curve_smoothing_window.get(), len(lower_valley_data))
                if window_len % 2 == 0:
                    window_len -= 1
                if window_len >= 3:
                    dynamic_lower = savgol_filter(lower_valley_data, window_length=window_len, polyorder=2)
                else:
                    dynamic_lower = np.array(lower_valley_data)
            else:
                dynamic_lower = np.array(lower_valley_data)

        else:
            dynamic_lower = np.full_like(self.tracker.times, self.fmin_plot.get())
        
        # Find dynamic upper boundary (above highest harmonic)
        max_harm_freqs = []
        max_harm_times = []
        
        for frame_data in self.harmonic_tracks:
            if frame_data['harmonics']:
                highest = max(frame_data['harmonics'], key=lambda h: h['actual_frequency'])
                max_harm_freqs.append(highest['actual_frequency'])
                max_harm_times.append(frame_data['time'])
        
        if max_harm_freqs:
            max_harm_interp = np.interp(self.tracker.times, max_harm_times, max_harm_freqs)
            
            upper_valley_data = []
            for t_idx in range(len(self.tracker.times)):
                search_start_idx = np.searchsorted(self.tracker.freqs, max_harm_interp[t_idx])
                search_end_idx = np.searchsorted(self.tracker.freqs, self.fmax_plot.get())
                
                if search_end_idx > search_start_idx:
                    valley_spectrum = self.tracker.log_magnitude[search_start_idx:search_end_idx, t_idx]
                    valley_local_idx = np.argmin(valley_spectrum)
                    valley_freq_idx = search_start_idx + valley_local_idx
                    upper_valley_data.append(self.tracker.freqs[valley_freq_idx])
                else:
                    upper_valley_data.append(self.fmax_plot.get())
            
            if len(upper_valley_data) >= self.curve_smoothing_window.get():
                window_len = min(self.curve_smoothing_window.get(), len(upper_valley_data))
                if window_len % 2 == 0:
                    window_len -= 1
                if window_len >= 3:
                    dynamic_upper = savgol_filter(upper_valley_data, window_length=window_len, polyorder=2)
                else:
                    dynamic_upper = np.array(upper_valley_data)
            else:
                dynamic_upper = np.array(upper_valley_data)

        else:
            dynamic_upper = np.full_like(self.tracker.times, self.fmax_plot.get())
        
        dynamic_upper = np.minimum(dynamic_upper, self.fmax_plot.get())
        
        self.boundary_data = {
            'valley_boundaries': valley_boundaries,
            'dynamic_lower': dynamic_lower,
            'dynamic_upper': dynamic_upper,
            'boundaries_with_edges': [dynamic_lower] + valley_boundaries + [dynamic_upper]
        }
    
    def redetect_harmonics(self):
        """Redetect harmonics with current parameters"""
        if self.tracker is None:
            return
        
        self.detect_harmonics()
        self.update_display(recompute=True)
    
    # ===== MOUSE EVENT HANDLERS =====
    
    def on_press(self, event):
        """Handle mouse press"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        # Right click = undo zoom
        if event.button == 3:
            if self.zoom_stack:
                xlim, ylim = self.zoom_stack.pop()
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
                self.canvas.draw_idle()
            return
        
        # Ctrl+Click = add changepoint
        if event.button == 1 and (event.key == 'control' or event.key == 'ctrl'):
            self.changepoints.append((event.xdata, event.ydata))
            self.changepoints.sort(key=lambda x: x[0])  # Keep sorted by time
            print(f"✓ Added changepoint at t={event.xdata:.3f}s, {len(self.changepoints)} total")
            self.update_display(recompute=False)
            return
        
        # Shift+Click = select region for dragging
        if event.button == 1 and (event.key == 'shift'):
            # Check harmonics first
            closest_harmonic = self.find_nearest_harmonic_contour(event.xdata, event.ydata)
            if closest_harmonic:
                harmonic_num, base_freq = closest_harmonic
                region_idx = self.get_region_for_time(event.xdata)
                self.selected_region = (harmonic_num, region_idx, base_freq, event.ydata)
                print(f"Grabbed H{harmonic_num} region {region_idx} at {event.ydata:.1f} Hz")
                return

            # Check valleys/boundaries
            closest_boundary = self.find_nearest_boundary(event.xdata, event.ydata)
            if closest_boundary:
                boundary_type, boundary_idx = closest_boundary
                region_idx = self.get_region_for_time(event.xdata)
                self.selected_region = (boundary_type, region_idx, boundary_idx, event.ydata)
                print(f"Grabbed {boundary_type} region {region_idx}")
                return


        # Left click - check boundaries first, then harmonics
        if event.button == 1:
            # Check if near valley or boundary
            closest_boundary = self.find_nearest_boundary(event.xdata, event.ydata)
            if closest_boundary:
                boundary_type, boundary_idx = closest_boundary
                self.dragging_harmonic = (boundary_type, boundary_idx, event.ydata)
                print(f"Grabbed {boundary_type} contour")
                return

            # Check if near harmonic ridge
            closest_harmonic = self.find_nearest_harmonic_contour(event.xdata, event.ydata)
            if closest_harmonic:
                harmonic_num, base_freq = closest_harmonic
                self.dragging_harmonic = (harmonic_num, base_freq, event.ydata)
                print(f"Grabbed H{harmonic_num} full contour (click at {event.ydata:.1f} Hz)")
                return
            
            # Otherwise, start drag for zoom
            self.drag_start = (event.xdata, event.ydata)

    def on_motion(self, event):
        """Handle mouse motion"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        # If dragging a region
        if self.selected_region is not None:
            if isinstance(self.selected_region[0], int):
                # Harmonic region
                harmonic_num, region_idx, base_freq, click_freq = self.selected_region
                freq_shift = event.ydata - click_freq
                
                time_start, time_end = self.get_region_bounds(region_idx)
                
                for frame_data in self.harmonic_tracks:
                    if time_start <= frame_data['time'] <= time_end:
                        for harmonic in frame_data['harmonics']:
                            if harmonic['harmonic_number'] == harmonic_num:
                                if 'original_frequency' not in harmonic:
                                    harmonic['original_frequency'] = harmonic['actual_frequency']
                                harmonic['actual_frequency'] = harmonic['original_frequency'] + freq_shift
                                break
            else:
                # Boundary region
                boundary_type, region_idx, boundary_idx, click_freq = self.selected_region
                freq_shift = event.ydata - click_freq
                
                time_start, time_end = self.get_region_bounds(region_idx)
                time_mask = (self.tracker.times >= time_start) & (self.tracker.times <= time_end)
                
                if boundary_type == 'valley':
                    if 'original_valley' not in self.boundary_data:
                        self.boundary_data['original_valley'] = [b.copy() for b in self.boundary_data['valley_boundaries']]
                    self.boundary_data['valley_boundaries'][boundary_idx][time_mask] = \
                        self.boundary_data['original_valley'][boundary_idx][time_mask] + freq_shift
                elif boundary_type == 'lower':
                    if 'original_lower' not in self.boundary_data:
                        self.boundary_data['original_lower'] = self.boundary_data['dynamic_lower'].copy()
                    self.boundary_data['dynamic_lower'][time_mask] = \
                        self.boundary_data['original_lower'][time_mask] + freq_shift
                elif boundary_type == 'upper':
                    if 'original_upper' not in self.boundary_data:
                        self.boundary_data['original_upper'] = self.boundary_data['dynamic_upper'].copy()
                    self.boundary_data['dynamic_upper'][time_mask] = \
                        self.boundary_data['original_upper'][time_mask] + freq_shift
            
            self.update_display(recompute=False)
            return
        
        # If dragging full contour
        if self.dragging_harmonic is not None:
            if isinstance(self.dragging_harmonic[0], int):
                # Harmonic
                harmonic_num, base_freq, click_freq = self.dragging_harmonic
                freq_shift = event.ydata - click_freq
                
                for frame_data in self.harmonic_tracks:
                    for harmonic in frame_data['harmonics']:
                        if harmonic['harmonic_number'] == harmonic_num:
                            if 'original_frequency' not in harmonic:
                                harmonic['original_frequency'] = harmonic['actual_frequency']
                            harmonic['actual_frequency'] = harmonic['original_frequency'] + freq_shift
                            break
            else:
                # Boundary
                boundary_type, boundary_idx, click_freq = self.dragging_harmonic
                freq_shift = event.ydata - click_freq
                
                if boundary_type == 'valley':
                    if 'original_valley' not in self.boundary_data:
                        self.boundary_data['original_valley'] = [b.copy() for b in self.boundary_data['valley_boundaries']]
                    self.boundary_data['valley_boundaries'][boundary_idx] = \
                        self.boundary_data['original_valley'][boundary_idx] + freq_shift
                elif boundary_type == 'lower':
                    if 'original_lower' not in self.boundary_data:
                        self.boundary_data['original_lower'] = self.boundary_data['dynamic_lower'].copy()
                    self.boundary_data['dynamic_lower'] = \
                        self.boundary_data['original_lower'] + freq_shift
                elif boundary_type == 'upper':
                    if 'original_upper' not in self.boundary_data:
                        self.boundary_data['original_upper'] = self.boundary_data['dynamic_upper'].copy()
                    self.boundary_data['dynamic_upper'] = \
                        self.boundary_data['original_upper'] + freq_shift
            
            self.update_display(recompute=False)
            return
        
        # Otherwise, draw zoom rectangle
        if self.drag_start is None:
            return
        
        if self.drag_rect is not None:
            self.drag_rect.remove()
            self.drag_rect = None
        
        x0, y0 = self.drag_start
        width = event.xdata - x0
        height = event.ydata - y0
        
        self.drag_rect = self.ax.add_patch(
            plt.Rectangle((x0, y0), width, height,
                        fill=False, edgecolor='yellow', linewidth=2, linestyle='--')
        )
        
        self.zoom_info_label.config(text=f"Time: {abs(width):.3f}s | Freq: {abs(height):.1f} Hz")
        self.canvas.draw_idle()



    def on_release(self, event):
        """Handle mouse release"""
        try:
            # Finalize region correction
            if self.selected_region is not None:
                if isinstance(self.selected_region[0], int):
                    # Harmonic region
                    harmonic_num, region_idx, base_freq, click_freq = self.selected_region
                    
                    if event.ydata is not None:
                        freq_shift = event.ydata - click_freq
                        time_start, time_end = self.get_region_bounds(region_idx)
                        
                        # Record corrections for this region
                        for frame_data in self.harmonic_tracks:
                            if time_start <= frame_data['time'] <= time_end:
                                for harmonic in frame_data['harmonics']:
                                    if harmonic['harmonic_number'] == harmonic_num:
                                        old_freq = harmonic.get('original_frequency', harmonic['actual_frequency'])
                                        new_freq = harmonic['actual_frequency']
                                        
                                        if abs(new_freq - old_freq) > 1.0:
                                            self.harmonic_corrections.append({
                                                'time': float(frame_data['time']),
                                                'harmonic_num': int(harmonic_num),
                                                'old_freq': float(old_freq),
                                                'new_freq': float(new_freq),
                                                'region': int(region_idx)
                                            })
                                        
                                        if 'original_frequency' in harmonic:
                                            del harmonic['original_frequency']
                                        break
                        
                        self.changes_made = True
                        print(f"✓ Shifted H{harmonic_num} region {region_idx} by {freq_shift:+.1f} Hz")
                else:
                    # Boundary region
                    boundary_type, region_idx, boundary_idx, click_freq = self.selected_region
                    
                    if event.ydata is not None:
                        freq_shift = event.ydata - click_freq
                        
                        # Clean up original storage
                        if 'original_valley' in self.boundary_data:
                            del self.boundary_data['original_valley']
                        if 'original_lower' in self.boundary_data:
                            del self.boundary_data['original_lower']
                        if 'original_upper' in self.boundary_data:
                            del self.boundary_data['original_upper']
                        
                        self.changes_made = True
                        print(f"✓ Shifted {boundary_type} region {region_idx} by {freq_shift:+.1f} Hz")
                
                self.selected_region = None
                self.update_display(recompute=False)
                self.update_correction_info()
                return
            
            # Finalize full contour correction
            if self.dragging_harmonic is not None:
                if isinstance(self.dragging_harmonic[0], int):
                    # Harmonic contour
                    harmonic_num, base_freq, click_freq = self.dragging_harmonic
                    
                    if event.ydata is not None:
                        freq_shift = event.ydata - click_freq
                        
                        # Record correction for each time point
                        for frame_data in self.harmonic_tracks:
                            for harmonic in frame_data['harmonics']:
                                if harmonic['harmonic_number'] == harmonic_num:
                                    old_freq = harmonic.get('original_frequency', harmonic['actual_frequency'])
                                    new_freq = harmonic['actual_frequency']
                                    
                                    if abs(new_freq - old_freq) > 1.0:
                                        self.harmonic_corrections.append({
                                            'time': float(frame_data['time']),
                                            'harmonic_num': int(harmonic_num),
                                            'old_freq': float(old_freq),
                                            'new_freq': float(new_freq)
                                        })
                                    
                                    if 'original_frequency' in harmonic:
                                        del harmonic['original_frequency']
                                    break
                        
                        self.changes_made = True
                        print(f"✓ Shifted H{harmonic_num} full contour by {freq_shift:+.1f} Hz")
                else:
                    # Boundary contour
                    boundary_type, boundary_idx, click_freq = self.dragging_harmonic
                    
                    if event.ydata is not None:
                        freq_shift = event.ydata - click_freq
                        
                        # Clean up original storage
                        if 'original_valley' in self.boundary_data:
                            del self.boundary_data['original_valley']
                        if 'original_lower' in self.boundary_data:
                            del self.boundary_data['original_lower']
                        if 'original_upper' in self.boundary_data:
                            del self.boundary_data['original_upper']
                        
                        self.changes_made = True
                        print(f"✓ Shifted {boundary_type} boundary by {freq_shift:+.1f} Hz")
                
                self.dragging_harmonic = None
                self.update_display(recompute=False)
                self.update_correction_info()
                return
            
            # Handle zoom
            if self.drag_start is None:
                return
            
            if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
                self.drag_start = None
                self.zoom_info_label.config(text="")
                if self.drag_rect is not None:
                    self.drag_rect.remove()
                    self.drag_rect = None
                    self.canvas.draw_idle()
                return
            
            x0, y0 = self.drag_start
            x1, y1 = event.xdata, event.ydata
            
            if self.drag_rect is not None:
                self.drag_rect.remove()
                self.drag_rect = None
            
            drag_dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            
            if drag_dist < 0.05:
                self.drag_start = None
                self.zoom_info_label.config(text="")
                return
            
            new_xlim = sorted([x0, x1])
            new_ylim = sorted([y0, y1])
            
            x_range = new_xlim[1] - new_xlim[0]
            y_range = new_ylim[1] - new_ylim[0]
            
            if x_range < 0.01 or y_range < 10:
                self.drag_start = None
                self.zoom_info_label.config(text="")
                return
            
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            self.zoom_stack.append((current_xlim, current_ylim))
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.canvas.draw_idle()
            
            self.drag_start = None
            self.zoom_info_label.config(text="")
            
        except Exception as e:
            print(f"ERROR in on_release: {e}")
            import traceback
            traceback.print_exc()
            self.drag_start = None
            self.selected_region = None
            if self.drag_rect is not None:
                try:
                    self.drag_rect.remove()
                except:
                    pass
                self.drag_rect = None


    def on_scroll(self, event):
        """Handle mouse wheel zoom"""
        try:
            if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
                return
            
            import sys
            if sys.platform == 'win32':
                import ctypes
                is_ctrl = bool(ctypes.windll.user32.GetKeyState(0x11) & 0x8000)
                is_shift = bool(ctypes.windll.user32.GetKeyState(0x10) & 0x8000)
            else:
                key = getattr(event, 'key', None)
                is_ctrl = (key == 'control')
                is_shift = (key == 'shift')
            
            is_ctrlshift = is_ctrl and is_shift
            
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            zoom_factor = 0.8 if event.button == 'up' else 1.25
            
            if is_ctrlshift:
                # Vertical zoom
                ydata = event.ydata
                y_range = (ylim[1] - ylim[0]) * zoom_factor
                y_center_ratio = (ydata - ylim[0]) / (ylim[1] - ylim[0])
                new_ylim = (ydata - y_range * y_center_ratio, ydata + y_range * (1 - y_center_ratio))
                self.ax.set_ylim(new_ylim)
            elif is_ctrl:
                # Horizontal zoom
                xdata = event.xdata
                x_range = (xlim[1] - xlim[0]) * zoom_factor
                x_center_ratio = (xdata - xlim[0]) / (xlim[1] - xlim[0])
                new_xlim = (xdata - x_range * x_center_ratio, xdata + x_range * (1 - x_center_ratio))
                self.ax.set_xlim(new_xlim)
            elif is_shift:
                # Horizontal pan
                x_range = xlim[1] - xlim[0]
                pan_amount = x_range * 0.1
                if event.button == 'up':
                    new_xlim = (xlim[0] + pan_amount, xlim[1] + pan_amount)
                else:
                    new_xlim = (xlim[0] - pan_amount, xlim[1] - pan_amount)
                self.ax.set_xlim(new_xlim)
            else:
                # Vertical pan
                y_range = ylim[1] - ylim[0]
                pan_amount = y_range * 0.1
                if event.button == 'up':
                    new_ylim = (ylim[0] + pan_amount, ylim[1] + pan_amount)
                else:
                    new_ylim = (ylim[0] - pan_amount, ylim[1] - pan_amount)
                self.ax.set_ylim(new_ylim)
            
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"ERROR in on_scroll: {e}")
            import traceback
            traceback.print_exc()
    
    def find_nearest_harmonic_contour(self, click_time, click_freq):
        """Find nearest harmonic contour to click location
        
        Returns: (harmonic_num, mean_freq) or None
        """
        if not self.harmonic_tracks:
            return None
        
        # For each harmonic, compute distance to its contour
        harmonic_distances = {}
        
        for harm_num in range(1, self.max_harmonics.get() + 1):
            harm_freqs = []
            harm_times = []
            
            for frame_data in self.harmonic_tracks:
                for harmonic in frame_data['harmonics']:
                    if harmonic['harmonic_number'] == harm_num:
                        harm_freqs.append(harmonic['actual_frequency'])
                        harm_times.append(frame_data['time'])
                        break
            
            if not harm_freqs:
                continue
            
            # Find frequency at clicked time (interpolate)
            freq_at_time = np.interp(click_time, harm_times, harm_freqs)
            distance = abs(freq_at_time - click_freq)
            
            if distance < 200:  # Within 200 Hz threshold
                harmonic_distances[harm_num] = (distance, np.mean(harm_freqs))
        
        if not harmonic_distances:
            return None
        
        # Return closest harmonic
        closest_num = min(harmonic_distances.keys(), key=lambda k: harmonic_distances[k][0])
        return (closest_num, harmonic_distances[closest_num][1])



    def find_nearest_boundary(self, click_time, click_freq):
        """Find nearest valley or dynamic boundary to click location
        
        Returns: (boundary_type, boundary_idx) or None
        boundary_type can be 'valley', 'lower', or 'upper'
        """
        if not self.boundary_data:
            return None
        
        min_dist = float('inf')
        closest = None
        threshold = 200  # Hz
        
        # Check valleys
        for i, boundary in enumerate(self.boundary_data['valley_boundaries']):
            freq_at_time = np.interp(click_time, self.tracker.times, boundary)
            dist = abs(freq_at_time - click_freq)
            
            if dist < threshold and dist < min_dist:
                min_dist = dist
                closest = ('valley', i)
        
        # Check lower boundary
        freq_at_time = np.interp(click_time, self.tracker.times, self.boundary_data['dynamic_lower'])
        dist = abs(freq_at_time - click_freq)
        if dist < threshold and dist < min_dist:
            min_dist = dist
            closest = ('lower', 0)
        
        # Check upper boundary
        freq_at_time = np.interp(click_time, self.tracker.times, self.boundary_data['dynamic_upper'])
        dist = abs(freq_at_time - click_freq)
        if dist < threshold and dist < min_dist:
            min_dist = dist
            closest = ('upper', 0)
        
        return closest

    def get_region_for_time(self, time):
        """Get region index for a given time based on changepoints"""
        if not self.changepoints:
            return 0
        
        for i, (cp_time, _) in enumerate(self.changepoints):
            if time < cp_time:
                return i
        
        return len(self.changepoints)

    def get_region_bounds(self, region_idx):
        """Get time bounds for a region"""
        if not self.changepoints:
            return (0, self.tracker.times[-1])
        
        if region_idx == 0:
            time_start = 0
            time_end = self.changepoints[0][0]
        elif region_idx >= len(self.changepoints):
            time_start = self.changepoints[-1][0]
            time_end = self.tracker.times[-1]
        else:
            time_start = self.changepoints[region_idx - 1][0]
            time_end = self.changepoints[region_idx][0]
        
        return (time_start, time_end)



















    # ===== DISPLAY METHODS =====
    
    def update_display(self, recompute=False):
        """Update the display"""
        try:
            if self.tracker is None:
                return
            
            if recompute or self.spec_image is None:
                # Full redraw
                self.ax.clear()
                
                # Plot spectrogram
                extent = [
                    self.tracker.times[0],
                    self.tracker.times[-1],
                    self.tracker.freqs[0],
                    self.tracker.freqs[-1]
                ]
                
                self.spec_image = self.ax.imshow(
                    self.tracker.log_magnitude,
                    aspect='auto',
                    origin='lower',
                    extent=extent,
                    cmap='magma',
                    interpolation='bilinear'
                )
                
                self.ax.set_xlabel('Time (s)', fontsize=8)
                self.ax.set_ylabel('Frequency (Hz)', fontsize=8)
                self.ax.set_ylim(self.fmin_plot.get(), self.fmax_plot.get())
            
            else:
                # Quick update - remove overlay elements
                for line in self.ax.lines[:]:
                    line.remove()
                
                collections_to_remove = [c for c in self.ax.collections 
                                        if isinstance(c, matplotlib.collections.PathCollection)]
                for collection in collections_to_remove:
                    collection.remove()
            
            # Draw valley boundaries
            if self.show_valleys.get() and self.boundary_data:
                valley_colors = ['cyan', 'magenta', 'lime', 'white', 'gold']
                
                for i, boundary in enumerate(self.boundary_data['valley_boundaries']):
                    self.ax.plot(self.tracker.times, boundary, ':', #dotted line
                               color=valley_colors[i % len(valley_colors)],
                               linewidth=2, alpha=0.7, label=f'Valley {i+1}')
                
                # Dynamic boundaries
                self.ax.plot(self.tracker.times, self.boundary_data['dynamic_lower'],
                           '--', color='red', linewidth=2, alpha=0.7, label='Lower')
                self.ax.plot(self.tracker.times, self.boundary_data['dynamic_upper'],
                           '--', color='orange', linewidth=2, alpha=0.7, label='Upper')
            
            # Draw harmonic ridges
            if self.show_harmonics.get() and self.harmonic_tracks:
                harmonic_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown']
                
                for harm_num in range(1, self.max_harmonics.get() + 1):
                    harm_times = []
                    harm_freqs = []
                    
                    for frame_data in self.harmonic_tracks:
                        for harmonic in frame_data['harmonics']:
                            if harmonic['harmonic_number'] == harm_num:
                                harm_times.append(frame_data['time'])
                                harm_freqs.append(harmonic['actual_frequency'])
                    
                    if harm_times and harm_num <= len(harmonic_colors):
                        self.ax.plot(harm_times, harm_freqs, '-', # solid line
                                   color=harmonic_colors[harm_num-1],
                                   markersize=3, linewidth=2, alpha=0.7,
                                   label=f'H{harm_num}')
            

            # Draw changepoints
            if self.changepoints:
                for cp_time, cp_freq in self.changepoints:
                    self.ax.axvline(cp_time, color='teal', linestyle=':', linewidth=2, alpha=0.7)
                    self.ax.scatter(cp_time, cp_freq, c='teal', marker='o', s=100, 
                                edgecolors='white', linewidths=2, zorder=10)






            # Update title
            filename = self.audio_files[self.current_file_idx].name
            save_marker = "" if self.changes_made else "✓ "
            f0_str = f"F0={self.fundamental_freq:.1f} Hz" if self.fundamental_freq else "No F0"
            
            self.ax.set_title(f'{save_marker}{filename} | {f0_str} | '
                            f'{len(self.harmonic_corrections)} corrections', fontsize=9)
            
            self.canvas.draw()
            self.update_stats()
            
        except Exception as e:
            print(f"ERROR in update_display: {e}")
            import traceback
            traceback.print_exc()
    
    def update_display_range(self):
        """Update frequency display range"""
        if self.tracker is None:
            return
        self.ax.set_ylim(self.fmin_plot.get(), self.fmax_plot.get())
        self.canvas.draw_idle()
    
    def reset_zoom(self):
        """Reset zoom to full view"""
        if self.tracker is None:
            return
        
        self.zoom_stack = []
        self.ax.set_xlim(0, self.tracker.times[-1])
        self.ax.set_ylim(self.fmin_plot.get(), self.fmax_plot.get())
        self.canvas.draw_idle()
    
    # ===== STATISTICS =====
    
    def update_stats(self):
        """Update statistics display"""
        if self.fundamental_freq is None:
            self.stats_label.config(text="No harmonics detected")
            return
        
        n_harmonics = len(self.harmonic_series)
        n_corrections = len(self.harmonic_corrections)
        
        if n_corrections > 0:
            freq_errors = [abs(c['new_freq'] - c['old_freq']) for c in self.harmonic_corrections]
            mean_error = np.mean(freq_errors)
            max_error = np.max(freq_errors)
            
            self.stats_label.config(
                text=f"F0: {self.fundamental_freq:.1f} Hz | Harmonics: {n_harmonics}\n"
                     f"Corrections: {n_corrections} | Mean error: {mean_error:.1f} Hz | Max: {max_error:.1f} Hz"
            )
        else:
            self.stats_label.config(
                text=f"F0: {self.fundamental_freq:.1f} Hz | Harmonics: {n_harmonics}\n"
                     f"Corrections: 0"
            )
    
    def update_correction_info(self):
        """Update correction count"""
        self.correction_info.config(text=f"Corrections: {len(self.harmonic_corrections)}")
    
    def update_prominence_label(self):
        """Update prominence label"""
        self.prom_label.config(text=f"{self.prominence_threshold.get():.1f}")



    def on_smoothing_change(self, value):
        """Handle smoothing slider change"""
        val = int(float(value))
        # Ensure odd number for savgol_filter
        if val % 2 == 0:
            val += 1
            self.curve_smoothing_window.set(val)
        self.smooth_label.config(text=f"{val}")
        
        # Recompute boundaries with new smoothing
        if self.tracker is not None and self.harmonic_tracks:
            self.compute_boundary_data()
            self.detect_harmonics()

            self.update_display(recompute=True)


    def update_smoothing_label(self):
        """Update smoothing label (without redraw)"""
        val = int(float(self.curve_smoothing_window.get()))
        if val % 2 == 0:
            val += 1
            self.curve_smoothing_window.set(val)
        self.smooth_label.config(text=f"{val}")



    def update_progress(self):
        """Update file progress"""
        self.file_number_entry.delete(0, tk.END)
        self.file_number_entry.insert(0, str(self.current_file_idx + 1))
        self.file_total_label.config(text=f"/ {len(self.audio_files)}")
        self.file_label.config(text=self.audio_files[self.current_file_idx].name)
    
    def print_debug_info(self):
        """Print debug information"""
        print("\n" + "="*50)
        print("DEBUG INFO")
        print("="*50)
        print(f"Audio loaded: {self.tracker is not None}")
        if self.tracker:
            print(f"Audio length: {len(self.tracker.y) / self.tracker.sr:.2f}s")
        print(f"Current file: {self.current_file_idx + 1}/{len(self.audio_files)}")
        print(f"F0: {self.fundamental_freq}")
        print(f"Harmonics: {len(self.harmonic_series)}")
        print(f"Corrections: {len(self.harmonic_corrections)}")
        print(f"Zoom stack: {len(self.zoom_stack)}")
        print("="*50 + "\n")
    
    # ===== ACTION METHODS =====
    
    def undo_last_correction(self):
        """Remove last correction"""
        if self.harmonic_corrections:
            removed = self.harmonic_corrections.pop()
            self.changes_made = True
            
            # Restore original frequency
            time_idx = removed['time_idx']
            harmonic_num = removed['harmonic_num']
            old_freq = removed['old_freq']
            
            for frame_data in self.harmonic_tracks:
                if abs(frame_data['time'] - self.tracker.times[time_idx]) < 0.001:
                    for harmonic in frame_data['harmonics']:
                        if harmonic['harmonic_number'] == harmonic_num:
                            harmonic['actual_frequency'] = old_freq
                            break
                    break
            
            self.update_display(recompute=False)
            self.update_correction_info()
            print(f"↶ Undid correction: H{harmonic_num} at t={removed['time']:.3f}s")
    
    def clear_all_corrections(self):
        """Clear all corrections"""
        if self.harmonic_corrections and messagebox.askyesno("Clear", "Remove all corrections?"):
            self.harmonic_corrections = []
            self.changes_made = True
            
            # Redetect to restore original frequencies
            self.detect_harmonics()
            self.update_display(recompute=True)
            self.update_correction_info()



    def clear_changepoints(self):
        """Clear all changepoints"""
        if self.changepoints and messagebox.askyesno("Clear", "Remove all changepoints?"):
            self.changepoints = []
            self.update_display(recompute=False)
            print("✓ Cleared changepoints")










    # ===== PLAYBACK CONTROLS =====
    
    def play_audio(self):
        """Play current audio"""
        if self.tracker and self.tracker.y is not None:
            pysoniq.set_gain(self.playback_gain.get())
            pysoniq.play(self.tracker.y, self.tracker.sr)
    
    def pause_audio(self):
        """Pause audio"""
        if pysoniq.is_paused():
            pysoniq.resume()
            if hasattr(self, 'pause_button'):
                self.pause_button.config(bg='yellow')
        else:
            pysoniq.pause()
            if hasattr(self, 'pause_button'):
                self.pause_button.config(bg='orange')
    
    def stop_audio(self):
        """Stop audio"""
        pysoniq.stop()
    
    def toggle_loop(self):
        """Toggle loop mode"""
        self.loop_enabled = not self.loop_enabled
        pysoniq.set_loop(self.loop_enabled)
        
        if self.loop_enabled:
            self.loop_button.config(bg='orange', relief=tk.SUNKEN)
        else:
            self.loop_button.config(bg='lightblue', relief=tk.RAISED)
    
    def update_gain_label(self):
        """Update gain label"""
        gain = self.playback_gain.get()
        gain_percent = int(gain * 100)
        self.gain_label.config(text=f"{gain_percent}%")
        pysoniq.set_gain(gain)
    
    # ===== FILE NAVIGATION =====
    
    def open_annotation_location(self):
        """Open annotation file location"""
        annotation_file = self.get_annotation_file()
        self.open_file_location(annotation_file)
    
    def open_save_location(self):
        """Open save directory"""
        self.open_file_location(self.annotation_dir)
    
    def open_file_location(self, path):
        """Open file/directory in system explorer"""
        if path is None:
            messagebox.showinfo("No Location", "No location set")
            return
        
        import subprocess
        import sys
        
        path = Path(path)
        
        if path.is_file():
            location = path.parent
        elif path.is_dir():
            location = path
        else:
            messagebox.showinfo("Not Found", f"Location does not exist:\n{path}")
            return
        
        try:
            if sys.platform == 'win32':
                os.startfile(str(location))
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(location)])
            else:
                subprocess.run(['xdg-open', str(location)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open location: {e}")
    
    def jump_to_file(self):
        """Jump to specific file number"""
        try:
            file_num = int(self.file_number_entry.get())
            
            if 1 <= file_num <= len(self.audio_files):
                if self.changes_made:
                    self.save_annotations()
                
                self.current_file_idx = file_num - 1
                self.load_current_file()
            else:
                messagebox.showwarning("Invalid File Number", 
                                      f"Enter number between 1 and {len(self.audio_files)}")
                self.update_progress()
        except ValueError:
            messagebox.showwarning("Invalid Input", "Enter valid number")
            self.update_progress()
    
    def previous_file(self):
        """Navigate to previous file"""
        if not self.audio_files:
            return
        
        was_looping = pysoniq.is_looping()
        was_playing = was_looping
        
        if self.changes_made:
            self.save_annotations()
        
        pysoniq.stop()
        
        self.current_file_idx = (self.current_file_idx - 1) % len(self.audio_files)
        self.load_current_file()
        
        if was_playing:
            self.play_audio()
    
    def next_file(self):
        """Navigate to next file"""
        if not self.audio_files:
            return
        
        was_looping = pysoniq.is_looping()
        was_playing = was_looping
        
        if self.changes_made:
            self.save_annotations()
        
        pysoniq.stop()
        
        self.current_file_idx = (self.current_file_idx + 1) % len(self.audio_files)
        self.load_current_file()
        
        if was_playing:
            self.play_audio()
    
    # ===== ANNOTATION SAVE/LOAD =====
    
    def save_annotations(self):
        """Save harmonic corrections to JSON"""
        try:
            if not self.audio_files or self.annotation_dir is None:
                return
            
            annotation_file = self.get_annotation_file()
            
            # Extract original detections for ML pipeline
            original_detections = []
            for frame_data in self.harmonic_tracks:
                for harmonic in frame_data['harmonics']:
                    original_detections.append({
                        'time': float(frame_data['time']),
                        'harmonic_num': int(harmonic['harmonic_number']),
                        'frequency': float(harmonic['actual_frequency']),
                        'magnitude': float(harmonic['magnitude'])
                    })
            
            data = {
                'audio_file': str(self.audio_files[self.current_file_idx]),
                'harmonic_corrections': self.harmonic_corrections,
                'original_detections': original_detections,
                'metadata': {
                    'sr': self.tracker.sr,
                    'n_fft': self.n_fft.get(),
                    'hop_length': self.hop_length.get(),
                    'f0': float(self.fundamental_freq) if self.fundamental_freq else None,
                    'num_harmonics': len(self.harmonic_series)
                }
            }
            
            with open(annotation_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.changes_made = False
            self.update_display(recompute=False)
            print(f"✓ Saved {len(self.harmonic_corrections)} corrections to {annotation_file.name}")
            
        except Exception as e:
            print(f"ERROR saving annotations: {e}")
            import traceback
            traceback.print_exc()


# ===== MAIN ENTRY POINT =====

def main():
    """Entry point for harmonic annotator"""
    root = tk.Tk()
    app = HarmonicAnnotator(root)
    root.geometry("1400x800")
    root.mainloop()


if __name__ == "__main__":
    main()