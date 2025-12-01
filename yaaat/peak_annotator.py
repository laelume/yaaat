# Remember to implement this at some point
# from jellyfish.utils.jelly_funcs import make_daily_directory
# daily_dir = make_daily_directory()

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
from scipy.signal import spectrogram, welch, find_peaks

import pysoniq

try: 
    from yaaat import utils
except ImportError:
    print("/utils subdir does not exist")
    import utils


class PeakAnnotator:
    """Interactive tool for annotating spectral peaks on dual-resolution spectrogram+PSD display"""
    
    def __init__(self, root):
        self.root = root
        # Only set title if root is actually a Tk window (not a Frame)
        if isinstance(root, tk.Tk):
            self.root.title("Peak Annotator")
        
        # Audio and spectrogram data
        self.audio_files = []
        self.current_file_idx = 0
        self.y = None
        self.sr = None
        self.S_db = None  # Vertical spectrogram
        self.freqs = None  # Spectrogram frequencies
        self.times = None  # Spectrogram times
        self.pfreqs = None  # PSD frequencies
        self.ppsd = None  # PSD values
        
        # Syllable tracking
        self.current_syllable = []  # Points in current syllable being built
        self.syllables = []  # List of completed syllables


        # Peak tracking - main data structures
        self.current_peaks = []  # Peaks currently being annotated (not yet saved)
        self.peak_annotations = []  # All peaks for current file
        
        # Spectrogram parameters (high temporal resolution)
        self.n_fft_spect = tk.IntVar(value=512)
        self.hop_spect = tk.IntVar(value=32)  # Small hop = high temporal resolution
        
        # PSD parameters (high frequency resolution)
        self.n_fft_psd = tk.IntVar(value=1024)
        self.hop_psd = tk.IntVar(value=512)
        
        # Display limits
        self.fmin_display = tk.IntVar(value=500)
        self.fmax_display = tk.IntVar(value=12000)
        
        # Zoom state
        self.zoom_stack = []
        
        # Drag state for zoom
        self.drag_start = None
        self.drag_rect = None
        self.dragging_harmonic = None
        
        # State tracking
        self.changes_made = False
        self.annotation_dir = None
        self.base_audio_dir = None
        
        # Cache the spectrogram image
        self.spec_image = None
        
        # Track totals across files
        self.total_peaks_across_files = 0
        self.total_files_annotated = 0
        self.total_skipped_files = 0  

        
        # Peak detection parameters
        self.auto_detected_peaks = None
        self.peak_prominence = tk.DoubleVar(value=0.1)  # For automatic peak detection
        self.peak_click_threshold_hz = tk.DoubleVar(value=100.0)  # Hz tolerance for click-to-peak
        
        # Visual options
        self.show_auto_peaks = tk.BooleanVar(value=True)  # Show automatically detected peaks
        self.show_freq_guides = tk.BooleanVar(value=False)
        self.show_psd = tk.BooleanVar(value=True)  # Show PSD overlay
        

        # Playback state
        self.playback_gain = tk.DoubleVar(value=1.0)
        self.loop_enabled = False


        self.setup_ui()

        # Auto-load last directory or default test audio
        self.root.after(100, self.auto_load_directory)
    
    def setup_ui(self):
        """Create the user interface"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== LEFT CONTROL PANEL =====
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Scrollable canvas for control panel
        canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create window in canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Update scroll region when frame changes
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scrollable_frame.bind("<Configure>", on_frame_configure)

        # Mousewheel scrolling - bind to canvas only, not globally
        # self.control_mousewheel_bound = False

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        # def bind_mousewheel(event):
        #     if not self.control_mousewheel_bound:
        #         print("DEBUG: bind_mousewheel called")
        #         canvas.bind_all("<MouseWheel>", on_mousewheel)
        #         self.control_mousewheel_bound = True
        # def unbind_mousewheel(event):
        #     if self.control_mousewheel_bound:
        #         print("DEBUG: unbind_mousewheel called")
        #         canvas.unbind_all("<MouseWheel>")
        #         self.control_mousewheel_bound = False
        # canvas.bind("<Enter>", bind_mousewheel)
        # canvas.bind("<Leave>", unbind_mousewheel)
        # Bind to scrollable_frame and all its children
        def bind_to_mousewheel(widget):
            widget.bind("<MouseWheel>", on_mousewheel)
            for child in widget.winfo_children():
                bind_to_mousewheel(child)

        # Initial binding
        bind_to_mousewheel(scrollable_frame)

        # Rebind whenever frame reconfigures (new widgets added)
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            bind_to_mousewheel(scrollable_frame)  # Rebind to new widgets

        scrollable_frame.bind("<Configure>", on_frame_configure)

        # ===== HEADER =====
        ttk.Label(scrollable_frame, text="Peak Annotator", font=('', 10, 'bold')).pack(pady=(0, 2))
        ttk.Label(scrollable_frame, text="Mark spectral peaks on dual-resolution spectrogram+PSD display", 
                  wraplength=400, font=('', 8, 'italic')).pack(padx=5, pady=(0, 3))

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # ===== INSTRUCTIONS =====
        ttk.Label(scrollable_frame, text="Instructions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        instructions = ttk.Label(scrollable_frame, 
                                text="• Click near peak: mark peak\n• Click + Drag: zoom to region\n• Right-click: undo zoom\n• Ctrl + scroll: horizontal zoom", 
                                wraplength=400, font=('', 8))
        instructions.pack(padx=5, pady=(0, 5))

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        
        
        # ===== FILE MANAGEMENT AND PLAYBACK CONTROLS =====

        # Create horizontal frame for load buttons (left) and play button (right)
        file_buttons_frame = ttk.Frame(scrollable_frame)
        file_buttons_frame.pack(fill=tk.X, pady=2)

        # Left side - File Management
        load_buttons_frame = ttk.Frame(file_buttons_frame)
        load_buttons_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        ttk.Label(load_buttons_frame, text="File Management:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Button(load_buttons_frame, text="Load Audio Directory", command=self.load_directory).pack(anchor=tk.W, pady=2)
        ttk.Button(load_buttons_frame, text="Load Test Audio", command=self.load_test_audio).pack(anchor=tk.W, pady=2)

        # Vertical separator (centered)
        ttk.Separator(file_buttons_frame, orient=tk.VERTICAL).grid(row=0, column=1, sticky='ns', padx=10)

        # Right side - Playback Controls
        play_controls_frame = ttk.Frame(file_buttons_frame)
        play_controls_frame.grid(row=0, column=2, sticky='nsew', padx=(5, 0))

        ttk.Label(play_controls_frame, text="Playback Controls:", font=('', 9, 'bold')).pack(anchor=tk.CENTER, pady=(0, 2))

        # Container for buttons and gain
        controls_container = ttk.Frame(play_controls_frame)
        controls_container.pack(anchor=tk.CENTER)

        # Horizontal button layout (left side)
        buttons_row = ttk.Frame(controls_container)
        buttons_row.pack(side=tk.LEFT, padx=(0, 10))

        # Play button
        play_button = tk.Button(buttons_row, text="▶", 
                            command=self.play_audio, 
                            bg='lightgreen', 
                            font=('', 12, 'bold'),
                            width=2,
                            height=1,
                            relief=tk.RAISED,
                            cursor='hand2')
        play_button.pack(side=tk.LEFT, padx=2)

        # Pause button
        self.pause_button = tk.Button(buttons_row, text="⏸", 
                                command=self.pause_audio, 
                                bg='yellow', 
                                font=('', 12, 'bold'),
                                width=2,
                                height=1,
                                relief=tk.RAISED,
                                cursor='hand2')
        self.pause_button.pack(side=tk.LEFT, padx=2)

        # Stop button
        stop_button = tk.Button(buttons_row, text="⏹", 
                            command=self.stop_audio, 
                            bg='lightcoral', 
                            font=('', 12, 'bold'),
                            width=2,
                            height=1,
                            relief=tk.RAISED,
                            cursor='hand2')
        stop_button.pack(side=tk.LEFT, padx=2)

        # Loop button
        self.loop_button = tk.Button(buttons_row, text="⟳", 
                            command=self.toggle_loop, 
                            bg='lightblue', 
                            font=('', 12, 'bold'),
                            width=2,
                            height=1,
                            relief=tk.RAISED,
                            cursor='hand2')
        self.loop_button.pack(side=tk.LEFT, padx=2)

        # Vertical gain fader (right side)
        gain_frame = ttk.Frame(controls_container)
        gain_frame.pack(side=tk.LEFT)

        ttk.Label(gain_frame, text="Gain", font=('', 7)).pack()

        self.playback_gain = tk.DoubleVar(value=1.0)
        gain_scale = ttk.Scale(gain_frame, from_=2.0, to=0.0, 
                            variable=self.playback_gain,
                            orient=tk.VERTICAL,
                            length=60,
                            command=lambda v: self.update_gain_label())
        gain_scale.pack()

        self.gain_label = ttk.Label(gain_frame, text="100%", font=('', 7))
        self.gain_label.pack()

        # Configure grid weights - equal distribution
        file_buttons_frame.columnconfigure(0, weight=1)
        file_buttons_frame.columnconfigure(2, weight=1)

        self.file_label = ttk.Label(scrollable_frame, text="No files loaded", wraplength=400, font=('', 8))
        self.file_label.pack(fill=tk.X, pady=2)
        
        # JSON annotation file button
        self.annotationfile_button = tk.Button(scrollable_frame, text="No annotation file", 
                                        font=('', 8), relief=tk.FLAT, 
                                        fg='blue', cursor='hand2',
                                        command=self.open_annotation_location,
                                        anchor='w')
        self.annotationfile_button.pack(fill=tk.X, pady=2)

        # Save directory button
        self.save_dir_button = tk.Button(scrollable_frame, text="No save directory", 
                                         font=('', 8), relief=tk.FLAT, 
                                         fg='blue', cursor='hand2',
                                         command=self.open_save_location,
                                         anchor='w')
        self.save_dir_button.pack(anchor=tk.W, pady=2)

        # Peak count info
        self.peak_info = ttk.Label(scrollable_frame, text="Peaks: 0 | Total: 0", wraplength=600, font=('', 8), justify=tk.LEFT)
        self.peak_info.pack(pady=2)


        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)


        # # Scrollable Frame (working - keep here)
        # ===== SPECTROGRAM PARAMETERS =====
        # ttk.Label(scrollable_frame, text="Spectrogram (temporal resolution):", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        # # n_fft buttons
        # ttk.Label(scrollable_frame, text="n_fft:", font=('', 8)).pack(anchor=tk.W)
        # nfft_spect_frame = ttk.Frame(scrollable_frame)
        # nfft_spect_frame.pack(fill=tk.X, expand=True, pady=2)
        # self.nfft_spect_buttons = []
        # for nfft in [256, 512, 1024, 2048]:
        #     btn = tk.Button(nfft_spect_frame, text=str(nfft), width=5,command=lambda n=nfft: self.change_nfft_spect(n))
        #     btn.pack(side=tk.LEFT, padx=2)
        #     self.nfft_spect_buttons.append((btn, nfft))
        # # hop buttons
        # ttk.Label(scrollable_frame, text="hop:", font=('', 8)).pack(anchor=tk.W)
        # hop_spect_frame = ttk.Frame(scrollable_frame)
        # hop_spect_frame.pack(fill=tk.X, pady=2)
        # self.hop_spect_buttons = []
        # for hop in [16, 32, 64, 128]:
        #     btn = tk.Button(hop_spect_frame, text=str(hop), width=5,command=lambda h=hop: self.change_hop_spect(h))
        #     btn.pack(side=tk.LEFT, padx=2)
        #     self.hop_spect_buttons.append((btn, hop))
        # ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        # # ===== PSD PARAMETERS =====
        # ttk.Label(scrollable_frame, text="PSD (frequency resolution):", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        # # n_fft PSD buttons
        # ttk.Label(scrollable_frame, text="n_fft:", font=('', 8)).pack(anchor=tk.W)
        # nfft_psd_frame = ttk.Frame(scrollable_frame)
        # nfft_psd_frame.pack(fill=tk.X, pady=2)
        # self.nfft_psd_buttons = []
        # for nfft in [512, 1024, 2048, 4096]:
        #     btn = tk.Button(nfft_psd_frame, text=str(nfft), width=5,command=lambda n=nfft: self.change_nfft_psd(n))
        #     btn.pack(side=tk.LEFT, padx=2)
        #     self.nfft_psd_buttons.append((btn, nfft))
        # # hop PSD buttons  
        # ttk.Label(scrollable_frame, text="hop:", font=('', 8)).pack(anchor=tk.W)
        # hop_psd_frame = ttk.Frame(scrollable_frame)
        # hop_psd_frame.pack(fill=tk.X, pady=2)
        # self.hop_psd_buttons = []
        # for hop in [256, 512, 1024]:
        #     btn = tk.Button(hop_psd_frame, text=str(hop), width=5,command=lambda h=hop: self.change_hop_psd(h))
        #     btn.pack(side=tk.LEFT, padx=2)
        #     self.hop_psd_buttons.append((btn, hop))
        # # Toggle PSD display
        # ttk.Checkbutton(scrollable_frame, text="Show PSD Overlay", variable=self.show_psd, command=self.toggle_psd).pack(anchor=tk.W, pady=2)

        # ===== SPECTROGRAM / PSD PARAMETERS (COLUMNS) =====
        spec_psd_container = ttk.Frame(scrollable_frame)
        spec_psd_container.pack(fill=tk.X, pady=3)

        # LEFT COLUMN - Spectrogram
        spec_col = ttk.Frame(spec_psd_container)
        spec_col.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        ttk.Label(spec_col, text="Spectrogram resolution:", 
                font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        # n_fft
        ttk.Label(spec_col, text="n_fft:", font=('', 8)).pack(anchor=tk.W)
        nfft_spect_frame = ttk.Frame(spec_col)
        nfft_spect_frame.pack(fill=tk.X, pady=2)

        self.nfft_spect_buttons = []
        for nfft in [256, 512, 1024, 2048]:
            btn = tk.Button(nfft_spect_frame, text=str(nfft), width=4,command=lambda n=nfft: self.change_nfft_spect(n))
            btn.pack(side=tk.LEFT, padx=2)
            self.nfft_spect_buttons.append((btn, nfft))

        # hop
        ttk.Label(spec_col, text="hop:", font=('', 8)).pack(anchor=tk.W)
        hop_spect_frame = ttk.Frame(spec_col)
        hop_spect_frame.pack(fill=tk.X, pady=2)

        self.hop_spect_buttons = []
        for hop in [16, 32, 64, 128]:
            btn = tk.Button(hop_spect_frame, text=str(hop), width=4,command=lambda h=hop: self.change_hop_spect(h))
            btn.pack(side=tk.LEFT, padx=2)
            self.hop_spect_buttons.append((btn, hop))


        # VERTICAL SEPARATOR
        ttk.Separator(spec_psd_container, orient=tk.VERTICAL).grid(row=0, column=1, sticky='ns', padx=4)


        # RIGHT COLUMN - PSD
        psd_col = ttk.Frame(spec_psd_container)
        psd_col.grid(row=0, column=2, sticky='nsew')

        ttk.Label(psd_col, text="PSD resolution:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        # n_fft PSD
        ttk.Label(psd_col, text="n_fft:", font=('', 8)).pack(anchor=tk.W)
        nfft_psd_frame = ttk.Frame(psd_col)
        nfft_psd_frame.pack(fill=tk.X, pady=2)

        self.nfft_psd_buttons = []
        for nfft in [512, 1024, 2048, 4096]:
            btn = tk.Button(nfft_psd_frame, text=str(nfft), width=4,command=lambda n=nfft: self.change_nfft_psd(n))
            btn.pack(side=tk.LEFT, padx=2)
            self.nfft_psd_buttons.append((btn, nfft))

        # hop PSD
        ttk.Label(psd_col, text="hop:", font=('', 8)).pack(anchor=tk.W)
        hop_psd_frame = ttk.Frame(psd_col)
        hop_psd_frame.pack(fill=tk.X, pady=2)

        self.hop_psd_buttons = []
        for hop in [256, 512, 1024]:
            btn = tk.Button(hop_psd_frame, text=str(hop), width=4,command=lambda h=hop: self.change_hop_psd(h))
            btn.pack(side=tk.LEFT, padx=2)
            self.hop_psd_buttons.append((btn, hop))

        # Show PSD checkbox
        ttk.Checkbutton(psd_col, text="Show PSD Overlay", variable=self.show_psd, command=self.toggle_psd).pack(anchor=tk.W, pady=2)


        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # ===== DISPLAY PARAMETERS =====

        ttk.Label(scrollable_frame, text="Display Range:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        # Frequency range
        freq_frame = ttk.Frame(scrollable_frame)
        freq_frame.pack(fill=tk.X, pady=2)
        ttk.Label(freq_frame, text="Freq (Hz):", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.fmin_display, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(freq_frame, text="-", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.fmax_display, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(freq_frame, text="↻", width=2, command=self.update_display_range).pack(side=tk.LEFT, padx=2)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)


        # ===== PEAK DETECTION PARAMETERS =====

        ttk.Label(scrollable_frame, text="Peak Detection:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        # Prominence slider
        prom_frame = ttk.Frame(scrollable_frame)
        prom_frame.pack(fill=tk.X, pady=2)
        ttk.Label(prom_frame, text="Prominence:", font=('', 8)).pack(side=tk.LEFT)
        ttk.Scale(prom_frame, from_=0.01, to=0.5, variable=self.peak_prominence, 
                 orient=tk.HORIZONTAL, command=lambda v: self.update_prominence()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.prom_label = ttk.Label(prom_frame, text=f"{self.peak_prominence.get():.2f}", font=('', 8), width=5)
        self.prom_label.pack(side=tk.LEFT)

        # Click threshold
        thresh_frame = ttk.Frame(scrollable_frame)
        thresh_frame.pack(fill=tk.X, pady=2)
        ttk.Label(thresh_frame, text="Click threshold (Hz):", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(thresh_frame, textvariable=self.peak_click_threshold_hz, width=6).pack(side=tk.LEFT, padx=2)

        # Show auto-detected peaks checkbox
        ttk.Checkbutton(scrollable_frame, text="Show Auto-Detected Peaks", 
                       variable=self.show_auto_peaks, command=self.toggle_auto_peaks).pack(anchor=tk.W, pady=2)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)


        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # Vertical and horizontal lines and annotation values
        ttk.Label(scrollable_frame, text="Guides:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(3, 2))
        guides_grid = ttk.Frame(scrollable_frame)
        guides_grid.pack(fill=tk.X, pady=2)

        self.show_time_guides = tk.BooleanVar(value=False)
        self.show_freq_guides = tk.BooleanVar(value=False)
        self.show_all_guides_var = tk.BooleanVar(value=False)
        self.hide_text = tk.BooleanVar(value=False) 

        self.show_bounding_box = tk.BooleanVar(value=False)
        self.bounding_box_shape = tk.StringVar(value='rectangle')

        # Initialize harmonics data structure
        self.harmonics = [
            {'multiplier': tk.DoubleVar(value=2.0), 'show': tk.BooleanVar(value=False), 'label': None, 'color': 'cyan', 'name': '2nd'},
            {'multiplier': tk.DoubleVar(value=3.0), 'show': tk.BooleanVar(value=False), 'label': None, 'color': 'orange', 'name': '3rd'}
        ]

        # 4-column grid layout
        ttk.Checkbutton(guides_grid, text="Time Lines", variable=self.show_time_guides, command=self.toggle_guides).grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Freq Lines", variable=self.show_freq_guides, command=self.toggle_guides).grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Show All", variable=self.show_all_guides_var, command=self.toggle_show_all).grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Hide Text", variable=self.hide_text, command=self.toggle_guides).grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)

        ttk.Checkbutton(guides_grid, text="Bounding Box", variable=self.show_bounding_box, command=self.toggle_guides).grid(row=0, column=1, sticky=tk.W, padx=2, pady=2)
        ttk.Radiobutton(guides_grid, text="Rectangle", variable=self.bounding_box_shape, value='rectangle', command=self.toggle_guides).grid(row=1, column=1, sticky=tk.W, padx=2, pady=2)
        ttk.Radiobutton(guides_grid, text="Polygon", variable=self.bounding_box_shape, value='polygon', command=self.toggle_guides).grid(row=1, column=2, sticky=tk.W, padx=2, pady=2)
        ttk.Radiobutton(guides_grid, text="Ellipse", variable=self.bounding_box_shape, value='ellipse', command=self.toggle_guides).grid(row=1, column=3, sticky=tk.W, padx=2, pady=2)

        # Harmonic controls
        for i, harmonic in enumerate(self.harmonics):
            row = 2 + i
            
            # Checkbox
            ttk.Checkbutton(guides_grid, text=f"Bound {harmonic['name']} Harmonic", 
                        variable=harmonic['show'], command=self.toggle_guides).grid(
                            row=row, column=1, sticky=tk.W, padx=2, pady=2)
            
            # Nudge buttons frame
            nudge_frame = ttk.Frame(guides_grid)
            nudge_frame.grid(row=row, column=2, columnspan=2, sticky=tk.W, padx=2, pady=2)
            
            down_btn = tk.Button(nudge_frame, text="▼", width=3, font=('', 8))
            down_btn.pack(side=tk.LEFT, padx=1)
            down_btn.bind('<ButtonPress-1>', lambda e, idx=i: self.start_continuous_harmonic(idx, 'down'))
            down_btn.bind('<ButtonRelease-1>', lambda e, idx=i: self.stop_continuous_harmonic(idx))
            
            harmonic['label'] = ttk.Label(nudge_frame, text=f"{harmonic['multiplier'].get():.2f}x", width=5, font=('', 8))
            harmonic['label'].pack(side=tk.LEFT, padx=2)
            
            up_btn = tk.Button(nudge_frame, text="▲", width=3, font=('', 8))
            up_btn.pack(side=tk.LEFT, padx=1)
            up_btn.bind('<ButtonPress-1>', lambda e, idx=i: self.start_continuous_harmonic(idx, 'up'))
            up_btn.bind('<ButtonRelease-1>', lambda e, idx=i: self.stop_continuous_harmonic(idx))


        # ===== ACTIONS =====

        ttk.Label(scrollable_frame, text="Actions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        # Action buttons grid
        button_grid = ttk.Frame(scrollable_frame)
        button_grid.pack(pady=2)

        buttons = [
                       
            ("Show Lines", self.toggle_guides),
            ("Load Audio", self.load_directory), 
            ("Remove Peak", self.clear_last_peak),
            ("Next File", self.next_file),
            
            
            ("Reset Disp", self.recompute_display),
            ("Play Audio", self.play_audio),
            ("Clear Peaks", self.clear_all),
            ("Prev File", self.previous_file),

            ("Reset Zoom", self.reset_zoom),
            ("Debug Info", self.print_debug_info),
            ("Autofill Peaks", self.auto_detect_peaks),
            ("Save Anno", self.save_annotations),
        ]

        # Arrange in 4-column grid
        for i, (text, command) in enumerate(buttons):
            row = i // 4
            col = i % 4
            ttk.Button(button_grid, text=text, command=command, width=12).grid(row=row, column=col, padx=2, pady=2, sticky='ew')

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)


        # ===== STATISTICS =====

        ttk.Label(scrollable_frame, text="Statistics:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.stats_label = ttk.Label(scrollable_frame, text="No peaks annotated", 
                                     justify=tk.LEFT, font=('', 8))
        self.stats_label.pack(fill=tk.X, pady=2)

        # ===== RIGHT SPECTROGRAM PANEL =====

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Navigation buttons
        nav_frame = ttk.Frame(plot_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(nav_frame, text="◄ Previous", command=self.previous_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next ►", command=self.next_file).pack(side=tk.LEFT, padx=5)
        
        # File navigation
        file_nav_frame = ttk.Frame(nav_frame)
        file_nav_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(file_nav_frame, text="File:", font=('', 9)).pack(side=tk.LEFT, padx=2)
        self.file_number_entry = ttk.Entry(file_nav_frame, width=6, justify=tk.CENTER)
        self.file_number_entry.pack(side=tk.LEFT, padx=2)
        self.file_total_label = ttk.Label(file_nav_frame, text="/ 0", font=('', 9))
        self.file_total_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(file_nav_frame, text="Go", command=self.jump_to_file, width=4).pack(side=tk.LEFT, padx=2)

        # Bind Enter key
        self.file_number_entry.bind('<Return>', lambda e: self.jump_to_file())

        ttk.Label(nav_frame, text="[Click near peak: mark | Drag: zoom | Right-click: reset zoom | Ctrl+Wheel: zoom]", 
                  font=('', 8, 'italic')).pack(side=tk.RIGHT, padx=10)
        
        # Spectrogram canvas
        self.fig = Figure(figsize=(10, 6))
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Zoom info display
        self.zoom_info_label = ttk.Label(plot_frame, text="", font=('', 8), foreground='blue')
        self.zoom_info_label.pack(pady=(2, 0))

        # Bind events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Initialize empty plot
        self.ax.set_xlabel('Frequency (Hz)', fontsize=8)
        self.ax.set_ylabel('Time (arbitrary units)', fontsize=8)
        self.ax.set_title('Load audio files to begin')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()
        
        # Update button highlights
        self.update_button_highlights()


    # ===== PARAMETER CHANGE METHODS =====
    
    def change_nfft_spect(self, new_nfft):
        """Change spectrogram n_fft and recompute"""
        self.n_fft_spect.set(new_nfft)
        self.update_button_highlights()
        if self.y is not None:
            self.recompute_display()
    
    def change_hop_spect(self, new_hop):
        """Change spectrogram hop and recompute"""
        self.hop_spect.set(new_hop)
        self.update_button_highlights()
        if self.y is not None:
            self.recompute_display()
    
    def change_nfft_psd(self, new_nfft):
        """Change PSD n_fft and recompute"""
        self.n_fft_psd.set(new_nfft)
        self.update_button_highlights()
        if self.y is not None:
            self.recompute_display()
    
    def change_hop_psd(self, new_hop):
        """Change PSD hop and recompute"""
        self.hop_psd.set(new_hop)
        self.update_button_highlights()
        if self.y is not None:
            self.recompute_display()


    def update_button_highlights(self):
        """Highlight currently selected parameters"""
        # Highlight spectrogram n_fft
        for btn, val in self.nfft_spect_buttons:
            if val == self.n_fft_spect.get():
                btn.config(bg='lightgreen', relief=tk.SUNKEN)
            else:
                btn.config(relief=tk.RAISED)
        
        # Highlight spectrogram hop
        for btn, val in self.hop_spect_buttons:
            if val == self.hop_spect.get():
                btn.config(bg='lightblue', relief=tk.SUNKEN)
            else:
                btn.config(relief=tk.RAISED)
        
        # Highlight PSD n_fft
        for btn, val in self.nfft_psd_buttons:
            if val == self.n_fft_psd.get():
                btn.config(bg='lightyellow', relief=tk.SUNKEN)
            else:
                btn.config(relief=tk.RAISED)

        # Highlight PSD hop
        for btn, val in self.hop_psd_buttons:
            if val == self.hop_psd.get():
                btn.config(bg='lightcoral', relief=tk.SUNKEN)
            else:
                btn.config(relief=tk.RAISED)

    
    def update_prominence(self):
        """Update prominence label when slider changes"""
        self.prom_label.config(text=f"{self.peak_prominence.get():.2f}")
        self.auto_detected_peaks = None  # Invalidate cache
        if self.show_auto_peaks.get() and self.y is not None:
            self.update_display(recompute=False)  # Redetect peaks with new prominence
    



    # ===== FILE MANAGEMENT =====
    



    def count_skipped_files(self):
        """Count total skipped files across all annotation files"""
        self.total_skipped_files = 0
        for audio_file in self.audio_files:
            relative_path = audio_file.relative_to(self.base_audio_dir).parent
            filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
            if filename_prefix and filename_prefix != '.':
                label_file = self.label_dir / f"{filename_prefix}_{audio_file.stem}_changepoint_annotations.json"
            else:
                label_file = self.label_dir / f"{audio_file.stem}_changepoint_annotations.json"
            
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        data = json.load(f)
                        if data.get('skipped', False):
                            self.total_skipped_files += 1
                except:
                    pass
        
        print(f"Total skipped files: {self.total_skipped_files}")






    def load_directory(self):
        """Load all .wav files from a directory"""
        directory = filedialog.askdirectory(title="Select Audio Directory")
        if not directory:
            return
        
        # Find all .wav files
        self.audio_files = natsorted(Path(directory).rglob('*.wav'))
        self.base_audio_dir = Path(directory)

        if not self.audio_files:
            messagebox.showwarning("No Files", "No .wav files found")
            return

        # Ask user where to save annotations
        response = messagebox.askyesnocancel(
            "Annotation Save Location",
            "Where do you want to save peak annotations?\n\n"
            "Yes = Choose existing directory\n"
            "No = Create new directory\n"
            "Cancel = Use default location"
        )

        if response is True:  # Yes - choose existing
            save_dir = filedialog.askdirectory(title="Select Annotation Save Directory")
            if save_dir:
                self.annotation_dir = Path(save_dir)
            else:
                return
        elif response is False:  # No - create new
            save_dir = filedialog.askdirectory(title="Select Parent Directory for New Folder")
            if save_dir:
                dataset_name = Path(directory).name
                self.annotation_dir = Path(save_dir) / f"{dataset_name}_peak_annotations"
                self.annotation_dir.mkdir(exist_ok=True)
            else:
                return
        else:  # Cancel - use default
            dataset_name = Path(directory).name
            default_dir = Path.home() / "yaaat_annotations" / f"{dataset_name}_peaks"
            default_dir.mkdir(parents=True, exist_ok=True)
            self.annotation_dir = default_dir

        self.annotation_dir.mkdir(exist_ok=True)
        print(f"Peak annotations will be saved to: {self.annotation_dir}")
        
        # Update save directory button
        self.save_dir_button.config(text=f"📁 {self.annotation_dir}")
        
        self.current_file_idx = 0
        self.count_total_peaks()
        self.count_skipped_files()
        self.load_current_file()
        
        print(f"✓ Loaded {len(self.audio_files)} files")
        utils.save_last_directory(self.base_audio_dir)


    def load_test_audio(self):
        """Load bundled test audio files"""
        from pathlib import Path
        
        # Go up to package dir, then up to repo root, then into test_files
        test_audio_dir = Path(__file__).parent / 'test_files' / 'test_audio' / 'kiwi'
        
        if not test_audio_dir.exists():
            print(f"DEBUG: Looking for test audio at: {test_audio_dir}")
            print(f"DEBUG: Directory exists: {test_audio_dir.exists()}")
            messagebox.showinfo("No Test Data", "Test audio files not found in package")
            return
        
        # Find all .wav files (recursively)
        self.audio_files = natsorted(test_audio_dir.rglob('*.wav'))
        self.base_audio_dir = test_audio_dir
        
        if not self.audio_files:
            print(f"DEBUG: No .wav files found in: {test_audio_dir}")
            messagebox.showwarning("No Files", "No .wav files found in test directory")
            return
        
        # Set up default save directory
        default_dir = Path.home() / "yaaat_annotations" / "test_audio"
        default_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir = default_dir
        
        # Update UI
        self.save_dir_button.config(text=f"📁 {self.label_dir}")
        
        self.current_file_idx = 0
        self.count_total_peaks()
        self.count_skipped_files()
        self.load_current_file()
        
        print(f"✓ Loaded {len(self.audio_files)} test files")
        save_last_directory(self.base_audio_dir)


    def auto_load_directory(self):
        """Auto-load last directory or default test audio on startup"""
        # Try last opened directory first
        last_dir = utils.load_last_directory()
        if last_dir and last_dir.exists():
            print(f"Auto-loading last directory: {last_dir}")
            # Simulate loading without dialog
            self.audio_files = natsorted(last_dir.rglob('*.wav'))
            self.base_audio_dir = last_dir
            if self.audio_files:
                # Use default annotation location
                dataset_name = last_dir.name
                self.annotation_dir = Path.home() / "yaaat_annotations" / f"{dataset_name}_peaks"
                self.annotation_dir.mkdir(parents=True, exist_ok=True)
                self.save_dir_button.config(text=f"📁 {self.annotation_dir}")
                self.current_file_idx = 0
                self.count_total_peaks()
                self.load_current_file()
                return
        
        # Fall back to test audio
        self.load_test_audio()


    def load_current_file(self):
        """Load the current audio file and its peak annotations"""
        if not self.audio_files:
            return
        
        audio_file = self.audio_files[self.current_file_idx]
        print(f"Loading {audio_file.name}...")
        
        # Load audio using pysoniq
        self.y, self.sr = pysoniq.load(str(audio_file))
        if self.y.ndim > 1:
            self.y = np.mean(self.y, axis=1)  # Convert to mono
        
        # Compute dual-resolution display
        self.compute_dual_view()
        self.spec_image = None  # Clear cached image
        
        # Clear all peak states
        self.peak_annotations = []
        self.current_peaks = []
        
        # Load existing annotations
        relative_path = audio_file.relative_to(self.base_audio_dir).parent
        filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
        if filename_prefix and filename_prefix != '.':
            annotation_file = self.annotation_dir / f"{filename_prefix}_{audio_file.stem}_peak_annotations.json"
        else:
            annotation_file = self.annotation_dir / f"{audio_file.stem}_peak_annotations.json"

        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                self.peak_annotations = data.get('peaks', [])
                print(f"✓ Loaded {len(self.peak_annotations)} peaks")
        else:
            print(f"No annotation file found")
        
        self.changes_made = False
        self.zoom_stack = []
        self.update_display(recompute=True)
        self.update_progress()
        
        # Update annotation file
        if annotation_file.exists() and self.peak_annotations:
            self.annotationfile_button.config(text=f"✓ {annotation_file.name}", foreground='green')
        else:
            self.annotationfile_button.config(text=f"→ {annotation_file.name}", foreground='blue')
    
    def compute_dual_view(self):
        """Compute vertical spectrogram + PSD overlay"""
        if self.y is None:
            return
        
        # Compute vertical spectrogram using unified function
        self.S_db, self.freqs, self.times = utils.compute_spectrogram_unified(
            self.y, 
            self.sr, 
            nfft=self.n_fft_spect.get(),
            hop=self.hop_spect.get(),
            fmin=0,
            fmax=self.sr / 2,
            scale='linear',  # Peak annotator uses linear scale
            orientation='vertical'
        )
        
        # Compute PSD (high frequency resolution)
        self.pfreqs, self.ppsd = utils.compute_psd(
            self.y, 
            self.sr,
            nfft_psd=self.n_fft_psd.get(),
            hop_psd=self.hop_psd.get()
        )
        
        print(f"Computed: S_db shape={self.S_db.shape}, freqs={len(self.freqs)}, times={len(self.times)}, psd={len(self.pfreqs)}")
    
    def count_total_peaks(self):
        """Count total peaks across all annotation files"""
        self.total_peaks_across_files = 0
        self.total_files_annotated = 0
        
        for audio_file in self.audio_files:
            relative_path = audio_file.relative_to(self.base_audio_dir).parent
            filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
            if filename_prefix and filename_prefix != '.':
                annotation_file = self.annotation_dir / f"{filename_prefix}_{audio_file.stem}_peak_annotations.json"
            else:
                annotation_file = self.annotation_dir / f"{audio_file.stem}_peak_annotations.json"
            
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    file_peaks = len(data.get('peaks', []))
                    if file_peaks > 0:
                        self.total_peaks_across_files += file_peaks
                        self.total_files_annotated += 1

        print(f"Total peaks: {self.total_peaks_across_files} across {self.total_files_annotated} files")
    

    # # Replaced with below more generalized open_file_location
    # def open_save_directory(self):
    #     """Open the save directory in the system file explorer"""
    #     if self.annotation_dir is None:
    #         messagebox.showinfo("No Directory", "No save directory set. Load audio files first.")
    #         return
        
    #     import subprocess
    #     import sys
        
    #     try:
    #         if sys.platform == 'win32':
    #             os.startfile(str(self.annotation_dir))
    #         elif sys.platform == 'darwin':  # macOS
    #             subprocess.run(['open', str(self.annotation_dir)])
    #         else:  # linux
    #             subprocess.run(['xdg-open', str(self.annotation_dir)])
    #     except Exception as e:
    #         messagebox.showerror("Error", f"Could not open directory: {e}")

    def open_file_location(self, path):
        """Open file or directory location in system file explorer
        
        Args:
            path: Path object or string - can be file or directory
        """
        if path is None:
            messagebox.showinfo("No Location", "No location set yet.")
            return
        
        import subprocess
        import sys
        import os
        
        path = Path(path)
        
        # If it's a file, open its parent directory
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
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(location)])
            else:  # linux
                subprocess.run(['xdg-open', str(location)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open location: {e}")

    def open_annotation_location(self):
        """Open current annotation file location"""
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            print("DEBUG: open_annotation_location called")
            messagebox.showinfo("No File", "Load audio files first.")
            return
        
        audio_file = self.audio_files[self.current_file_idx]
        relative_path = audio_file.relative_to(self.base_audio_dir).parent
        filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
        
        if filename_prefix and filename_prefix != '.':
            annotation_file = self.annotation_dir / f"{filename_prefix}_{audio_file.stem}_peak_annotations.json"
        else:
            annotation_file = self.annotation_dir / f"{audio_file.stem}_peak_annotations.json"
        
        self.open_file_location(annotation_file)

    def open_save_location(self):
        """Open save directory location"""
        print("DEBUG: open_save_location called")
        self.open_file_location(self.annotation_dir)


    # ===== MOUSE EVENT HANDLERS =====
    
    def on_press(self, event):
        """Handle mouse button press"""
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
        
        # Left click - start potential drag or peak marking
        if event.button == 1:
            self.drag_start = (event.xdata, event.ydata)
    
    def on_motion(self, event):
        """Handle mouse motion - draw zoom rectangle"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        if self.drag_start is None:
            return
        
        # Remove previous rectangle
        if self.drag_rect is not None:
            self.drag_rect.remove()
            self.drag_rect = None
        
        # Draw new rectangle
        x0, y0 = self.drag_start
        width = event.xdata - x0
        height = event.ydata - y0
        
        self.drag_rect = self.ax.add_patch(
            plt.Rectangle((x0, y0), width, height,
                        fill=False, edgecolor='yellow', linewidth=2, linestyle='--')
        )

        # Add dimension display
        x_range = abs(width)
        y_range = abs(height)
        self.zoom_info_label.config(text=f"Freq: {x_range:.1f} Hz | Time: {y_range:.1f} units")

        self.canvas.draw_idle()
    
    def on_release(self, event):
        """Handle mouse button release - either zoom or mark peak"""
        try:
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
            
            # Calculate drag distance
            drag_dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            
            # Remove drag rectangle
            if self.drag_rect is not None:
                self.drag_rect.remove()
                self.drag_rect = None
            
            # If drag distance is small, treat as click (mark peak)
            if drag_dist < 50:  # Threshold for click vs drag (in display units)
                # Check if clicking near existing peak to remove it
                if self.remove_nearby_peak(x0, y0):
                    print("Removed nearby peak")
                else:
                    # Find and mark nearest PSD peak
                    self.mark_nearest_peak(x0, y0)
            else:
                # Zoom to selected region
                new_xlim = sorted([x0, x1])
                new_ylim = sorted([y0, y1])
                
                # Prevent extreme zooms
                x_range = new_xlim[1] - new_xlim[0]
                y_range = new_ylim[1] - new_ylim[0]
                
                if x_range < 10 or y_range < 0.1:
                    print(f"! Zoom too small, ignoring")
                    self.drag_start = None
                    self.zoom_info_label.config(text="")
                    return
                
                # Save current view to zoom stack
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
            if self.drag_rect is not None:
                try:
                    self.drag_rect.remove()
                except:
                    pass
                self.drag_rect = None
    
    def on_scroll(self, event):
        """Handle mouse wheel for zoom"""
        try:
            if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
                return

            # Check for Ctrl and Shift keys
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

            # Determine zoom factor
            zoom_factor = 0.8 if event.button == 'up' else 1.25

            if is_ctrlshift:
                # Vertical zoom (time axis)
                ydata = event.ydata
                y_range = (ylim[1] - ylim[0]) * zoom_factor
                y_center_ratio = (ydata - ylim[0]) / (ylim[1] - ylim[0])
                new_ylim = (ydata - y_range * y_center_ratio, ydata + y_range * (1 - y_center_ratio))
                self.ax.set_ylim(new_ylim)
            elif is_ctrl:
                # Horizontal zoom (frequency axis)
                xdata = event.xdata
                x_range = (xlim[1] - xlim[0]) * zoom_factor
                x_center_ratio = (xdata - xlim[0]) / (xlim[1] - xlim[0])
                new_xlim = (xdata - x_range * x_center_ratio, xdata + x_range * (1 - x_center_ratio))
                self.ax.set_xlim(new_xlim)
            elif is_shift:
                # Horizontal pan (frequency axis)
                x_range = xlim[1] - xlim[0]
                pan_amount = x_range * 0.1
                if event.button == 'up':
                    new_xlim = (xlim[0] + pan_amount, xlim[1] + pan_amount)
                else:
                    new_xlim = (xlim[0] - pan_amount, xlim[1] - pan_amount)
                self.ax.set_xlim(new_xlim)
            else:
                # Vertical pan (time axis)
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
    
    # ===== PEAK DETECTION AND ANNOTATION =====
    
    def auto_detect_peaks(self):
        """Automatically detect all peaks in PSD above prominence threshold"""
        if self.ppsd is None:
            return
        
        # Find peaks in PSD within display range
        mask = self.pfreqs <= self.fmax_display.get()
        ppsd_scaled = self.ppsd[mask] * len(self.times)
        
        peak_indices, properties = find_peaks(ppsd_scaled, prominence=self.peak_prominence.get())
        
        # Add all detected peaks to annotations
        for idx in peak_indices:
            freq = self.pfreqs[mask][idx]
            amplitude = ppsd_scaled[idx]
            prominence = properties['prominences'][list(peak_indices).index(idx)]
            
            # Check if peak already exists (avoid duplicates)
            if not any(abs(p['freq'] - freq) < 1.0 for p in self.peak_annotations):
                self.peak_annotations.append({
                    'freq': float(freq),
                    'amplitude_normalized': float(self.ppsd[mask][idx]),
                    'prominence': float(prominence),
                    'auto_detected': True
                })
        
        self.changes_made = True
        self.update_display(recompute=False)
        print(f"✓ Auto-detected {len(peak_indices)} peaks")
    
    def mark_nearest_peak(self, click_freq, click_y):
        """Find and mark the nearest PSD peak to click location"""
        if self.ppsd is None:
            return
        
        # Find PSD peaks within display range
        mask = (self.pfreqs >= self.fmin_display.get()) & (self.pfreqs <= self.fmax_display.get())
        ppsd_scaled = self.ppsd[mask] * len(self.times)
        pfreqs_masked = self.pfreqs[mask]
        
        peak_indices, properties = find_peaks(ppsd_scaled, prominence=self.peak_prominence.get())
        
        if len(peak_indices) == 0:
            print("No peaks detected near click")
            return
        
        # Find nearest peak to click
        peak_freqs = pfreqs_masked[peak_indices]
        distances = np.abs(peak_freqs - click_freq)
        nearest_idx = np.argmin(distances)
        
        # Check if within threshold
        if distances[nearest_idx] > self.peak_click_threshold_hz.get():
            print(f"Nearest peak is {distances[nearest_idx]:.1f} Hz away (threshold: {self.peak_click_threshold_hz.get()} Hz)")
            return
        
        # Get peak properties
        peak_idx_in_scaled = peak_indices[nearest_idx]
        freq = float(pfreqs_masked[peak_idx_in_scaled])
        amplitude = float(ppsd_scaled[peak_idx_in_scaled])
        prominence = float(properties['prominences'][nearest_idx])
        
        # Check if peak already annotated
        if any(abs(p['freq'] - freq) < 1.0 for p in self.peak_annotations):
            print(f"Peak at {freq:.1f} Hz already annotated")
            return
        
        # Add peak annotation (store normalized amplitude, not scaled)
        self.peak_annotations.append({
            'freq': freq,
            'amplitude_normalized': float(self.ppsd[self.pfreqs == freq][0]) if any(self.pfreqs == freq) else 0.0,
            'prominence': prominence,
            'auto_detected': False
        })
        
        self.changes_made = True
        self.update_display(recompute=False)
        print(f"+ Peak at {freq:.1f} Hz (prominence: {prominence:.3f})")
    
    def remove_nearby_peak(self, click_freq, click_y):
        """Remove peak annotation near click location"""
        threshold_hz = self.peak_click_threshold_hz.get()
        
        # Find closest peak within threshold
        min_dist = float('inf')
        closest_idx = None
        
        for i, peak in enumerate(self.peak_annotations):
            dist = abs(peak['freq'] - click_freq)
            if dist < threshold_hz and dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        if closest_idx is not None:
            removed = self.peak_annotations.pop(closest_idx)
            self.changes_made = True
            self.update_display(recompute=False)
            print(f"- Removed peak at {removed['freq']:.1f} Hz")
            return True
        
        return False
    
    # ===== DISPLAY METHODS =====
    
    def update_display(self, recompute=False):
            """Update the display with spectrogram, PSD, and peak annotations"""
            try:
                if self.y is None:
                    return
                
                if recompute or self.spec_image is None:
                    # Full redraw
                    self.ax.clear()
                    
                    # Plot vertical spectrogram
                    freq_mask = self.freqs <= self.fmax_display.get()
                    S_db_display = self.S_db[:, freq_mask]
                    freqs_display = self.freqs[freq_mask]
                    
                    extent = [freqs_display.min(), freqs_display.max(), 0, len(self.times)]
                    self.spec_image = self.ax.imshow(S_db_display, aspect="auto", extent=extent, 
                                                    origin="lower", cmap='magma')
                    
                    # Overlay PSD curve
                    if self.show_psd.get():
                        mask_psd = self.pfreqs <= self.fmax_display.get()
                        ppsd_scaled = self.ppsd[mask_psd] * len(self.times)
                        self.ax.plot(self.pfreqs[mask_psd], ppsd_scaled, 
                                color="red", linewidth=1.5, label="PSD", zorder=5)
                        
                        # Show auto-detected peaks if enabled
                        # if self.show_auto_peaks.get():
                        #     peak_indices, _ = find_peaks(ppsd_scaled, prominence=self.peak_prominence.get())
                        #     self.ax.scatter(self.pfreqs[mask_psd][peak_indices], ppsd_scaled[peak_indices],
                        #                 color='orange', marker='x', s=50, linewidths=2, 
                        #                 label='Auto-detected', zorder=6)
                        if self.show_auto_peaks.get():
                            if self.auto_detected_peaks is None or recompute:
                                self.auto_detected_peaks, _ = find_peaks(ppsd_scaled, prominence=self.peak_prominence.get())
                            self.ax.scatter(self.pfreqs[mask_psd][self.auto_detected_peaks], 
                                        ppsd_scaled[self.auto_detected_peaks],
                                        color='orange', marker='x', s=50, linewidths=2, 
                                        label='Auto-detected', zorder=6)
                    
                    # Set axis limits
                    self.ax.set_xlim(self.fmin_display.get(), self.fmax_display.get())
                    self.ax.set_ylim(0, len(self.times))
                    
                    self.ax.set_xlabel('Frequency (Hz)', fontsize=8)
                    self.ax.set_ylabel('Time (arbitrary units)', fontsize=8)
                
                else:
                    # Quick update - remove only scatter plots, keep PSD line
                    collections_to_remove = [c for c in self.ax.collections 
                                            if isinstance(c, matplotlib.collections.PathCollection)]
                    for collection in collections_to_remove:
                        collection.remove()
                    
                    # Redraw PSD line if enabled
                    if self.show_psd.get():
                        mask_psd = self.pfreqs <= self.fmax_display.get()
                        ppsd_scaled = self.ppsd[mask_psd] * len(self.times)
                        
                        # Remove old PSD line if exists
                        for line in self.ax.lines[:]:
                            if line.get_color() == 'red':
                                line.remove()
                        
                        # Plot fresh PSD
                        self.ax.plot(self.pfreqs[mask_psd], ppsd_scaled, 
                                color="red", linewidth=1.5, label="PSD", zorder=5)
                        
                        # Show auto-detected peaks if enabled
                        # if self.show_auto_peaks.get():
                        #     peak_indices, _ = find_peaks(ppsd_scaled, prominence=self.peak_prominence.get())
                        #     self.ax.scatter(self.pfreqs[mask_psd][peak_indices], ppsd_scaled[peak_indices],
                        #                 color='orange', marker='x', s=50, linewidths=2, 
                        #                 label='Auto-detected', zorder=6)
                        if self.show_auto_peaks.get():
                            if self.auto_detected_peaks is None or recompute:
                                self.auto_detected_peaks, _ = find_peaks(ppsd_scaled, prominence=self.peak_prominence.get())
                            self.ax.scatter(self.pfreqs[mask_psd][self.auto_detected_peaks], 
                                        ppsd_scaled[self.auto_detected_peaks],
                                        color='orange', marker='x', s=50, linewidths=2, 
                                        label='Auto-detected', zorder=6)
                
                # Plot annotated peaks (applies to both recompute and quick update)
                if self.peak_annotations:
                    freqs = [p['freq'] for p in self.peak_annotations]
                    # Scale normalized amplitudes to current display
                    amplitudes = [p.get('amplitude_normalized', 0.0) * len(self.times) for p in self.peak_annotations]

                    
                    self.ax.scatter(freqs, amplitudes, 
                                c='cyan', marker='o', s=100, edgecolors='white', 
                                linewidths=2, label='Annotated', zorder=10)
                    
                    # Show frequency guide lines if enabled
                    if self.show_freq_guides.get():
                        for freq in freqs:
                            self.ax.axvline(x=freq, color='cyan', linestyle='--', 
                                        linewidth=1, alpha=0.5, zorder=4)
                
                # Update title
                filename = self.audio_files[self.current_file_idx].name
                save_marker = "" if self.changes_made else "✓ "
                self.ax.set_title(f'{save_marker}{filename} | {len(self.peak_annotations)} peaks | '
                                f'spec: n_fft={self.n_fft_spect.get()} hop={self.hop_spect.get()} | '
                                f'psd: n_fft={self.n_fft_psd.get()}', fontsize=9)
                
                self.canvas.draw()
                self.update_stats()
                self.update_peak_info()
                
            except Exception as e:
                print(f"ERROR in update_display: {e}")
                import traceback
                traceback.print_exc()
            
    
    def update_display_range(self):
        """Update frequency display range without recomputing"""
        if self.y is None:
            return
        self.ax.set_xlim(self.fmin_display.get(), self.fmax_display.get())
        self.canvas.draw_idle()
    
    def recompute_display(self):
        """Recompute spectrogram and PSD with current parameters"""
        if self.y is None:
            return
        
        # Save current zoom
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        self.compute_dual_view()
        self.spec_image = None
        self.update_display(recompute=True)
        
        # Restore zoom
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()
    
    def toggle_auto_peaks(self):
        """Toggle display of auto-detected peaks"""
        if self.y is not None:
            self.update_display(recompute=False)
    
    def toggle_guides(self):
        """Toggle frequency guide lines and bounding boxes"""
        if self.y is None:
            return
        
        # Remove all existing guide lines and patches
        for line in self.ax.lines[:]:
            # Keep only the PSD line (red)
            if line.get_color() != 'red':
                line.remove()
        
        for patch in self.ax.patches[:]:
            patch.remove()
        
        # Remove all text annotations
        for txt in self.ax.texts[:]:
            txt.remove()
        
        # Redraw with current guide settings
        if self.peak_annotations:
            freqs = [p['freq'] for p in self.peak_annotations]
            # Recalculate amplitudes (y-positions) from current PSD at saved frequencies
            amplitudes = []
            for p in self.peak_annotations:
                closest_idx = np.argmin(np.abs(self.pfreqs - p['freq']))
                amp_normalized = float(self.ppsd[closest_idx])
                amplitudes.append(amp_normalized * len(self.times))
            
            # Time guide lines (horizontal)
            if self.show_time_guides.get():
                for amp in amplitudes:
                    self.ax.axhline(y=amp, color='lime', linestyle='--', linewidth=1.5, alpha=0.5, zorder=4)
                    
                    # Add text if not hidden
                    if not self.hide_text.get():
                        self.ax.text(self.ax.get_xlim()[1] * 0.95, amp,
                                f"{amp:.1f}", color='lime', fontsize=9, fontweight='bold',
                                family='monospace', alpha=0.9,
                                verticalalignment='center', horizontalalignment='right')
            
            # Frequency guide lines (vertical)
            if self.show_freq_guides.get():
                for freq in freqs:
                    self.ax.axvline(x=freq, color='yellow', linestyle='--', linewidth=1.5, alpha=0.5, zorder=4)
                    
                    # Add text if not hidden
                    if not self.hide_text.get():
                        self.ax.text(freq, self.ax.get_ylim()[1] * 0.95,
                                f"{freq:.1f}Hz", color='yellow', fontsize=9, fontweight='bold',
                                family='monospace', alpha=0.9, rotation=90,
                                verticalalignment='top', horizontalalignment='right')
            
            # Bounding boxes
            if self.show_bounding_box.get():
                f_min, f_max = min(freqs), max(freqs)
                a_min, a_max = min(amplitudes), max(amplitudes)
                
                shape_type = self.bounding_box_shape.get()
                
                if shape_type == 'rectangle':
                    shape = plt.Rectangle(
                        (f_min, a_min), f_max - f_min, a_max - a_min,
                        fill=False, edgecolor='white', linewidth=2.5, alpha=0.8, zorder=11
                    )
                elif shape_type == 'ellipse':
                    from matplotlib.patches import Ellipse
                    center_f = (f_min + f_max) / 2
                    center_a = (a_min + a_max) / 2
                    width = f_max - f_min
                    height = a_max - a_min
                    shape = Ellipse(
                        (center_f, center_a), width, height,
                        fill=False, edgecolor='white', linewidth=2.5, alpha=0.8, zorder=11
                    )
                elif shape_type == 'polygon':
                    from matplotlib.patches import Polygon
                    points = [(freq, amp) for freq, amp in zip(freqs, amplitudes)]
                    shape = Polygon(
                        points, closed=True,
                        fill=False, edgecolor='white', linewidth=2.5, alpha=0.8, zorder=11
                    )
                
                self.ax.add_patch(shape)
            
            # Draw harmonics
            for harmonic in self.harmonics:
                if harmonic['show'].get() and self.show_bounding_box.get():
                    multiplier = harmonic['multiplier'].get()
                    harmonic_f_min = f_min * multiplier
                    harmonic_f_max = f_max * multiplier
                    
                    shape_type = self.bounding_box_shape.get()
                    
                    if shape_type == 'rectangle':
                        harmonic_shape = plt.Rectangle(
                            (harmonic_f_min, a_min), harmonic_f_max - harmonic_f_min, a_max - a_min,
                            fill=False, edgecolor=harmonic['color'], linewidth=2, linestyle='-', alpha=0.6, zorder=11
                        )
                    elif shape_type == 'ellipse':
                        from matplotlib.patches import Ellipse
                        harmonic_center_f = (harmonic_f_min + harmonic_f_max) / 2
                        center_a = (a_min + a_max) / 2
                        harmonic_width = harmonic_f_max - harmonic_f_min
                        height = a_max - a_min
                        harmonic_shape = Ellipse(
                            (harmonic_center_f, center_a), harmonic_width, height,
                            fill=False, edgecolor=harmonic['color'], linewidth=2, linestyle='-', alpha=0.6, zorder=11
                        )
                    elif shape_type == 'polygon':
                        from matplotlib.patches import Polygon
                        harmonic_points = [(freq * multiplier, amp) for freq, amp in zip(freqs, amplitudes)]
                        harmonic_shape = Polygon(
                            harmonic_points, closed=True,
                            fill=False, edgecolor=harmonic['color'], linewidth=2, linestyle='-', alpha=0.6, zorder=11
                        )
                    
                    self.ax.add_patch(harmonic_shape)
        
        self.canvas.draw_idle()

    def toggle_psd(self):
        """Toggle PSD overlay display"""
        if self.y is not None:
            self.update_display(recompute=False)


    def toggle_show_all(self):
        if self.show_all_guides_var.get():
            # Show All checked - enable both
            self.show_time_guides.set(True)
            self.show_freq_guides.set(True)
        else:
            # Show All unchecked - disable both
            self.show_time_guides.set(False)
            self.show_freq_guides.set(False)
        self.toggle_guides()

    def nudge_harmonic(self, harmonic_index, direction):
        """Nudge harmonic multiplier up or down"""
        harmonic = self.harmonics[harmonic_index]
        current = harmonic['multiplier'].get()
        
        if direction == 'up':
            harmonic['multiplier'].set(current + 0.01)
        elif direction == 'down' and current > 0.02:
            harmonic['multiplier'].set(current - 0.01)
        
        harmonic['label'].config(text=f"{harmonic['multiplier'].get():.2f}x")
        if harmonic['show'].get():
            self.toggle_guides()

    def start_continuous_harmonic(self, harmonic_index, direction):
        """Start continuous harmonic nudging"""
        self.nudge_harmonic(harmonic_index, direction)
        self.harmonic_repeat_ids = getattr(self, 'harmonic_repeat_ids', {})
        self.harmonic_repeat_ids[harmonic_index] = self.root.after(
            200, self.continue_harmonic, harmonic_index, direction)

    def continue_harmonic(self, harmonic_index, direction):
        """Continue harmonic nudging"""
        self.nudge_harmonic(harmonic_index, direction)
        self.harmonic_repeat_ids[harmonic_index] = self.root.after(
            50, self.continue_harmonic, harmonic_index, direction)

    def stop_continuous_harmonic(self, harmonic_index):
        """Stop continuous harmonic nudging"""
        if hasattr(self, 'harmonic_repeat_ids') and harmonic_index in self.harmonic_repeat_ids:
            self.root.after_cancel(self.harmonic_repeat_ids[harmonic_index])
            del self.harmonic_repeat_ids[harmonic_index]



    def reset_zoom(self):
        """Reset zoom to full view"""
        try:
            if self.y is None:
                return
            
            self.zoom_stack = []
            self.ax.set_xlim(self.fmin_display.get(), self.fmax_display.get())
            self.ax.set_ylim(0, len(self.times))
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"ERROR in reset_zoom: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== STATISTICS AND INFO =====
    
    def update_stats(self):
        """Update peak statistics display"""
        if not self.peak_annotations:
            self.stats_label.config(text="No peaks annotated")
            return
        
        freqs = [p['freq'] for p in self.peak_annotations]
        amps = [p.get('amplitude_normalized', 0.0) for p in self.peak_annotations]
        proms = [p['prominence'] for p in self.peak_annotations]
        
        mean_freq = np.mean(freqs)
        std_freq = np.std(freqs)
        min_freq = np.min(freqs)
        max_freq = np.max(freqs)
        mean_prom = np.mean(proms)
        
        self.stats_label.config(
            text=f"Peak freq: μ={mean_freq:.1f} Hz, σ={std_freq:.1f} Hz\n"
                 f"Range: {min_freq:.1f} - {max_freq:.1f} Hz\n"
                 f"Mean prominence: {mean_prom:.3f}"
        )
    
    def update_peak_info(self):
        """Update peak count info label"""
        self.peak_info.config(
            text=f"Peaks: {len(self.peak_annotations)} | Total: {self.total_peaks_across_files}"
        )
    
    def update_progress(self):
        """Update file progress indicator"""
        self.file_number_entry.delete(0, tk.END)
        self.file_number_entry.insert(0, str(self.current_file_idx + 1))
        self.file_total_label.config(text=f"/ {len(self.audio_files)}")
        self.file_label.config(text=self.audio_files[self.current_file_idx].name)
    
    def print_debug_info(self):
        """Print comprehensive debug information"""
        print("\n" + "="*50)
        print("DEBUG INFO")
        print("="*50)
        print(f"Audio loaded: {self.y is not None}")
        if self.y is not None:
            print(f"Audio length: {len(self.y) / self.sr:.2f}s")
        print(f"Current file: {self.current_file_idx + 1}/{len(self.audio_files)}")
        print(f"Peak annotations: {len(self.peak_annotations)}")
        print(f"Zoom stack depth: {len(self.zoom_stack)}")
        
        if self.y is not None:
            print(f"Spectrogram shape: {self.S_db.shape}")
            print(f"PSD length: {len(self.pfreqs)}")
            print(f"Spec params: n_fft={self.n_fft_spect.get()}, hop={self.hop_spect.get()}")
            print(f"PSD params: n_fft={self.n_fft_psd.get()}")
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        print(f"Current xlim: {xlim}")
        print(f"Current ylim: {ylim}")
        print("="*50 + "\n")
    
    # ===== ACTION METHODS =====
    
    def clear_last_peak(self):
        """Remove the last annotated peak"""
        if self.peak_annotations:
            removed = self.peak_annotations.pop()
            self.changes_made = True
            self.update_display(recompute=False)
            print(f"- Removed peak at {removed['freq']:.1f} Hz")
    
    def clear_all(self):
        """Clear all peak annotations"""
        if self.peak_annotations and messagebox.askyesno("Clear", "Remove all peaks?"):
            self.peak_annotations = []
            self.changes_made = True
            self.update_display(recompute=False)
    


    def play_audio(self):
        """Play the current audio file with gain applied"""
        if self.y is not None:
            
            # Set main gain (will be applied on each loop iteration)
            pysoniq.set_gain(self.playback_gain.get())
            pysoniq.play(self.y, self.sr)
            
    def pause_audio(self):
        """Pause audio playback"""        
        if pysoniq.is_paused():  # This is correct - calling the function
            pysoniq.resume()
            # Update pause button appearance
            if hasattr(self, 'pause_button'):
                self.pause_button.config(bg='yellow')
            # Restore loop button if it was enabled
            import pysoniq.pause as pause_module
            if pause_module.was_looping() and hasattr(self, 'loop_button'):
                self.loop_button.config(bg='orange', relief=tk.SUNKEN)
                self.loop_enabled = True
        else:
            pysoniq.pause()
            # Update pause button appearance
            if hasattr(self, 'pause_button'):
                self.pause_button.config(bg='orange')

    def stop_audio(self):
        """Stop audio playback"""
        print("DEBUG: stop_audio() called from changepoint_annotator.py")       
        pysoniq.stop()
        print("DEBUG: pysoniq.stop() completed from changepoint_annotator.py")
        # Don't reset loop button - just stop playback
        # User can manually disable loop if desired

    def toggle_loop(self):
        """Toggle loop mode"""        
        self.loop_enabled = not self.loop_enabled
        
        # Set loop state FIRST
        pysoniq.set_loop(self.loop_enabled)
        
        # Update button appearance
        if self.loop_enabled:
            self.loop_button.config(bg='orange', relief=tk.SUNKEN)
        else:
            self.loop_button.config(bg='lightblue', relief=tk.RAISED)
        
        # Set loop state
        pysoniq.set_loop(self.loop_enabled)
        print(f"Loop {'enabled' if self.loop_enabled else 'disabled'}")

    def update_gain_label(self):
        """Update gain label and master gain when slider moves"""        
        gain = self.playback_gain.get()
        gain_percent = int(gain * 100)
        self.gain_label.config(text=f"{gain_percent}%")
        
        # Update main gain (takes effect on next loop iteration)
        pysoniq.set_gain(gain)
    
    def open_save_directory(self):
        """Open the save directory in the system file explorer"""
        if self.label_dir is None:
            messagebox.showinfo("No Directory", "No save directory set. Load audio files first.")
            return
        
        import subprocess
        import sys
        import os
        
        try:
            if sys.platform == 'win32':
                os.startfile(str(self.label_dir))
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(self.label_dir)])
            else:  # linux
                subprocess.run(['xdg-open', str(self.label_dir)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open directory: {e}")

    def save_annotations(self):
        """Save peak annotations to JSON"""
        try:
            if not self.audio_files or self.annotation_dir is None:
                return
            
            audio_file = self.audio_files[self.current_file_idx]
            relative_path = audio_file.relative_to(self.base_audio_dir).parent
            filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
            
            if filename_prefix and filename_prefix != '.':
                annotation_file = self.annotation_dir / f"{filename_prefix}_{audio_file.stem}_peak_annotations.json"
            else:
                annotation_file = self.annotation_dir / f"{audio_file.stem}_peak_annotations.json"
            
            # Calculate peak statistics
            if self.peak_annotations:
                freqs = [p['freq'] for p in self.peak_annotations]
                peak_stats = {
                    'num_peaks': len(self.peak_annotations),
                    'mean_freq': float(np.mean(freqs)),
                    'std_freq': float(np.std(freqs)),
                    'min_freq': float(np.min(freqs)),
                    'max_freq': float(np.max(freqs)),
                    'freq_range': float(np.max(freqs) - np.min(freqs))
                }
            else:
                peak_stats = {}
            
            data = {
                'audio_file': str(audio_file),
                'peaks': self.peak_annotations,
                'peak_stats': peak_stats,
                'spec_params': {
                    'n_fft': self.n_fft_spect.get(),
                    'hop_length': self.hop_spect.get(),
                    'fmin_display': self.fmin_display.get(),
                    'fmax_display': self.fmax_display.get()
                },
                'psd_params': {
                    'n_fft': self.n_fft_psd.get(),
                    'hop_length': self.hop_psd.get()
                },
                'detection_params': {
                    'prominence_threshold': self.peak_prominence.get(),
                    'click_threshold_hz': self.peak_click_threshold_hz.get()
                }
            }
            
            with open(annotation_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.changes_made = False
            self.count_total_peaks()  # Recalculate totals
            self.update_display(recompute=False)
            self.update_peak_info()
            print(f"✓ Saved {len(self.peak_annotations)} peaks to {annotation_file.name}")
            
        except Exception as e:
            print(f"ERROR saving annotations: {e}")
            import traceback
            traceback.print_exc()
    





        # ===== NAVIGATION =====

        
        def skip_file(self):
            """Mark current file as skipped with reason and create blank annotation file"""
            if not self.audio_files:
                return
            
            # Create dialog for skip reason
            dialog = tk.Toplevel(self.root)
            dialog.title("Skip File")
            dialog.geometry("300x150")
            
            ttk.Label(dialog, text="Reason for skipping:", font=('', 10, 'bold')).pack(pady=10)
            
            reason_var = tk.StringVar(value="Noisy")
            
            ttk.Radiobutton(dialog, text="Noisy", variable=reason_var, value="Noisy").pack(anchor=tk.W, padx=20)
            ttk.Radiobutton(dialog, text="Truncated", variable=reason_var, value="Truncated").pack(anchor=tk.W, padx=20)
            ttk.Radiobutton(dialog, text="Other", variable=reason_var, value="Other").pack(anchor=tk.W, padx=20)
            
            result = {'confirmed': False, 'reason': None}
            
            def on_ok():
                result['confirmed'] = True
                result['reason'] = reason_var.get()
                dialog.destroy()
            
            def on_cancel():
                dialog.destroy()
            
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)
            ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
            
            # Make dialog modal
            dialog.transient(self.root)
            dialog.grab_set()
            self.root.wait_window(dialog)
            
            # If user cancels, don't skip
            if not result['confirmed']:
                return
            
            reason = result['reason']
            
            audio_file = self.audio_files[self.current_file_idx]
            relative_path = audio_file.relative_to(self.base_audio_dir).parent
            filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
            
            if filename_prefix and filename_prefix != '.':
                label_file = self.label_dir / f"{filename_prefix}_{audio_file.stem}_changepoint_annotations.json"
            else:
                label_file = self.label_dir / f"{audio_file.stem}_changepoint_annotations.json"
            
            # Create blank annotation file marked as skipped with reason
            data = {
                'audio_file': str(audio_file),
                'skipped': True,
                'skip_reason': reason,
                'annotations': [],
                'syllables': [],
                'syllable_metrics': [],
                'spec_params': {
                    'n_fft': self.n_fft.get(),
                    'hop_length': self.hop_length.get(),
                    'fmin_calc': self.fmin_calc.get(),
                    'fmax_calc': self.fmax_calc.get(),
                    'fmin_display': self.fmin_display.get(),
                    'fmax_display': self.fmax_display.get()
                },
                'psd_params': {
                    'n_fft': self.n_fft_psd.get(),
                    'nperseg': self.nperseg_psd.get(),
                    'fmin': self.fmin_psd.get(),
                    'fmax': self.fmax_psd.get()
                }
            }
            
            with open(label_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Skipped file: {audio_file.name} - Reason: {reason}")

            # Recount skipped files
            self.count_skipped_files()
            
            # Move to next file
            self.next_file()


    
    def jump_to_file(self):
        """Jump to a specific file number"""
        try:
            file_num = int(self.file_number_entry.get())
            
            if 1 <= file_num <= len(self.audio_files):
                if self.changes_made:
                    self.save_annotations()
                
                self.current_file_idx = file_num - 1
                self.load_current_file()
            else:
                messagebox.showwarning("Invalid File Number", 
                                    f"Please enter a number between 1 and {len(self.audio_files)}")
                self.update_progress()
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid number")
            self.update_progress()
    
    def previous_file(self):
        """Navigate to previous file"""
        if not self.audio_files:
            return
        
        # Check if currently playing        
        was_looping = pysoniq.is_looping()
        was_playing = was_looping  # If looping, assume it was playing
        
        # Auto-finish current syllable if it has points
        if len(self.current_syllable) >= 2:
            print("Auto-finishing syllable before navigation...")
            self.syllables.append(self.current_syllable[:])
            self.current_syllable = []
            self.rebuild_annotations()
        
        if self.changes_made:
            self.save_annotations()
        
        # Stop current playback
        pysoniq.stop()
        
        self.current_file_idx = (self.current_file_idx - 1) % len(self.audio_files)
        self.load_current_file()
        
        # Resume playback if it was playing
        if was_playing:
            self.play_audio()
    
    def next_file(self):
        """Navigate to next file"""
        if not self.audio_files:
            return
        
        # Check if currently playing        
        was_looping = pysoniq.is_looping()
        was_playing = was_looping  # If looping, assume it was playing
        
        # Auto-finish current syllable if it has points
        if len(self.current_syllable) >= 2:
            print("Auto-finishing syllable before navigation...")
            self.syllables.append(self.current_syllable[:])
            self.current_syllable = []
            self.rebuild_annotations()
        
        if self.changes_made:
            self.save_annotations()
        
        # Stop current playback
        pysoniq.stop()
        
        self.current_file_idx = (self.current_file_idx + 1) % len(self.audio_files)
        self.load_current_file()
        
        # Resume playback if it was playing
        if was_playing:
            self.play_audio()


# ===== MAIN ENTRY POINT =====

def main():
    """Entry point for the peak annotator application"""
    root = tk.Tk()
    app = PeakAnnotator(root)
    root.geometry("1400x800")
    root.mainloop()


if __name__ == "__main__":
    main()