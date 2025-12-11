"""
Base Layer - Core audio visualization and navigation template
Extracted from changepoint_annotator.py (original file) 
Provides: audio loading, spectrogram, plotting, navigation, playback, zoom/pan
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pathlib import Path
from natsort import natsorted

import pysoniq

# try: 
#     from yaaat import utils
# except ImportError: 
#     print("/utils subdir does not exist")
#     import utils

from utils import utils 

class BaseLayer:
    """Base class for audio annotation tools
    
    Provides core functionality:
    - Audio loading and file management
    - Spectrogram computation and display
    - File navigation with auto-save
    - Playback controls with gain/loop
    - Zoom/pan interaction
    - Parameter controls (n_fft, hop, freq range, scale)
    
    Subclass and override:
    - setup_custom_controls() - Add algorithm-specific UI
    - process_audio() - Run custom detection/analysis
    - draw_custom_overlays() - Draw algorithm-specific visualization
    - save_custom_data() - Save algorithm-specific annotations
    - load_custom_data() - Load algorithm-specific annotations
    """
    
    # Initialize global state
    # Global, per-file, per-class point annotations
    global_point_annotations = {}

    def __init__(self, root):
        self.root = root
        if isinstance(root, tk.Tk):
            self.root.title("Base Annotator - YAAAT")
        
        # Audio and spectrogram data
        self.audio_files = []
        self.current_file_idx = 0
        self.y = None
        self.sr = None
        self.S_db = None
        self.freqs = None
        self.times = None
        self.base_audio_dir = None
        
        # STFT/Spectrogram parameters
        self.n_fft = tk.IntVar(value=256)
        self.hop_length = tk.IntVar(value=64)
        self.fmin_calc = tk.IntVar(value=400)
        self.fmax_calc = tk.IntVar(value=16000)
        self.y_scale = tk.StringVar(value='linear')
        
        # Display/plotting limits
        self.fmin_display = tk.IntVar(value=500)
        self.fmax_display = tk.IntVar(value=8000)
        
        # Playback state
        self.playback_gain = tk.DoubleVar(value=1.0)
        self.loop_enabled = False
        
        # Zoom state
        self.zoom_stack = []
        
        # Drag state for zoom
        self.drag_start = None
        self.drag_rect = None
        
        # State
        self.changes_made = False
        self.annotation_dir = None
        
        # Cache the spectrogram image
        self.spec_image = None

        # Waveform display state
        self.show_waveform = tk.BooleanVar(value=False)
        self.waveform_alpha = tk.DoubleVar(value=0.2)
        self.waveform_ax = None  # secondary axis for waveform
        
        # Navigation repeat timer
        self.repeat_id = None
        
        self.setup_ui()
        self.root.after(100, self.auto_load_directory)



    def setup_ui(self):
        """Create the base user interface"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== LEFT CONTROL PANEL =====
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        
        # Scrollable canvas for control panel
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
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)
        
        # Store scrollable_frame for subclass access
        self.control_panel = scrollable_frame
        
        # Header
        ttk.Label(scrollable_frame, text="Audio Annotator", font=('', 10, 'bold')).pack(pady=(0, 2))
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # ===== FILE MANAGEMENT AND PLAYBACK =====
        file_buttons_frame = ttk.Frame(scrollable_frame)
        file_buttons_frame.pack(fill=tk.X, pady=2)
        
        # Left - File Management
        load_frame = ttk.Frame(file_buttons_frame)
        load_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        ttk.Label(load_frame, text="File Management:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Button(load_frame, text="Load Audio Directory", command=self.load_directory).pack(anchor=tk.W, pady=2)
        ttk.Button(load_frame, text="Load Test Audio", command=self.load_test_audio).pack(anchor=tk.W, pady=2)
        
        # Separator
        ttk.Separator(file_buttons_frame, orient=tk.VERTICAL).grid(row=0, column=1, sticky='ns', padx=10)
        
        # Right - Playback
        play_frame = ttk.Frame(file_buttons_frame)
        play_frame.grid(row=0, column=2, sticky='nsew', padx=(5, 0))
        
        ttk.Label(play_frame, text="Playback Controls:", font=('', 9, 'bold')).pack(anchor=tk.CENTER, pady=(0, 2))
        
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
        
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        

        # ===== SPECTROGRAM PARAMETERS =====
        ttk.Label(scrollable_frame, text="Spectrogram:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        # n_fft buttons
        ttk.Label(scrollable_frame, text="n_fft:", font=('', 8)).pack(anchor=tk.W)
        nfft_frame = ttk.Frame(scrollable_frame)
        nfft_frame.pack(fill=tk.X, pady=2)
        
        self.nfft_buttons = []
        for nfft in [128, 256, 512, 1024, 2048, 4096]:
            btn = tk.Button(nfft_frame, text=str(nfft), width=5, command=lambda n=nfft: self.change_nfft(n))
            btn.pack(side=tk.LEFT, padx=2)
            self.nfft_buttons.append((btn, nfft))
        
        # hop_length buttons
        ttk.Label(scrollable_frame, text="hop:", font=('', 8)).pack(anchor=tk.W)
        hop_frame = ttk.Frame(scrollable_frame)
        hop_frame.pack(fill=tk.X, pady=2)
        
        self.hop_buttons = []
        for hop in [16, 32, 64, 128, 256, 512]:
            btn = tk.Button(hop_frame, text=str(hop), width=5, command=lambda h=hop: self.change_hop(h))
            btn.pack(side=tk.LEFT, padx=2)
            self.hop_buttons.append((btn, hop))
        
        # Frequency range
        freq_frame = ttk.Frame(scrollable_frame)
        freq_frame.pack(fill=tk.X, pady=2)
        ttk.Label(freq_frame, text="Freq (Hz):", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.fmin_display, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(freq_frame, text="-", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.fmax_display, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(freq_frame, text="↻", width=2, command=self.update_display_range).pack(side=tk.LEFT, padx=2)
        
        # Scale toggle
        scale_frame = ttk.Frame(scrollable_frame)
        scale_frame.pack(fill=tk.X, pady=2)
        ttk.Label(scale_frame, text="Scale:", font=('', 9)).pack(side=tk.LEFT)
        self.scale_button = tk.Button(scale_frame, text="Linear", width=8, command=self.toggle_scale, bg='lightgreen')
        self.scale_button.pack(side=tk.LEFT, padx=2)
        
        waveform_frame = ttk.LabelFrame(scrollable_frame, text="Waveform", padding=3)
        waveform_frame.pack(fill=tk.X, pady=2)

        # Show waveform checkbox
        ttk.Checkbutton(
            waveform_frame,
            text="Show waveform",
            variable=self.show_waveform,
            command=lambda: self.update_display()
        ).pack(anchor=tk.W)

        # Alpha slider (about half panel width via fixed length)
        wf_alpha_frame = ttk.Frame(waveform_frame)
        wf_alpha_frame.pack(fill=tk.X, pady=1)

        ttk.Label(wf_alpha_frame, text="Transparency:", font=('', 8)).pack(side=tk.LEFT)

        self.waveform_alpha_scale = ttk.Scale(
            wf_alpha_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.waveform_alpha,
            length=150,  # ~half width of typical control panel
            command=self.on_waveform_alpha_change
        )
        self.waveform_alpha_scale.pack(side=tk.LEFT, padx=4)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        



        # ===== CUSTOM CONTROLS (override in subclass) =====
        ttk.Label(scrollable_frame, text="Custom Controls:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.setup_custom_controls()
        
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        



        # ===== ACTIONS =====
        ttk.Label(scrollable_frame, text="Actions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        button_grid = ttk.Frame(scrollable_frame)
        button_grid.pack(pady=2)
        
        buttons = [
            ("Reset Zoom", self.reset_zoom),
            ("Next File", self.next_file),
            ("Prev File", self.previous_file),
        ]
        
        for i, (text, command) in enumerate(buttons):
            ttk.Button(button_grid, text=text, command=command, width=12).grid(
                row=i//3, column=i%3, padx=2, pady=2, sticky='ew')
        



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
        
        ttk.Label(nav_frame, text="[Click + Drag: zoom | Ctrl + Wheel: zoom horizontal | Right-click: reset zoom]", 
                  font=('', 8, 'italic')).pack(side=tk.RIGHT, padx=10)
        
        # Spectrogram canvas
        self.fig = Figure(figsize=(10, 6))
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation buttons below plot
        nav_bottom_frame = ttk.Frame(plot_frame)
        nav_bottom_frame.pack(fill=tk.X, pady=(0, 0))
        
        button_center_frame = ttk.Frame(nav_bottom_frame)
        button_center_frame.pack(anchor=tk.CENTER)
        
        prev_btn = tk.Button(button_center_frame, text="◄ Previous", width=12, font=('', 8))
        prev_btn.pack(side=tk.LEFT, padx=3)
        prev_btn.bind('<ButtonPress-1>', lambda e: self.start_continuous_nav('prev'))
        prev_btn.bind('<ButtonRelease-1>', lambda e: self.stop_continuous_nav())
        
        next_btn = tk.Button(button_center_frame, text="Next ►", width=12, font=('', 8))
        next_btn.pack(side=tk.LEFT, padx=3)
        next_btn.bind('<ButtonPress-1>', lambda e: self.start_continuous_nav('next'))
        next_btn.bind('<ButtonRelease-1>', lambda e: self.stop_continuous_nav())
        
        # Zoom info display
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
        
        self.update_button_highlights()


    def setup_custom_controls(self):
        """Override this method to add algorithm-specific controls"""
        ttk.Label(self.control_panel, text="No custom controls", font=('', 8, 'italic')).pack(pady=2)




    # ===== AUDIO LOADING =====
    
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
            "Where to save annotations?\n\n"
            "Yes = Choose existing directory\n"
            "No = Create new directory\n"
            "Cancel = Use default location"
        )
        
        if response is True:
            save_dir = filedialog.askdirectory(title="Select Annotation Directory")
            if save_dir:
                self.annotation_dir = Path(save_dir)
            else:
                dataset_name = Path(directory).name
                default_dir = Path.home() / "yaaat_annotations" / dataset_name
                default_dir.mkdir(parents=True, exist_ok=True)
                self.annotation_dir = default_dir
        elif response is False:
            save_dir = filedialog.askdirectory(title="Select Parent Directory")
            if save_dir:
                dataset_name = Path(directory).name
                self.annotation_dir = Path(save_dir) / f"{dataset_name}_annotations"
                self.annotation_dir.mkdir(exist_ok=True)
                self.load_global_point_annotations() # Load shared annotations
            else:
                return
        else:
            dataset_name = Path(directory).name
            default_dir = Path.home() / "yaaat_annotations" / dataset_name
            default_dir.mkdir(parents=True, exist_ok=True)
            self.annotation_dir = default_dir
        
        self.annotation_dir.mkdir(exist_ok=True)
        self.load_global_point_annotations() # Load shared annotations
        
        self.current_file_idx = 0
        self.load_current_file()
        
        print(f"✓ Loaded {len(self.audio_files)} files")
        print(f"Annotations will be saved to: {self.annotation_dir}")
        utils.save_last_directory(self.base_audio_dir)
    
    def load_test_audio(self):
        """Load bundled test audio files"""
        test_audio_dir = Path('test_files') / 'test_audio' / 'syllables' / 'kiwi'
        
        if not test_audio_dir.exists():
            messagebox.showinfo("No Test Data", "Test audio files not found")
            return
        
        self.audio_files = natsorted(test_audio_dir.rglob('*.wav'))
        self.base_audio_dir = test_audio_dir
        
        if not self.audio_files:
            messagebox.showwarning("No Files", "No .wav files in test directory")
            return
        
        default_dir = Path.home() / "yaaat_annotations" / "test_audio"
        default_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_dir = default_dir
        self.load_global_point_annotations() # Load shared annotations
        
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
                self.annotation_dir = Path.home() / "yaaat_annotations" / dataset_name
                self.annotation_dir.mkdir(parents=True, exist_ok=True)
                self.load_global_point_annotations()  # Load shared annotations

                self.current_file_idx = 0
                self.load_current_file()
                return
        
        self.load_test_audio()
    
    def load_current_file(self):
        """Load current audio file"""
        if not self.audio_files:
            return
        
        audio_file = self.audio_files[self.current_file_idx]
        print(f"Loading {audio_file.name}...")
        

        # If just switching tabs, skip reload
        if hasattr(self, "_skip_reload") and self._skip_reload:
            # Only recompute spectrogram; do NOT reset annotations or re-run detection
            self.compute_spectrogram()
            self.update_display(recompute_spec=True)
            return

        # Load audio
        self.y, self.sr = pysoniq.load(str(audio_file))
        if self.y.ndim > 1:
            self.y = np.mean(self.y, axis=1)
        
        # Compute spectrogram
        self.compute_spectrogram()
        self.spec_image = None
        
        # Load custom data (override in subclass)
        self.load_custom_data()
        
        # Process audio (override in subclass)
        self.process_audio()
        
        self.changes_made = False
        self.zoom_stack = []
        self.update_display(recompute_spec=True)
        self.update_progress()
    
    def process_audio(self):
        """Override this method to run custom detection/analysis"""
        pass
    
    def load_custom_data(self):
        """Override this method to load algorithm-specific annotations"""
        pass
    









    # Get bucket (global or class-specific) for current file
    def _get_point_bucket(self, scope="class"):
        if not self.audio_files:
            return None

        audio_path = str(self.audio_files[self.current_file_idx])
        file_dict = BaseLayer.global_point_annotations.setdefault(audio_path, {})

        if scope == "global":
            return file_dict.setdefault("global", [])

        class_name = self.__class__.__name__
        return file_dict.setdefault(class_name, [])













    # ===== SPECTROGRAM =====
    
    def compute_spectrogram(self):
        """Compute spectrogram with current parameters"""
        self.S_db, self.freqs, self.times = utils.compute_spectrogram_unified(
            self.y, 
            self.sr,
            nfft=self.n_fft.get(),
            hop=self.hop_length.get(),
            fmin=self.fmin_calc.get(),
            fmax=self.fmax_calc.get(),
            scale=self.y_scale.get(),
            n_mels=256,
            orientation='horizontal'
        )
    
    def change_nfft(self, new_nfft):
        """Change n_fft and recompute"""
        self.n_fft.set(new_nfft)
        self.update_button_highlights()
        if self.y is not None:
            self.recompute_spectrogram()
    
    def change_hop(self, new_hop):
        """Change hop_length and recompute"""
        self.hop_length.set(new_hop)
        self.update_button_highlights()
        if self.y is not None:
            self.recompute_spectrogram()
    
    def recompute_spectrogram(self):
        """Recompute spectrogram with new parameters"""
        if self.y is None:
            return
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        self.compute_spectrogram()
        self.spec_image = None
        self.process_audio()
        self.update_display(recompute_spec=True)
        
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()



    # Annotations for calling with API
    def add_annotation_point(self, time_s, freq_hz, label=None, scope="class"):
        bucket = self._get_point_bucket(scope)
        if bucket is None:
            return

        bucket.append({
            "t": float(time_s),
            "f": float(freq_hz),
            "label": "" if label is None else str(label),
            "scope": scope,
        })

        self.changes_made = True
        self.update_display()




    def update_button_highlights(self):
        """Highlight currently selected parameters"""
        for btn, val in self.nfft_buttons:
            if val == self.n_fft.get():
                btn.config(bg='lightgreen', relief=tk.SUNKEN)
            else:
                btn.config(relief=tk.RAISED, bg='SystemButtonFace')
        
        for btn, val in self.hop_buttons:
            if val == self.hop_length.get():
                btn.config(bg='lightblue', relief=tk.SUNKEN)
            else:
                btn.config(relief=tk.RAISED, bg='SystemButtonFace')
    
    def toggle_scale(self):
        """Toggle between linear and mel scale"""
        if self.y_scale.get() == 'linear':
            self.y_scale.set('mel')
            self.scale_button.config(text='Mel', bg='lightyellow')
        else:
            self.y_scale.set('linear')
            self.scale_button.config(text='Linear', bg='lightgreen')
        
        if self.y is not None:
            self.compute_spectrogram()
            self.spec_image = None
            self.process_audio()
            self.update_display(recompute_spec=True)
    
    def _convert_ylim_to_scale(self, fmin_hz, fmax_hz):
        """Convert Hz limits to current scale"""
        if self.y_scale.get() == 'mel':
            return utils.hz_to_mel(fmin_hz), utils.hz_to_mel(fmax_hz)
        else:
            return fmin_hz, fmax_hz



    def on_waveform_alpha_change(self, value):
        """Update waveform transparency when slider moves."""
        if self.show_waveform.get():
            self.update_display()



    def draw_waveform(self):
        """Draw waveform and smoothed curve on a twin y-axis."""
        if self.y is None or self.sr is None:
            return

        # Create or clear twin axis
        if self.waveform_ax is None:
            self.waveform_ax = self.ax.twinx()
            # Put amplitude axis on the right, slightly offset if needed
            self.waveform_ax.spines['right'].set_position(('outward', 40))
            self.waveform_ax.yaxis.set_label_position('right')
            self.waveform_ax.yaxis.set_ticks_position('right')
        else:
            self.waveform_ax.cla()

        # Time axis
        t = np.arange(len(self.y)) / self.sr

        alpha = float(self.waveform_alpha.get())
        alpha = min(max(alpha, 0.0), 1.0)

        # Raw waveform
        self.waveform_ax.plot(t, self.y, color='cyan', alpha=alpha, linewidth=0.6)

        # Smooth curve (5 ms window)
        window = max(1, int(self.sr * 0.005))
        if window > 1 and len(self.y) >= window:
            smooth = np.convolve(self.y, np.ones(window) / window, mode='same')
            self.waveform_ax.plot(t, smooth, color='yellow', linewidth=1, alpha=min(1.0, alpha + 0.3))

        self.waveform_ax.set_ylabel("Amplitude", fontsize=8)
        self.waveform_ax.tick_params(axis='y', labelsize=7)
        self.waveform_ax.grid(False)




    # ===== DISPLAY =====
    
    def update_display(self, recompute_spec=False):
        """Update the display"""
        try:
            if self.y is None:
                return
            
            if recompute_spec or self.spec_image is None:
                # Full redraw
                self.ax.clear()

                # Remove previous waveform axis on full redraw
                if self.waveform_ax is not None:
                    try:
                        self.waveform_ax.remove()
                    except Exception:
                        pass
                    self.waveform_ax = None

                extent = [
                    self.times[0],
                    self.times[-1],
                    self.freqs[0],
                    self.freqs[-1]
                ]
                
                self.spec_image = self.ax.imshow(
                    self.S_db,
                    aspect='auto',
                    origin='lower',
                    extent=extent,
                    cmap='magma',
                    interpolation='bilinear'
                )
                
                self.ax.set_xlabel('Time (s)', fontsize=8)
                if self.y_scale.get() == 'mel':
                    self.ax.set_ylabel('Frequency (mel)', fontsize=8)
                else:
                    self.ax.set_ylabel('Frequency (Hz)', fontsize=8)
                
                ymin, ymax = self._convert_ylim_to_scale(self.fmin_display.get(), self.fmax_display.get())
                self.ax.set_ylim(ymin, ymax)
            
            else:
                # Remove overlays only
                import matplotlib.collections
                collections_to_remove = [c for c in self.ax.collections 
                                        if isinstance(c, matplotlib.collections.PathCollection)]
                for collection in collections_to_remove:
                    collection.remove()
                
                for patch in self.ax.patches[:]:
                    patch.remove()
                
                for line in self.ax.lines[:]:
                    line.remove()
                
                # Clear previous text on line move
                for text in self.ax.texts[:]:
                    text.remove()

                # Clear waveform axis contents
                if self.waveform_ax is not None:
                    self.waveform_ax.cla()
            
            # Draw custom overlays (override in subclass)
            self.draw_custom_overlays()
            
            # Draw waveform if enabled
            if self.show_waveform.get():
                self.draw_waveform()
            elif self.waveform_ax is not None:
                # If turned off, remove from figure
                try:
                    self.waveform_ax.remove()
                except Exception:
                    pass
                self.waveform_ax = None

            # Update title
            filename = self.audio_files[self.current_file_idx].name
            save_marker = "" if self.changes_made else "✓ "
            self.ax.set_title(f'{save_marker}{filename} | n_fft={self.n_fft.get()} hop={self.hop_length.get()}', 
                            fontsize=9)
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"ERROR in update_display: {e}")
            import traceback
            traceback.print_exc()
    
    def draw_custom_overlays(self):
        """Override this method to draw algorithm-specific overlays"""
        pass


    # Draw global and class-specific points
    def draw_shared_point_annotations(self):
        if not self.audio_files:
            return
        audio_path = str(self.audio_files[self.current_file_idx])
        file_dict = BaseLayer.global_point_annotations.get(audio_path, {})

        # global points
        for ann in file_dict.get("global", []):
            self.ax.scatter(ann["t"], ann["f"], s=30, edgecolors="white", facecolors="none")
            if ann["label"]:
                self.ax.text(ann["t"], ann["f"], ann["label"], fontsize=7, color="white",
                             va="bottom", ha="left")

        # class-specific points
        class_name = self.__class__.__name__
        for ann in file_dict.get(class_name, []):
            self.ax.scatter(ann["t"], ann["f"], s=30, edgecolors="cyan", facecolors="none")
            if ann["label"]:
                self.ax.text(ann["t"], ann["f"], ann["label"], fontsize=7, color="cyan",
                             va="bottom", ha="left")





    def update_display_range(self):
        """Update frequency display range"""
        if self.y is None:
            return
        ymin, ymax = self._convert_ylim_to_scale(self.fmin_display.get(), self.fmax_display.get())
        self.ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()
    
    def reset_zoom(self):
        """Reset zoom to full view"""
        if self.y is None:
            return
        
        self.zoom_stack = []
        full_xlim = (0, len(self.y) / self.sr)
        ymin, ymax = self._convert_ylim_to_scale(self.fmin_display.get(), self.fmax_display.get())
        
        self.ax.set_xlim(full_xlim)
        self.ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()
    
    # ===== MOUSE INTERACTION =====
    
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
        
        # Check for custom press handling (override in subclass)
        if self.on_custom_press(event):
            return
        
        # Left click - start drag for zoom
        if event.button == 1:
            self.drag_start = (event.xdata, event.ydata)
    
    def on_custom_press(self, event):
        """Override to handle custom mouse press events
        
        Returns:
            True if event was handled, False otherwise
        """
        return False
    
    def on_motion(self, event):
        """Handle mouse motion"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        # Check for custom motion handling (override in subclass)
        if self.on_custom_motion(event):
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
        
        x_range = abs(width)
        y_range = abs(height)
        self.zoom_info_label.config(text=f"Time: {x_range:.3f}s | Freq: {y_range:.1f} Hz")
        
        self.canvas.draw_idle()
    
    def on_custom_motion(self, event):
        """Override to handle custom mouse motion events
        
        Returns:
            True if event was handled, False otherwise
        """
        return False
    
    def on_release(self, event):
        """Handle mouse release"""
        try:
            # Check for custom release handling (override in subclass)
            if self.on_custom_release(event):
                self.drag_start = None
                if self.drag_rect is not None:
                    self.drag_rect.remove()
                    self.drag_rect = None
                return
            
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
            
            # If small drag, treat as click
            if drag_dist < 0.05:
                self.drag_start = None
                self.zoom_info_label.config(text="")
                return
            
            # Zoom to selected region
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
            if self.drag_rect is not None:
                try:
                    self.drag_rect.remove()
                except:
                    pass
                self.drag_rect = None
    
    def on_custom_release(self, event):
        """Override to handle custom mouse release events
        
        Returns:
            True if event was handled, False otherwise
        """
        return False
    
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
    



    # ===== PLAYBACK =====
    
    def play_audio(self):
        """Play current audio"""
        if self.y is not None:
            pysoniq.set_gain(self.playback_gain.get())
            pysoniq.play(self.y, self.sr)
    
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
    
    def jump_to_file(self):
        """Jump to specific file number"""
        try:
            file_num = int(self.file_number_entry.get())
            
            if 1 <= file_num <= len(self.audio_files):
                if self.changes_made and not getattr(self, "_syncing_tabs", False):
                    self.save_custom_data()
                
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
        
        if self.changes_made and not getattr(self, "_syncing_tabs", False):
            self.save_custom_data()
            self.save_global_point_annotations()

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
        
        if self.changes_made and not getattr(self, "_syncing_tabs", False):
            self.save_custom_data()
            self.save_global_point_annotations()

        pysoniq.stop()
        
        self.current_file_idx = (self.current_file_idx + 1) % len(self.audio_files)
        self.load_current_file()
        
        if was_playing:
            self.play_audio()
    
    def start_continuous_nav(self, direction):
        """Start continuous navigation when button held"""
        if direction == 'next':
            self.next_file()
        else:
            self.previous_file()
        
        self.repeat_id = self.root.after(300, self.continue_nav, direction)
    
    def continue_nav(self, direction):
        """Continue navigation while button held"""
        if direction == 'next':
            self.next_file()
        else:
            self.previous_file()
        
        self.repeat_id = self.root.after(150, self.continue_nav, direction)
    
    def stop_continuous_nav(self):
        """Stop continuous navigation when button released"""
        if self.repeat_id:
            self.root.after_cancel(self.repeat_id)
            self.repeat_id = None
    
    def update_progress(self):
        """Update file progress"""
        self.file_number_entry.delete(0, tk.END)
        self.file_number_entry.insert(0, str(self.current_file_idx + 1))
        self.file_total_label.config(text=f"/ {len(self.audio_files)}")
        self.file_label.config(text=self.audio_files[self.current_file_idx].name)
    



    # ===== SAVE/LOAD =====
    


    # Save global/shared annotations
    def save_global_point_annotations(self):
        if not self.annotation_dir:
            return
        out = self.annotation_dir / "_global_point_annotations.json"
        with open(out, "w") as f:
            json.dump(BaseLayer.global_point_annotations, f, indent=2)

    # Load global/shared annotations
    def load_global_point_annotations(self):
        if not self.annotation_dir:
            return
        inp = self.annotation_dir / "_global_point_annotations.json"
        if inp.exists():
            with open(inp, "r") as f:
                BaseLayer.global_point_annotations = json.load(f)




    def save_custom_data(self):
        """Override this method to save algorithm-specific annotations"""
        pass





def main():
    """Entry point for base layer"""
    root = tk.Tk()
    app = BaseLayer(root)
    root.geometry("1400x800")
    root.mainloop()


if __name__ == "__main__":
    main()