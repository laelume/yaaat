# from jellyfish.utils.jelly_funcs import make_daily_directory
# daily_dir = make_daily_directory()

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.collections
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import librosa
import librosa.display
from pathlib import Path
import json
from natsort import natsorted
import sounddevice as sd


class ChangepointAnnotator:
    """Interactive tool for annotating time-frequency changepoints on spectrograms"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Spectrogram Changepoint Annotator")
        
        # Audio and spectrogram data
        self.audio_files = []
        self.current_file_idx = 0
        self.y = None
        self.sr = None
        self.S_db = None
        self.freqs = None
        self.times = None
        
        # Syllable tracking
        self.current_syllable = []  # Points in current syllable being built
        self.syllables = []  # List of completed syllables
        
        # Annotation data (auto-generated from syllables)
        self.annotations = []
        
        # STFT/Spectrogram parameters
        self.n_fft = tk.IntVar(value=1024)
        self.hop_length = tk.IntVar(value=256)
        self.fmin_calc = tk.IntVar(value=500)
        self.fmax_calc = tk.IntVar(value=16000)
        self.y_scale = tk.StringVar(value='linear')
        
        # PSD parameters (for future use)
        self.n_fft_psd = tk.IntVar(value=2048)
        self.nperseg_psd = tk.IntVar(value=512)
        self.fmin_psd = tk.IntVar(value=500)
        self.fmax_psd = tk.IntVar(value=16000)
        
        # Display/plotting limits
        self.fmin_display = tk.IntVar(value=500)
        self.fmax_display = tk.IntVar(value=8000)
        
        # Zoom state
        self.zoom_stack = []
        
        # Drag state for zoom
        self.drag_start = None
        self.drag_rect = None
        self.dragging_harmonic = None
        
        # Double-click detection
        self.last_click_time = 0
        self.last_click_pos = None
        
        # State
        self.changes_made = False
        self.label_dir = None
        
        # Cache the spectrogram image
        self.spec_image = None

        # Track total across all files
        self.total_syllables_across_files = 0
        self.total_skipped_files = 0  

        self.harmonic_repeat_ids = {}  # Track repeat timers by harmonic index
        self.harmonics = []  # Will be initialized in setup_ui
        
        self.bounding_box_shape = tk.StringVar(value='rectangle')
        self.show_second_harmonic = tk.BooleanVar(value=False)
        self.harmonic_multiplier = tk.DoubleVar(value=2.0)  # For nudging
        self.show_third_harmonic = tk.BooleanVar(value=False)
        self.third_harmonic_multiplier = tk.DoubleVar(value=3.0)

        self.setup_ui()
    
    def setup_ui(self):
        """Create the user interface"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== LEFT CONTROL PANEL =====
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # # Direct frame - NO scrolling
        # scrollable_frame = ttk.Frame(control_frame, width=456)
        # scrollable_frame.pack(fill=tk.BOTH, expand=True)

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

        # Mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Control Panel
        ttk.Label(scrollable_frame, text="Syllable Annotator", font=('', 10, 'bold')).pack(pady=(0, 2))
        ttk.Label(scrollable_frame, text="Mark onset, offset and changepoints in vocalizations", wraplength=400, font=('', 8, 'italic')).pack(padx=5, pady=(0, 3))

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # Instructions
        ttk.Label(scrollable_frame, text="Instructions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        instructions = ttk.Label(scrollable_frame, text="• Click: add point\n• Click near existing point: remove point\n• Click + Drag: zoom to region\n• Right-click: undo zoom\n• Ctrl + scroll: horizontal zoom", wraplength=400, font=('', 8))
        instructions.pack(padx=5, pady=(0, 5))

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # File management parameters heading
        ttk.Label(scrollable_frame, text="File Management:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        # File loading
        ttk.Button(scrollable_frame, text="Load Audio Directory", command=self.load_directory).pack(anchor=tk.E, pady=2)

        self.file_label = ttk.Label(scrollable_frame, text="No files loaded", wraplength=400, font=('', 8))
        self.file_label.pack(fill=tk.X, pady=2)
        
        # JSON annotation filename (visible location)
        self.filename_label = ttk.Label(scrollable_frame, text="No annotation file", wraplength=400, justify=tk.LEFT, font=('', 8), foreground='blue')
        self.filename_label.pack(fill=tk.X, pady=2)

        # Add button to choose custom annotation directory
        ttk.Button(scrollable_frame, text="Choose Annotation Directory", command=self.load_custom_annotation_dir).pack(anchor=tk.W, pady=2)

        # Save directory (clickable link)
        self.save_dir_button = tk.Button(scrollable_frame, text="No save directory", 
                                         font=('', 8), relief=tk.FLAT, 
                                         fg='blue', cursor='hand2',
                                         command=self.open_save_directory,
                                         bg='SystemButtonFace', anchor='w')
        self.save_dir_button.pack(anchor=tk.W, pady=2)

        # Syllable and annotation information
        self.syllable_info = ttk.Label(scrollable_frame, text="Unsaved Points: 0 | Saved Points: 0 | Saved Syllables: 0", wraplength=600, font=('', 8), justify=tk.LEFT)
        self.syllable_info.pack(pady=2)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # Spectrogram parameters heading
        ttk.Label(scrollable_frame, text="Spectrogram:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        # nfft/hop  parameter buttons
        ttk.Label(scrollable_frame, text="n_fft:", font=('', 8)).pack(anchor=tk.W)
        nfft_frame = ttk.Frame(scrollable_frame)
        nfft_frame.pack(fill=tk.X, expand=True, pady=2)

        self.nfft_buttons = []
        for nfft in [256, 512, 1024, 2048, 4096]:
            btn = tk.Button(nfft_frame, text=str(nfft), width=5,command=lambda n=nfft: self.change_nfft(n))
            btn.pack(side=tk.LEFT, padx=2)
            self.nfft_buttons.append((btn, nfft))

        ttk.Label(scrollable_frame, text="hop:", font=('', 8)).pack(anchor=tk.W)
        hop_frame = ttk.Frame(scrollable_frame)
        hop_frame.pack(fill=tk.X, pady=2)

        self.hop_buttons = []
        for hop in [32, 64, 128, 256, 512]:
            btn = tk.Button(hop_frame, text=str(hop), width=5,command=lambda h=hop: self.change_hop(h))
            btn.pack(side=tk.LEFT, padx=2)
            self.hop_buttons.append((btn, hop))

        # Compact freq range
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
        self.scale_button = tk.Button(scale_frame, text="Linear", width=8,command=self.toggle_scale, bg='lightgreen')
        self.scale_button.pack(side=tk.LEFT, padx=2)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # Actions heading
        ttk.Label(scrollable_frame, text="Actions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        # Action buttons        
        # Create a frame for the button grid
        button_grid = ttk.Frame(scrollable_frame)
        button_grid.pack(pady=2)

        # Define buttons
        buttons = [
            ("Clear Previous", self.clear_last),
            ("Load Audio", self.load_directory), 
            ("Finish Syllable", self.finish_syllable),
            ("Next File", self.next_file),
            
            ("Clear All", self.clear_all),
            ("Play Audio", self.play_audio),
            ("Save Annotations", self.save_annotations),
            ("Previous File", self.previous_file),

            ("Reset Zoom", self.reset_zoom),
            ("Debug Info", self.print_debug_info),
            ("Bounding Box", self.toggle_bounding_box), 
            ("Skip File", self.skip_file) 

        ]

        # Arrange in 3x3 grid
        for i, (text, command) in enumerate(buttons):
            row = i // 4  # Integer division for row
            col = i % 4   # Modulo for column
            ttk.Button(button_grid, text=text, command=command, width=12).grid(
                row=row, column=col, padx=2, pady=2, sticky='ew'
            )

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # Statistics
        ttk.Label(scrollable_frame, text="Statistics:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.stats_label = ttk.Label(scrollable_frame, text="WHAT WHY", justify=tk.LEFT, font=('', 8))
        self.stats_label.pack(fill=tk.X, pady=2)

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

        # 4-column grid layout - ALL must use row= and column=, NOT side=
        ttk.Checkbutton(guides_grid, text="Time Lines", variable=self.show_time_guides, command=self.toggle_guides).grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Freq Lines", variable=self.show_freq_guides, command=self.toggle_guides).grid(row=0, column=1, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Show All", variable=self.show_all_guides_var, command=self.toggle_show_all).grid(row=0, column=2, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Hide Text", variable=self.hide_text, command=self.toggle_guides).grid(row=0, column=3, sticky=tk.W, padx=2, pady=2)

        ttk.Checkbutton(guides_grid, text="Bounding Box", variable=self.show_bounding_box, command=self.toggle_guides).grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        
        ttk.Radiobutton(guides_grid, text="Rectangle", variable=self.bounding_box_shape, value='rectangle', command=self.toggle_guides).grid(row=1, column=1, sticky=tk.W, padx=2, pady=2)
        ttk.Radiobutton(guides_grid, text="Polygon", variable=self.bounding_box_shape, value='polygon', command=self.toggle_guides).grid(row=1, column=2, sticky=tk.W, padx=2, pady=2)
        ttk.Radiobutton(guides_grid, text="Ellipse", variable=self.bounding_box_shape, value='ellipse', command=self.toggle_guides).grid(row=1, column=3, sticky=tk.W, padx=2, pady=2)
        
        ttk.Checkbutton(guides_grid, text="Bound 2nd Harmonic", variable=self.show_second_harmonic, command=self.toggle_guides).grid(row=2, column=1, sticky=tk.W, padx=2, pady=2)

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

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # # ===== SPECTROGRAM PARAMETERS =====
        # ttk.Label(scrollable_frame, text="Spectrogram Parameters:", font=('', 10, 'bold')).pack(pady=(0, 5))

        # # n_fft buttons
        # ttk.Label(scrollable_frame, text="n_fft:", font=('', 9)).pack(anchor=tk.W, padx=5)
        # nfft_frame = ttk.Frame(scrollable_frame)
        # nfft_frame.pack(fill=tk.X, pady=2)

        # self.nfft_buttons = []
        # nfft_values = [256, 512, 1024, 2048, 4096]
        # for nfft in nfft_values:
        #     btn = tk.Button(nfft_frame, text=str(nfft), width=6,command=lambda n=nfft: self.change_nfft(n))
        #     btn.pack(side=tk.LEFT, padx=2)
        #     self.nfft_buttons.append((btn, nfft))

        # # hop_length buttons
        # ttk.Label(scrollable_frame, text="hop_length:", font=('', 9)).pack(anchor=tk.W, padx=5, pady=(5, 0))
        # hop_frame = ttk.Frame(scrollable_frame)
        # hop_frame.pack(fill=tk.X, pady=2)

        # self.hop_buttons = []
        # hop_values = [64, 128, 256, 512]
        # for hop in hop_values:
        #     btn = tk.Button(hop_frame, text=str(hop), width=6, command=lambda h=hop: self.change_hop(h))
        #     btn.pack(side=tk.LEFT, padx=2)
        #     self.hop_buttons.append((btn, hop))

        # # Display frequency limits
        # ttk.Label(scrollable_frame, text="Display Freq Range (Hz):", font=('', 9)).pack(anchor=tk.W, padx=5, pady=(5, 0))
        # disp_freq_frame = ttk.Frame(scrollable_frame)
        # disp_freq_frame.pack(fill=tk.X, pady=2)

        # ttk.Label(disp_freq_frame, text="fmin:").pack(side=tk.LEFT, padx=2)
        # ttk.Entry(disp_freq_frame, textvariable=self.fmin_display, width=8).pack(side=tk.LEFT, padx=2)
        # ttk.Label(disp_freq_frame, text="fmax:").pack(side=tk.LEFT, padx=2)
        # ttk.Entry(disp_freq_frame, textvariable=self.fmax_display, width=8).pack(side=tk.LEFT, padx=2)

        # ttk.Button(scrollable_frame, text="Update Display Range", command=self.update_display_range).pack(fill=tk.X, pady=5)

        # ttk.Button(scrollable_frame, text="Recompute Spectrogram", command=self.recompute_spectrogram).pack(fill=tk.X, pady=2)

        # # Toggle mel / linear scale on y-axis        
        # scale_frame = ttk.Frame(scrollable_frame)
        # scale_frame.pack(fill=tk.X, pady=5)
        # ttk.Label(scale_frame, text="Y-axis scale:").pack(side=tk.LEFT, padx=5)
        # self.scale_button = tk.Button(scale_frame, text="Linear", width=10,command=self.toggle_scale, bg='lightgreen')
        # self.scale_button.pack(side=tk.LEFT, padx=5)

        # ttk.Button(scrollable_frame, text="Reset Zoom", command=self.reset_zoom).pack(fill=tk.X, pady=2)

        # ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # ===== PSD PARAMETERS (for future use) =====
        ttk.Label(scrollable_frame, text="Multiresolution PSD Parameters:", font=('', 10, 'bold')).pack(pady=(0, 2))
        ttk.Label(scrollable_frame, text="(future jellyfish analysis)", font=('', 8, 'italic')).pack()
        
        psd_param_frame = ttk.Frame(scrollable_frame)
        psd_param_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(psd_param_frame, text="n_fft:").grid(row=0, column=0, sticky=tk.W, padx=2)
        ttk.Entry(psd_param_frame, textvariable=self.n_fft_psd, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(psd_param_frame, text="nperseg:").grid(row=1, column=0, sticky=tk.W, padx=2)
        ttk.Entry(psd_param_frame, textvariable=self.nperseg_psd, width=8).grid(row=1, column=1, padx=2)
        
        # ===== RIGHT SPECTROGRAM PANEL =====
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Navigation buttons
        nav_frame = ttk.Frame(plot_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(nav_frame, text="◄ Previous", command=self.previous_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next ►", command=self.next_file).pack(side=tk.LEFT, padx=5)
        
        # self.progress_label = ttk.Label(nav_frame, text="")
        # self.progress_label.pack(side=tk.LEFT, padx=20)
        file_nav_frame = ttk.Frame(nav_frame)
        file_nav_frame.pack(side=tk.LEFT, padx=20)

        ttk.Label(file_nav_frame, text="File:", font=('', 9)).pack(side=tk.LEFT, padx=2)
        self.file_number_entry = ttk.Entry(file_nav_frame, width=6, justify=tk.CENTER)
        self.file_number_entry.pack(side=tk.LEFT, padx=2)
        self.file_total_label = ttk.Label(file_nav_frame, text="/ 0", font=('', 9))
        self.file_total_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(file_nav_frame, text="Go", command=self.jump_to_file, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_nav_frame, text="Find Skipped", command=self.find_next_skipped, width=12).pack(side=tk.LEFT, padx=5)

        # Bind Enter key to jump
        self.file_number_entry.bind('<Return>', lambda e: self.jump_to_file())

        ttk.Label(nav_frame, text="[Click: annotate point | Click + Drag: zoom region | Ctrl + Wheel: zoom horizontal(vertical) | Right-click: reset zoom]", 
                  font=('', 8, 'italic')).pack(side=tk.RIGHT, padx=10)
        
        # Spectrogram canvas
        self.fig = Figure(figsize=(10, 6))
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)  # Reduce whitespace
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Additional navigation buttons below plot
        nav_bottom_frame = ttk.Frame(plot_frame)
        nav_bottom_frame.pack(fill=tk.X, pady=(0, 0))

        # Create inner frame for centering
        button_center_frame = ttk.Frame(nav_bottom_frame)
        button_center_frame.pack(anchor=tk.CENTER)

        prev_btn = tk.Button(button_center_frame, text="◄ Previous", width=12, font=('', 8))
        prev_btn.pack(side=tk.LEFT, padx=3)
        prev_btn.bind('<ButtonPress-1>', lambda e: self.start_continuous_nav('prev'))
        prev_btn.bind('<ButtonRelease-1>', lambda e: self.stop_continuous_nav())

        tk.Button(button_center_frame, text="Finish Syllable", command=self.finish_syllable, width=12, font=('', 8)).pack(side=tk.LEFT, padx=3)

        next_btn = tk.Button(button_center_frame, text="Next ►", width=12, font=('', 8))
        next_btn.pack(side=tk.LEFT, padx=3)
        next_btn.bind('<ButtonPress-1>', lambda e: self.start_continuous_nav('next'))
        next_btn.bind('<ButtonRelease-1>', lambda e: self.stop_continuous_nav())
        # Add zoom info display
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
        
        # Update button highlights
        self.update_button_highlights()

    def find_next_skipped(self):
        """Jump to next file that is skipped or has no annotations"""
        if not self.audio_files:
            messagebox.showinfo("No Audio", "Load audio files first")
            return
        
        # Start searching from next file
        start_idx = (self.current_file_idx + 1) % len(self.audio_files)
        
        for i in range(len(self.audio_files)):
            check_idx = (start_idx + i) % len(self.audio_files)
            audio_file = self.audio_files[check_idx]
            
            # Build annotation filename
            relative_path = audio_file.relative_to(self.base_audio_dir).parent
            filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
            
            if filename_prefix and filename_prefix != '.':
                label_file = self.label_dir / f"{filename_prefix}_{audio_file.stem}_changepoint_annotations.json"
            else:
                label_file = self.label_dir / f"{audio_file.stem}_changepoint_annotations.json"
            
            # Check if file doesn't exist OR is marked as skipped
            is_skipped = False
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        data = json.load(f)
                        is_skipped = data.get('skipped', False)
                        has_syllables = len(data.get('syllables', [])) == 0
                        
                        # Consider it skipped if marked OR has no syllables
                        if is_skipped or has_syllables:
                            is_skipped = True
                except:
                    pass
            else:
                # No annotation file = skipped
                is_skipped = True
            
            if is_skipped:
                # Found a skipped file - jump to it
                if self.changes_made:
                    self.save_annotations()
                
                self.current_file_idx = check_idx
                self.load_current_file()
                print(f"✓ Found skipped file: {audio_file.name}")
                return
        
        # No skipped files found
        messagebox.showinfo("All Annotated", 
            "No skipped or unannotated files found!\nAll files have annotations.")

    def print_debug_info(self):
        """Print comprehensive debug information"""
        print("\n" + "="*50)
        print("DEBUG INFO")
        print("="*50)
        print(f"Audio loaded: {self.y is not None}")
        if self.y is not None:
            print(f"Audio length: {len(self.y) / self.sr:.2f}s")
        print(f"Current file: {self.current_file_idx + 1}/{len(self.audio_files)}")
        print(f"Syllables: {len(self.syllables)}")
        print(f"Current syllable points: {len(self.current_syllable)}")
        print(f"Total annotations: {len(self.annotations)}")
        print(f"Zoom stack depth: {len(self.zoom_stack)}")
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        print(f"Current xlim: {xlim}")
        print(f"Current ylim: {ylim}")
        print(f"Y-scale: {self.y_scale.get()}")
        print(f"Display range: {self.fmin_display.get()} - {self.fmax_display.get()} Hz")
        print(f"Spec image cached: {self.spec_image is not None}")
        print("="*50 + "\n")
            
    def change_nfft(self, new_nfft):
        """Change n_fft and recompute"""
        self.n_fft.set(new_nfft)
        self.update_button_highlights()
        self.reset_double_click_tracking()  # Reset since display is changing
        if self.y is not None:
            self.recompute_spectrogram()
    
    def change_hop(self, new_hop):
        """Change hop_length and recompute"""
        self.hop_length.set(new_hop)
        self.update_button_highlights()
        self.reset_double_click_tracking()  # Reset since display is changing
        if self.y is not None:
            self.recompute_spectrogram()
    
    def _convert_ylim_to_scale(self, fmin_hz, fmax_hz):
        """Convert Hz limits to current scale (mel or linear)"""
        if self.y_scale.get() == 'mel':
            return librosa.hz_to_mel(fmin_hz), librosa.hz_to_mel(fmax_hz)
        else:
            return fmin_hz, fmax_hz

    def reset_double_click_tracking(self):
        """Reset double-click tracking state"""
        self.last_click_time = 0
        self.last_click_pos = None
    
    def update_button_highlights(self):
        """Highlight currently selected parameters"""
        # Highlight n_fft
        for btn, val in self.nfft_buttons:
            if val == self.n_fft.get():
                btn.config(bg='lightgreen', relief=tk.SUNKEN)
            else:
                btn.config(bg='SystemButtonFace', relief=tk.RAISED)
        
        # Highlight hop_length
        for btn, val in self.hop_buttons:
            if val == self.hop_length.get():
                btn.config(bg='lightblue', relief=tk.SUNKEN)
            else:
                btn.config(bg='SystemButtonFace', relief=tk.RAISED)
    
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
                self.reset_double_click_tracking()
            return
        
        # Left click - check if near harmonic box edge first
        if event.button == 1:
            # Check if clicking near any harmonic box edge
            if self.show_bounding_box.get() and self.annotations:
                times = [ann['time'] for ann in self.annotations]
                freqs = [ann['freq'] for ann in self.annotations]
                t_min, t_max = min(times), max(times)
                f_min, f_max = min(freqs), max(freqs)
                
                # Find the CLOSEST harmonic edge across all harmonics
                closest_edge = None
                closest_dist = float('inf')
                
                for i, harmonic in enumerate(self.harmonics):
                    if harmonic['show'].get():
                        multiplier = harmonic['multiplier'].get()
                        h_f_min = f_min * multiplier
                        h_f_max = f_max * multiplier
                        
                        # Check if click is within time bounds
                        if t_min <= event.xdata <= t_max:
                            # Distance to top edge
                            dist_top = abs(event.ydata - h_f_max)
                            if dist_top < closest_dist and dist_top < 100:  # 100 Hz threshold
                                closest_dist = dist_top
                                closest_edge = (i, 'top', f_max)
                            
                            # Distance to bottom edge
                            dist_bottom = abs(event.ydata - h_f_min)
                            if dist_bottom < closest_dist and dist_bottom < 100:
                                closest_dist = dist_bottom
                                closest_edge = (i, 'bottom', f_min)
                
                # If found a close edge, grab it
                if closest_edge is not None:
                    self.dragging_harmonic = closest_edge
                    harmonic_idx = closest_edge[0]
                    edge_type = closest_edge[1]
                    print(f"Grabbed {self.harmonics[harmonic_idx]['name']} harmonic {edge_type} edge (distance: {closest_dist:.1f} Hz)")
                    return
            
            # If not dragging harmonic, normal behavior (zoom or annotate)
            self.drag_start = (event.xdata, event.ydata)
    
    def on_motion(self, event):
        """Handle mouse motion - draw zoom rectangle or drag harmonic edge"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        # If dragging harmonic edge
        if self.dragging_harmonic is not None:
            harmonic_idx, edge, base_freq = self.dragging_harmonic
            harmonic = self.harmonics[harmonic_idx]
            
            # Calculate new multiplier from mouse position
            new_multiplier = event.ydata / base_freq
            
            # Clamp to reasonable range
            new_multiplier = max(0.1, min(10.0, new_multiplier))
            
            # Update multiplier and label
            harmonic['multiplier'].set(new_multiplier)
            harmonic['label'].config(text=f"{new_multiplier:.2f}x")
            
            # Redraw
            self.update_display(recompute_spec=False)
            return
        
        # Normal zoom rectangle drawing
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
        self.zoom_info_label.config(text=f"Time: {x_range:.3f}s | Freq: {y_range:.1f} Hz")

        self.canvas.draw_idle()
    
    def on_release(self, event):
        """Handle mouse button release"""
        try:
            # If was dragging harmonic, finalize
            if self.dragging_harmonic is not None:
                self.dragging_harmonic = None
                print("Released harmonic edge")
                return
        
            if self.drag_start is None:
                return
            
            if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
                self.drag_start = None
                # Clear zoom info display
                self.zoom_info_label.config(text="")
                if self.drag_rect is not None:
                    self.drag_rect.remove()
                    self.drag_rect = None
                    self.canvas.draw_idle()
                return
            
            x0, y0 = self.drag_start
            x1, y1 = event.xdata, event.ydata
            
            print(f"Release: x0={x0:.3f}, y0={y0:.1f}, x1={x1:.3f}, y1={y1:.1f}, dblclick={event.dblclick}")
            
            # Calculate drag distance
            drag_dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            print(f"Drag distance: {drag_dist:.4f}")
            
            # Remove drag rectangle
            if self.drag_rect is not None:
                self.drag_rect.remove()
                self.drag_rect = None
            
            # If drag distance is small, treat as click
            if drag_dist < 0.05:  # Threshold for click vs drag
                # Validate coordinates are in valid range
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                
                if xlim[0] <= x0 <= xlim[1] and ylim[0] <= y0 <= ylim[1]:
                    # FIRST: Check if clicking near existing point (if so, remove it)
                    if self.remove_nearby_annotation(x0, y0):
                        print("Removed nearby point")
                    else:
                        # Otherwise, add new point to current syllable
                        self.current_syllable.append({
                            'time': float(x0),
                            'freq': float(y0)
                        })
                        
                        self.changes_made = True
                        self.rebuild_annotations()
                        self.update_display(recompute_spec=False)
                        print(f"+ Point {len(self.current_syllable)}: t={x0:.3f}s, f={y0:.0f}Hz")
                else:
                    print(f"! Ignoring click outside valid range: x={x0}, y={y0}")
                    self.drag_start = None
                    return
            
            # Otherwise, zoom to selected region
            else:
                # Set new limits
                new_xlim = sorted([x0, x1])
                new_ylim = sorted([y0, y1])
                
                # Prevent extreme zooms (less than 0.01s or 10Hz range)
                x_range = new_xlim[1] - new_xlim[0]
                y_range = new_ylim[1] - new_ylim[0]
                
                if x_range < 0.01 or y_range < 10:
                    print(f"! Zoom too small, ignoring: x_range={x_range}, y_range={y_range}")
                    self.drag_start = None
                    return
                
                # Save current view to zoom stack
                current_xlim = self.ax.get_xlim()
                current_ylim = self.ax.get_ylim()
                self.zoom_stack.append((current_xlim, current_ylim))
                print(f"Saved zoom state: xlim={current_xlim}, ylim={current_ylim}")
                print(f"New zoom: xlim={new_xlim}, ylim={new_ylim}")
                
                self.ax.set_xlim(new_xlim)
                self.ax.set_ylim(new_ylim)
                self.canvas.draw_idle()
                self.reset_double_click_tracking()  # Reset after zoom
            
            self.drag_start = None
            
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
        """Handle mouse wheel combo actions"""
        try:
            if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
                return

            import sys
            if sys.platform == 'win32':
                import ctypes
                is_ctrl = bool(ctypes.windll.user32.GetKeyState(0x11) & 0x8000)
                is_shift = bool(ctypes.windll.user32.GetKeyState(0x10) & 0x8000)
                is_ctrlshift = is_ctrl and is_shift
            else:
                key = getattr(event, 'key', None)
                is_ctrl = (key == 'control')
                is_shift = (key == 'shift')
                is_ctrlshift = False

            print(f"Scroll: button={event.button}, ctrl={is_ctrl}, shift={is_shift}")

            # Save current limits for zoom stack
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            print(f"Current limits: xlim={xlim}, ylim={ylim}")

            # No modifiers: pan vertically
            if not is_ctrl and not is_shift:
                y_range = ylim[1] - ylim[0]
                pan_amount = y_range * 0.1
                if event.button == 'up':
                    new_ylim = (ylim[0] + pan_amount, ylim[1] + pan_amount)
                else:
                    new_ylim = (ylim[0] - pan_amount, ylim[1] - pan_amount)
                self.ax.set_ylim(new_ylim)
                self.canvas.draw_idle()
                return

            # Determine zoom factor based on scroll direction
            zoom_factor = 0.8 if event.button == 'up' else 1.25

            # Apply zoom based on key states
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

            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"ERROR in on_scroll: {e}")
            import traceback
            traceback.print_exc()
    
    def reset_zoom(self):
        """Reset zoom to full view"""
        try:
            if self.y is None:
                print("Cannot reset zoom: no audio loaded")
                return
            
            self.zoom_stack = []
            full_xlim = (0, len(self.y) / self.sr)
            ymin, ymax = self._convert_ylim_to_scale(self.fmin_display.get(), self.fmax_display.get())
            
            print(f"Reset zoom to: xlim={full_xlim}, ylim=({ymin}, {ymax})")
            
            self.ax.set_xlim(full_xlim)
            self.ax.set_ylim(ymin, ymax)
            self.canvas.draw_idle()
            self.reset_double_click_tracking()  # Reset after view change
            
        except Exception as e:
            print(f"ERROR in reset_zoom: {e}")
            import traceback
            traceback.print_exc()
    
    def update_display_range(self):
        """Update the frequency display range without recomputing"""
        if self.y is None:
            return
        ymin, ymax = self._convert_ylim_to_scale(self.fmin_display.get(), self.fmax_display.get())
        self.ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()

    def toggle_scale(self):
        """Toggle between linear and mel frequency scale"""
        if self.y_scale.get() == 'linear':
            self.y_scale.set('mel')
            self.scale_button.config(text='Mel', bg='lightyellow')
        else:
            self.y_scale.set('linear')
            self.scale_button.config(text='Linear', bg='lightgreen')
        
        self.reset_double_click_tracking()  # Reset since coordinate system changes
        if self.y is not None:
            self.compute_spectrogram()  # Recompute with new scale
            self.spec_image = None  # Clear cache
            self.update_display(recompute_spec=True)

    def finish_syllable(self):
        """Mark current syllable as complete and start new one"""

        # Calculate total points across all syllables
        saved_points = sum(len(syllable) for syllable in self.syllables)
        total_points = len(self.current_syllable) + saved_points
        # Ensure that each syllable has at least 2 points (onset and offset)
        if total_points < 2:
            messagebox.showwarning("Need at least 2 total points across all syllables")
            return
        if len(self.current_syllable) < 1:
            messagebox.showwarning("Need at least 1 more point to add annotation")
            return
        
        self.syllables.append(self.current_syllable[:])
        self.current_syllable = []
        self.changes_made = True
        self.rebuild_annotations()
        self.update_display(recompute_spec=False)
        print(f"✓ Syllable complete ({len(self.syllables)} total)")
        print(f"  Total annotations now: {len(self.annotations)}")

    def rebuild_annotations(self):
        """Rebuild annotations list from syllables with auto-labeling"""
        try:
            print(f"Rebuilding: {len(self.syllables)} syllables, {len(self.current_syllable)} current points")

            # Combine all syllables and sort by time
            all_points = []
            for syllable in self.syllables:
                all_points.extend(syllable)
            
            # Sort points by time
            all_points.sort(key=lambda x: x['time'])
                
            self.annotations = []    
            # First = onset, last = offset, middle = changepoint
            for i, point in enumerate(all_points):
                if i == 0:
                    label = 'onset'
                elif i == len(all_points) - 1:  # Change this line
                    label = 'offset'
                else:
                    label = 'changepoint'
                
                self.annotations.append({
                    'time': point['time'],
                    'freq': point['freq'],
                    'label': label
                })
            
            # Add current syllable (in progress) - all marked as changepoint for now
            for point in self.current_syllable:
                self.annotations.append({
                    'time': point['time'],
                    'freq': point['freq'],
                    'label': 'changepoint'  # Temporary until syllable finished
                })
            
            # Update syllable info label
            unsaved_points = len(self.current_syllable)
            saved_syllables = len(self.syllables)
            saved_points = sum(len(syllable) for syllable in self.syllables)
            
            self.syllable_info.config(
                    text=f"Unsaved Points: {unsaved_points} | Saved Points: {saved_points} | This File: {saved_syllables} syll | Total: {self.total_syllables_across_files} syll")
            
            print(f"Rebuilt {len(self.annotations)} total annotations")
            
        except Exception as e:
            print(f"ERROR in rebuild_annotations: {e}")
            import traceback
            traceback.print_exc()

    def remove_nearby_annotation(self, x, y):
        """Remove annotation near the click location"""
        if x is None or y is None:
            return False
        
        # Thresholds in interpretable units
        time_threshold_ms = 10  # milliseconds
        time_threshold_s = time_threshold_ms / 1000.0  # convert to seconds
        freq_threshold_hz = 10  # Hz
        
        # Find closest annotation within threshold
        min_dist = float('inf')
        closest_idx = None
        closest_syllable_idx = None
        
        # Check current syllable
        for i, point in enumerate(self.current_syllable):
            time_diff = abs(point['time'] - x)
            freq_diff = abs(point['freq'] - y)
            
            # Check if within thresholds
            if time_diff < time_threshold_s and freq_diff < freq_threshold_hz:
                # Calculate combined distance for finding closest
                dist = np.sqrt((time_diff/time_threshold_s)**2 + (freq_diff/freq_threshold_hz)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    closest_syllable_idx = -1  # Current syllable
        
        # Check completed syllables
        for syll_idx, syllable in enumerate(self.syllables):
            for point_idx, point in enumerate(syllable):
                time_diff = abs(point['time'] - x)
                freq_diff = abs(point['freq'] - y)
                
                # Check if within thresholds
                if time_diff < time_threshold_s and freq_diff < freq_threshold_hz:
                    # Calculate combined distance for finding closest
                    dist = np.sqrt((time_diff/time_threshold_s)**2 + (freq_diff/freq_threshold_hz)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = point_idx
                        closest_syllable_idx = syll_idx
        
        # Remove the closest point
        if closest_idx is not None:
            if closest_syllable_idx == -1:
                # Remove from current syllable
                removed = self.current_syllable.pop(closest_idx)
                print(f"- Removed point from current syllable: t={removed['time']:.3f}s")
            else:
                # Remove from completed syllable
                removed = self.syllables[closest_syllable_idx].pop(closest_idx)
                print(f"- Removed point from syllable {closest_syllable_idx}: t={removed['time']:.3f}s")
                # If syllable now has < 2 points, remove entire syllable
                if len(self.syllables[closest_syllable_idx]) < 2:
                    self.syllables.pop(closest_syllable_idx)
                    print(f"  Removed entire syllable {closest_syllable_idx} (< 2 points)")
            
            self.changes_made = True
            self.rebuild_annotations()
            self.update_display(recompute_spec=False)
            return True
        
        return False

    def count_total_syllables(self):
        """Count total syllables across all annotation files"""
        self.total_syllables_across_files = 0
        for audio_file in self.audio_files:
            relative_path = audio_file.relative_to(self.base_audio_dir).parent
            filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
            if filename_prefix and filename_prefix != '.':
                label_file = self.label_dir / f"{filename_prefix}_{audio_file.stem}_changepoint_annotations.json"
            else:
                label_file = self.label_dir / f"{audio_file.stem}_changepoint_annotations.json"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    data = json.load(f)
                    file_syllables = len(data.get('syllables', []))
                    self.total_syllables_across_files += file_syllables
                    print(f"File: {label_file.name}, Syllables: {file_syllables}")

        print(f"Total syllables across all files: {self.total_syllables_across_files}")

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
        
        # Store base directory for relative paths
        self.base_audio_dir = Path(directory)

        if not self.audio_files:
            messagebox.showwarning("No Files", "No .wav files found")
            return

        # NOTE: THIS NEEDS TO BE RE-IMPLEMENTED DURING THE FULL LAUNCH OF JELLYFISH DYNAMITE OR WHATEVER IT EVENTUALLY BECOMES        
        # # Set up label directory
        # from jellyfish.utils.jelly_funcs import make_daily_directory
        # daily_dir = make_daily_directory()
        # dataset_name = Path(directory).name  # Get the folder name
        # self.label_dir = daily_dir / dataset_name

        # Ask user where to save annotations
        from tkinter import messagebox
        response = messagebox.askyesnocancel(
            "Annotation Save Location",
            "Where do you want to save annotations?\n\n"
            "Yes = Choose existing directory\n"
            "No = Create new directory\n"
            "Cancel = Use default location"
        )

        if response is True:  # Yes - choose existing
            save_dir = filedialog.askdirectory(title="Select Annotation Save Directory")
            if save_dir:
                self.label_dir = Path(save_dir)
            else:
                return  # User cancelled
        elif response is False:  # No - create new
            save_dir = filedialog.askdirectory(title="Select Parent Directory for New Folder")
            if save_dir:
                dataset_name = Path(directory).name
                self.label_dir = Path(save_dir) / f"{dataset_name}_annotations"
                self.label_dir.mkdir(exist_ok=True)
            else:
                return
        else:  # Cancel - use default
            dataset_name = Path(directory).name
            default_dir = Path.home() / "yaaat_annotations" / dataset_name
            default_dir.mkdir(parents=True, exist_ok=True)
            self.label_dir = default_dir

        self.label_dir.mkdir(exist_ok=True)
        print(f"Annotations will be saved to: {self.label_dir}")
        
        # Update save directory button
        self.save_dir_button.config(text=f"📁 {self.label_dir}")
        
        self.current_file_idx = 0
        self.count_total_syllables()
        self.count_skipped_files()
        self.load_current_file()
        
        print(f"✓ Loaded {len(self.audio_files)} files")
    
    def load_current_file(self):
        """Load the current audio file and its annotations"""
        if not self.audio_files:
            return
        
        self.reset_double_click_tracking()  # Reset for new file
        
        audio_file = self.audio_files[self.current_file_idx]
        print(f"Loading {audio_file.name}...")
        
        # Load audio
        self.y, self.sr = librosa.load(str(audio_file), sr=None)
        
        # Compute spectrogram
        self.compute_spectrogram()
        self.spec_image = None  # Clear cached image to force redraw
        self.file_was_annotated = False  # Track if file had previous annotations

        # Clear all annotation states before loading new file
        print(f"DEBUG: Before clearing - annotations: {len(self.annotations)}, syllables: {len(self.syllables)}")
        self.annotations = []
        self.current_syllable = []
        self.syllables = []
        print(f"DEBUG: After clearing - annotations: {len(self.annotations)}, syllables: {len(self.syllables)}")

        # Load existing annotations
        relative_path = audio_file.relative_to(self.base_audio_dir).parent
        filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
        if filename_prefix and filename_prefix != '.':
            label_file = self.label_dir / f"{filename_prefix}_{audio_file.stem}_changepoint_annotations.json"
        else:
            label_file = self.label_dir / f"{audio_file.stem}_changepoint_annotations.json"

        if label_file.exists():
            with open(label_file, 'r') as f:
                data = json.load(f)
                
                # Debug: print what was loaded
                print(f"JSON contents: syllables={len(data.get('syllables', []))}, annotations={len(data.get('annotations', []))}")
                
                # Load syllables first
                self.syllables = data.get('syllables', [])
                self.current_syllable = []
                
                # Validate syllable structure
                if self.syllables:
                    print(f"First syllable structure: {self.syllables[0]}")
                    # Check if syllables have the right format
                    valid_syllables = []
                    for syll in self.syllables:
                        if isinstance(syll, list) and len(syll) > 0:
                            if all('time' in p and 'freq' in p for p in syll):
                                valid_syllables.append(syll)
                            else:
                                print(f"Invalid syllable format, skipping: {syll}")
                        else:
                            print(f"Invalid syllable (not a list or empty), skipping: {syll}")
                    
                    self.syllables = valid_syllables
                
                # rebuild annotations from syllables
                if self.syllables:
                    self.rebuild_annotations()
                    self.file_was_annotated = True  # Mark file as previously annotated

                    print(f"✓ Loaded {len(self.syllables)} syllables, {len(self.annotations)} annotations")
                    print(f"DEBUG: syllable_info text after rebuild: {self.syllable_info.cget('text')}")


                else:
                    print("⚠ No valid syllables found in file")
                    self.annotations = []
        else:
            print(f"No annotation file found at {label_file}")
            self.annotations = []
            self.current_syllable = []
            self.syllables = []
        
        self.changes_made = False
        self.zoom_stack = []
        self.update_display(recompute_spec=True)  # force full recompute
        self.update_progress()
        
        # Update filename label (label_file previously defined above)
        print(f"DEBUG: Setting filename label to: {label_file.name}")

        # Check if file is marked as skipped
        is_skipped = False
        skip_reason = ""
        if label_file.exists():
            with open(label_file, 'r') as f:
                data = json.load(f)
                is_skipped = data.get('skipped', False)
                skip_reason = data.get('skip_reason', '')

        if is_skipped:
            # Count which skipped file this is
            skipped_count = 0
            for i in range(self.current_file_idx + 1):
                audio_f = self.audio_files[i]
                rel_path = audio_f.relative_to(self.base_audio_dir).parent
                fname_prefix = str(rel_path).replace('/', '_').replace('\\', '_')
                if fname_prefix and fname_prefix != '.':
                    lbl_file = self.label_dir / f"{fname_prefix}_{audio_f.stem}_changepoint_annotations.json"
                else:
                    lbl_file = self.label_dir / f"{audio_f.stem}_changepoint_annotations.json"
                
                if lbl_file.exists():
                    try:
                        with open(lbl_file, 'r') as f:
                            d = json.load(f)
                            if d.get('skipped', False):
                                skipped_count += 1
                    except:
                        pass
            
            display_text = f"⊘ SKIPPED ({skipped_count}/{self.total_skipped_files}): {label_file.name}"
            if skip_reason:
                display_text += f"\nReason: {skip_reason}"
            self.filename_label.config(text=display_text, foreground='orange')
        elif self.file_was_annotated:
            self.filename_label.config(text=f"✓ {label_file}", foreground='green')
        else:
            self.filename_label.config(text=f"→ {label_file}", foreground='blue')

        print(f"DEBUG: Filename label text is now: {self.filename_label.cget('text')}")


    def load_custom_annotation_dir(self):
        """Copy all annotations from custom directory to daily directory"""
        if not self.audio_files:
            messagebox.showinfo("No Audio", "Load audio files first")
            return
        
        # Select directory
        custom_dir = filedialog.askdirectory(
            title="Select Annotation Directory",
            initialdir=str(self.label_dir.parent) if self.label_dir else None
        )
        
        if not custom_dir:
            return
        
        custom_dir = Path(custom_dir)
        
        # Find all JSON annotation files in custom directory
        json_files = list(custom_dir.glob("*_changepoint_annotations.json"))
        
        if not json_files:
            messagebox.showwarning("No Annotations", 
                f"No annotation files found in:\n{custom_dir}")
            return
        
        # Copy all JSON files to daily directory
        import shutil
        copied_count = 0
        for json_file in json_files:
            dest_file = self.label_dir / json_file.name
            shutil.copy2(json_file, dest_file)
            copied_count += 1
            print(f"Copied: {json_file.name}")
        
        # Reload current file to show copied annotations
        self.load_current_file()
        
        messagebox.showinfo("Annotations Copied", 
            f"Copied {copied_count} annotation files\n\n"
            f"From: {custom_dir}\n"
            f"To: {self.label_dir}")
        
        print(f"✓ Copied {copied_count} files from {custom_dir} to {self.label_dir}")

    def compute_spectrogram(self):
        """Compute spectrogram with current parameters"""
        if self.y_scale.get() == 'mel':
            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(
                y=self.y, 
                sr=self.sr,
                n_fft=self.n_fft.get(),
                hop_length=self.hop_length.get(),
                fmin=self.fmin_calc.get(),
                fmax=self.fmax_calc.get(),
                n_mels=256  # Frequency resolution: number of mel bins, higher = more bands = finer resolution
            )
            self.S_db = librosa.power_to_db(S, ref=np.max)
            self.freqs = librosa.mel_frequencies(n_mels=256, 
                                                fmin=self.fmin_calc.get(), 
                                                fmax=self.fmax_calc.get())
        else:
            # Compute linear STFT spectrogram
            S = np.abs(librosa.stft(self.y, n_fft=self.n_fft.get(), 
                                    hop_length=self.hop_length.get()))
            
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft.get())
            freq_mask = (freqs >= self.fmin_calc.get()) & (freqs <= self.fmax_calc.get())
            
            self.S_db = librosa.amplitude_to_db(S[freq_mask, :], ref=np.max)
            self.freqs = freqs[freq_mask]
        
        self.times = librosa.frames_to_time(
            np.arange(self.S_db.shape[1]), 
            sr=self.sr, 
            hop_length=self.hop_length.get()
        )
    
    def recompute_spectrogram(self):
        """Recompute spectrogram with new parameters"""
        if self.y is None:
            return
        
        # Save current view limits to preserve zoom
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        self.compute_spectrogram()
        self.spec_image = None
        self.update_display(recompute_spec=True)
        
        # Restore zoom view
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()
    
    def toggle_guides(self):
        if self.y is not None:
            self.update_display(recompute_spec=True)

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
        self.harmonic_repeat_ids[harmonic_index] = self.root.after(
            200, self.continue_harmonic, harmonic_index, direction)

    def continue_harmonic(self, harmonic_index, direction):
        """Continue harmonic nudging"""
        self.nudge_harmonic(harmonic_index, direction)
        self.harmonic_repeat_ids[harmonic_index] = self.root.after(
            50, self.continue_harmonic, harmonic_index, direction)

    def stop_continuous_harmonic(self, harmonic_index):
        """Stop continuous harmonic nudging"""
        if harmonic_index in self.harmonic_repeat_ids:
            self.root.after_cancel(self.harmonic_repeat_ids[harmonic_index])
            del self.harmonic_repeat_ids[harmonic_index]

    def update_display(self, recompute_spec=False):
        """Update the spectrogram display with annotations
        Args:
            recompute_spec: If True, redraw the entire spectrogram. If False, only update annotations.
        """
        try:
            print(f"update_display called: recompute_spec={recompute_spec}")
            
            if recompute_spec or self.spec_image is None:
                # Full redraw - expensive operation
                self.ax.clear()

                y_axis_param = 'mel' if self.y_scale.get() == 'mel' else 'hz'
                
                # Plot spectrogram
                self.spec_image = librosa.display.specshow(
                    self.S_db,
                    x_axis='time',
                    y_axis=y_axis_param,
                    sr=self.sr,
                    hop_length=self.hop_length.get(),
                    fmin=self.fmin_calc.get(),
                    fmax=self.fmax_calc.get(),
                    cmap='magma',
                    ax=self.ax
                )
                
                # Set display limits
                ymin, ymax = self._convert_ylim_to_scale(self.fmin_display.get(), self.fmax_display.get())
                self.ax.set_ylim(ymin, ymax)
            else:
                # Quick update - remove scatter plots and patches
                collections_to_remove = [c for c in self.ax.collections 
                                        if isinstance(c, matplotlib.collections.PathCollection)]
                for collection in collections_to_remove:
                    collection.remove()
                
                # Remove patches (bounding boxes) so they can be redrawn
                for patch in self.ax.patches[:]:
                    patch.remove()

            # Update title to reflect current save state
            filename = self.audio_files[self.current_file_idx].name
            if self.changes_made:
                saved = ""
            elif self.file_was_annotated:
                saved = "✓ "  # Green check shown via title
            else:
                saved = "⊙ "  # Circle for new file
            self.ax.set_title(f'{saved}{filename} | {len(self.syllables)} syllables | '
                            f'n_fft={self.n_fft.get()} hop={self.hop_length.get()}')
            
            # plot all annotations with correct colors
            colors = {'onset': 'green', 'offset': 'magenta', 'changepoint': 'cyan'}
            markers = {'onset': '.', 'offset': '.', 'changepoint': '.'}
            
            print(f"DEBUG: Plotting {len(self.annotations)} annotations")
            if self.annotations:
                print(f"DEBUG: First annotation: {self.annotations[0]}")
                print(f"DEBUG: Last annotation: {self.annotations[-1]}")
            
            for ann in self.annotations:
                self.ax.scatter(ann['time'], ann['freq'], 
                            c=colors[ann['label']], 
                            marker=markers[ann['label']],
                            s=100, 
                            linewidths=1,
                            zorder=10)

            # Add time and frequency guide lines
            if self.annotations and (self.show_time_guides.get() or self.show_freq_guides.get()):
                times = [ann['time'] for ann in self.annotations]
                freqs = [ann['freq'] for ann in self.annotations]
                
                if self.show_time_guides.get():
                    for t in times:
                        self.ax.axvline(x=t, color='lime', linestyle='--', linewidth=1.5, alpha=0.5)
                
                if self.show_freq_guides.get():
                    for f in freqs:
                        self.ax.axhline(y=f, color='yellow', linestyle='--', linewidth=1.5, alpha=0.5)
                
                # Add text annotations
                for ann in self.annotations:
                    # Time annotation on vertical line (lime) - only if time guides enabled
                    if self.show_time_guides.get() and not self.hide_text.get():
                        self.ax.text(ann['time'], self.ax.get_ylim()[1] * 0.95, 
                                    f"{ann['time']:.3f}s", 
                                    color='lime', fontsize=9, fontweight='bold', 
                                    family='monospace', alpha=0.9, rotation=90,
                                    verticalalignment='top', horizontalalignment='right')
                    
                    # Frequency annotation on horizontal line (yellow) - only if freq guides enabled
                    if self.show_freq_guides.get() and not self.hide_text.get():
                        self.ax.text(self.ax.get_xlim()[0] + 0.01, ann['freq'], 
                                    f"{ann['freq']:.1f}Hz", 
                                    color='yellow', fontsize=9, fontweight='bold',
                                    family='monospace', alpha=0.9,
                                    verticalalignment='center', horizontalalignment='left')
                
                # Calculate and display total duration and frequency delta
                if len(self.syllables) > 0:
                    first_syllable = sorted(self.syllables[0], key=lambda x: x['time'])
                    duration = first_syllable[-1]['time'] - first_syllable[0]['time']
                    freq_delta = max(p['freq'] for p in first_syllable) - min(p['freq'] for p in first_syllable)
                    
                    self.ax.text(0.02, 0.95, 
                                f"Duration: {duration:.3f}s\nΔFreq: {freq_delta:.1f}Hz", 
                                transform=self.ax.transAxes, 
                                fontsize=10, 
                                verticalalignment='top', 
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # Draw bounding box if enabled
            if self.show_bounding_box.get() and self.annotations:
                times = [ann['time'] for ann in self.annotations]
                freqs = [ann['freq'] for ann in self.annotations]
                t_min, t_max = min(times), max(times)
                f_min, f_max = min(freqs), max(freqs)
                
                shape_type = self.bounding_box_shape.get()
                
                if shape_type == 'rectangle':
                    shape = plt.Rectangle(
                        (t_min, f_min), t_max - t_min, f_max - f_min,
                        fill=False, edgecolor='white', linewidth=2.5, alpha=0.8, zorder=11
                    )
                elif shape_type == 'ellipse':
                    from matplotlib.patches import Ellipse
                    center_t = (t_min + t_max) / 2
                    center_f = (f_min + f_max) / 2
                    width = t_max - t_min
                    height = f_max - f_min
                    shape = Ellipse(
                        (center_t, center_f), width, height,
                        fill=False, edgecolor='white', linewidth=2.5, alpha=0.8, zorder=11
                    )
                elif shape_type == 'polygon':
                    from matplotlib.patches import Polygon
                    points = [(ann['time'], ann['freq']) for ann in self.annotations]
                    shape = Polygon(
                        points, closed=True,
                        fill=False, edgecolor='white', linewidth=2.5, alpha=0.8, zorder=11
                    )
                
                self.ax.add_patch(shape)

            # Draw all enabled harmonics
            for harmonic in self.harmonics:
                if harmonic['show'].get() and self.show_bounding_box.get() and self.annotations:
                    multiplier = harmonic['multiplier'].get()
                    harmonic_f_min = f_min * multiplier
                    harmonic_f_max = f_max * multiplier
                    
                    if shape_type == 'rectangle':
                        harmonic_shape = plt.Rectangle(
                            (t_min, harmonic_f_min), t_max - t_min, harmonic_f_max - harmonic_f_min,
                            fill=False, edgecolor=harmonic['color'], linewidth=2, linestyle='-', alpha=0.6, zorder=11
                        )
                    elif shape_type == 'ellipse':
                        from matplotlib.patches import Ellipse
                        center_t = (t_min + t_max) / 2
                        harmonic_center_f = (harmonic_f_min + harmonic_f_max) / 2
                        width = t_max - t_min
                        harmonic_height = harmonic_f_max - harmonic_f_min
                        harmonic_shape = Ellipse(
                            (center_t, harmonic_center_f), width, harmonic_height,
                            fill=False, edgecolor=harmonic['color'], linewidth=2, linestyle='-', alpha=0.6, zorder=11
                        )
                    elif shape_type == 'polygon':
                        from matplotlib.patches import Polygon
                        harmonic_points = [(ann['time'], ann['freq'] * multiplier) for ann in self.annotations]
                        harmonic_shape = Polygon(
                            harmonic_points, closed=True,
                            fill=False, edgecolor=harmonic['color'], linewidth=2, linestyle='-', alpha=0.6, zorder=11
                        )
                    
                    self.ax.add_patch(harmonic_shape)
                    
            self.canvas.draw()  # Force immediate draw, not idle
            self.update_stats()
            
        except Exception as e:
            print(f"ERROR in update_display: {e}")
            import traceback
            traceback.print_exc()

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

    def update_stats(self):
        """Update annotation statistics"""
        if not self.annotations:
            self.stats_label.config(text="No annotations")
            return

        # Find onset and offset
        onset_point = next((a for a in self.annotations if a['label'] == 'onset'), None)
        offset_point = next((a for a in self.annotations if a['label'] == 'offset'), None)
        
        if onset_point and offset_point:
            onset_time = onset_point['time']
            offset_time = offset_point['time']
            duration = offset_time - onset_time
            
            # Calculate frequency range
            freqs = [a['freq'] for a in self.annotations]
            delta_freq = max(freqs) - min(freqs)
            mean_freq = sum(freqs) / len(freqs)
            delta_onset_offset = abs(offset_point['freq'] - onset_point['freq'])
            max_freq = max(freqs)
            min_freq = min(freqs)
            delta_onset_maxfreq = abs(max_freq - onset_point['freq'])
            delta_maxfreq_offset = abs(max_freq - offset_point['freq'])

            
            self.stats_label.config(
                        text=f"Onset: {onset_time:.3f}s | Offset: {offset_time:.3f}s | Duration: {duration:.3f}s\n"
                            f"Mean freq: {mean_freq:.1f} Hz | Max freq: {max_freq:.1f} Hz| Min freq: {min_freq:.1f} Hz | ΔFreq: {delta_freq:.1f} Hz\n"
                            f"ΔOnset-Max: {delta_onset_maxfreq:.1f} Hz | ΔMax-Offset: {delta_maxfreq_offset:.1f} Hz | ΔOnset-Offset: {delta_onset_offset:.1f} Hz")

        else:
            self.stats_label.config(text="Incomplete annotation")
    
    def update_progress(self):
        """Update file progress indicator"""
        self.file_number_entry.delete(0, tk.END)
        self.file_number_entry.insert(0, str(self.current_file_idx + 1))
        self.file_total_label.config(text=f"/ {len(self.audio_files)}")
        self.file_label.config(text=self.audio_files[self.current_file_idx].name)

    def jump_to_file(self):
        """Jump to a specific file number"""
        try:
            file_num = int(self.file_number_entry.get())
            
            if 1 <= file_num <= len(self.audio_files):
                # Save current file if changes made
                if self.changes_made:
                    self.save_annotations()
                
                # Jump to requested file (convert from 1-indexed to 0-indexed)
                self.current_file_idx = file_num - 1
                self.load_current_file()
            else:
                messagebox.showwarning("Invalid File Number", 
                                    f"Please enter a number between 1 and {len(self.audio_files)}")
                self.update_progress()  # Reset to current file
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid number")
            self.update_progress()  # Reset to current file

    def clear_last(self):
        """Remove the last point from current syllable"""
        if self.current_syllable:
            self.current_syllable.pop()
            self.changes_made = True
            self.rebuild_annotations()
            self.update_display(recompute_spec=False)
        elif self.syllables:
            # If no current syllable, undo last completed syllable
            self.current_syllable = self.syllables.pop()
            self.changes_made = True
            self.rebuild_annotations()
            self.update_display(recompute_spec=False)
    
    def clear_all(self):
        """Clear all annotations"""
        if (self.annotations or self.current_syllable) and messagebox.askyesno("Clear", "Remove all?"):
            self.annotations = []
            self.current_syllable = []
            self.syllables = []
            self.changes_made = True
            self.update_display(recompute_spec=False)
    
    def play_audio(self):
        """Play the current audio file"""
        if self.y is not None:
            sd.play(self.y, self.sr)
    
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
        """Save annotations to JSON"""
        try:
            if not self.audio_files:
                print("ERROR: No audio files loaded")
                return
            
            if self.label_dir is None:
                print("ERROR: label_dir is None")
                return
            
            print(f"Label dir: {self.label_dir}")
            print(f"Label dir exists: {self.label_dir.exists()}")
            
            audio_file = self.audio_files[self.current_file_idx]
            relative_path = audio_file.relative_to(self.base_audio_dir).parent
            filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
            if filename_prefix and filename_prefix != '.':
                label_file = self.label_dir / f"{filename_prefix}_{audio_file.stem}_changepoint_annotations.json"
            else:
                label_file = self.label_dir / f"{audio_file.stem}_changepoint_annotations.json"
            
            print(f"Attempting to save to: {label_file}")
            print(f"Number of annotations: {len(self.annotations)}")
            print(f"Number of syllables: {len(self.syllables)}")
            
            # Calculate syllable metrics
            syllable_metrics = []
            for i, syllable in enumerate(self.syllables):
                # Sort by time
                syllable_sorted = sorted(syllable, key=lambda x: x['time'])
                
                # Get onset and offset (first and last in time)
                onset_time = syllable_sorted[0]['time']
                offset_time = syllable_sorted[-1]['time']
                
                # Calculate duration
                duration = offset_time - onset_time
                
                # Get frequency range
                all_freqs = [point['freq'] for point in syllable]
                freq_min = min(all_freqs)
                freq_max = max(all_freqs)
                frequency_spread = freq_max - freq_min
                
                syllable_metrics.append({
                    'syllable_index': i,
                    'onset_time': onset_time,
                    'offset_time': offset_time,
                    'syllable_duration': duration,
                    'frequency_min': freq_min,
                    'frequency_max': freq_max,
                    'frequency_spread': frequency_spread,
                    'num_points': len(syllable)
                })
            
            data = {
                'audio_file': str(audio_file),
                'annotations': self.annotations,
                'syllables': self.syllables,  # Save syllable structure
                'syllable_metrics': syllable_metrics,  # Add metrics
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
                self.count_total_syllables()  # Recalculate after save
                self.rebuild_annotations()  # Update display with new total

            
            self.changes_made = False
            self.update_display(recompute_spec=False)
            print(f"✓ Saved successfully to {label_file}")
            
            # Print syllable metrics for verification
            for metric in syllable_metrics:
                print(f"  Syllable {metric['syllable_index']}: "
                    f"dur={metric['syllable_duration']:.3f}s, "
                    f"freq_spread={metric['frequency_spread']:.1f}Hz")
            
        except Exception as e:
            print(f"ERROR saving annotations: {e}")
            import traceback
            traceback.print_exc()

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

    def previous_file(self):
        """Navigate to previous file"""
        if not self.audio_files:
            return
        
        # Auto-finish current syllable if it has points
        if len(self.current_syllable) >= 2:
            print("Auto-finishing syllable before navigation...")
            self.syllables.append(self.current_syllable[:])
            self.current_syllable = []
            self.rebuild_annotations()
        
        if self.changes_made:
            self.save_annotations()
        
        self.current_file_idx = (self.current_file_idx - 1) % len(self.audio_files)
        self.load_current_file()
    
    def next_file(self):
        """Navigate to next file"""
        if not self.audio_files:
            return
        
        # Auto-finish current syllable if it has points
        if len(self.current_syllable) >= 2:
            print("Auto-finishing syllable before navigation...")
            self.syllables.append(self.current_syllable[:])
            self.current_syllable = []
            self.rebuild_annotations()
        
        if self.changes_made:
            self.save_annotations()
        
        self.current_file_idx = (self.current_file_idx + 1) % len(self.audio_files)
        self.load_current_file()

    def start_continuous_nav(self, direction):
        """Start continuous navigation when button held down"""
        # First navigation happens immediately
        if direction == 'next':
            self.next_file()
        else:
            self.previous_file()
        
        # Schedule repeated navigation (300ms initial delay, then 150ms repeats)
        self.repeat_id = self.root.after(300, self.continue_nav, direction)

    def continue_nav(self, direction):
        """Continue navigation while button held"""
        if direction == 'next':
            self.next_file()
        else:
            self.previous_file()
        
        # Schedule next repeat (faster now)
        self.repeat_id = self.root.after(150, self.continue_nav, direction)

    def stop_continuous_nav(self):
        """Stop continuous navigation when button released"""
        if self.repeat_id:
            self.root.after_cancel(self.repeat_id)
            self.repeat_id = None
    
    def toggle_bounding_box(self):
        """Toggle bounding box checkbox"""
        self.show_bounding_box.set(not self.show_bounding_box.get())
        self.toggle_guides()

# Launch the app
# root = tk.Tk()
# app = ChangepointAnnotator(root)
# root.geometry("1000x800")
# root.mainloop()

def main():
    """Entry point for the application"""
    root = tk.Tk()
    app = ChangepointAnnotator(root)
    root.geometry("1000x800")
    root.mainloop()

if __name__ == "__main__":
    main()
