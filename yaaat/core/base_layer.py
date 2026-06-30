"""
core/base_layer.py

Root parent class for all YAAAT annotator tabs.
Owns all shared spectrogram state and the primary UI composition point.
Delegates I/O, navigation, visualization, and interaction to core modules.

Subclass hooks (override in tab subclasses):
    - setup_custom_controls()  : add tab-specific controls to the control panel
    - process_audio()          : run tab-specific detection or analysis
    - draw_custom_overlays()   : draw tab-specific matplotlib overlays
    - save_custom_data()       : save tab-specific annotation data
    - load_custom_data()       : load tab-specific annotation data
    - on_custom_press()        : handle tab-specific mouse press
    - on_custom_motion()       : handle tab-specific mouse motion
    - on_custom_release()      : handle tab-specific mouse release
"""

import json
import logging
import traceback

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

from yaaat.core import audio_utils
from yaaat.core import file_nav
from yaaat.core import annotation_io
from yaaat.core import visualization
from yaaat.core import interaction

# (つ -' _ '- )つ    (つ -' _ '- )つ
# CONFIG import — all tunable defaults live in config.py.
# BaseLayer reads from CONFIG so changing config.py affects all tabs.
# (つ -' _ '- )つ    (つ -' _ '- )つ
from yaaat.config import CONFIG

logger = logging.getLogger(__name__)


# (つ -' _ '- )つ    (つ -' _ '- )つ

class BaseLayer:
    """Root parent class for all YAAAT annotator tabs.

    Owns spectrogram state, UI composition, playback, and spectrogram
    computation. All other concerns delegate to core modules.
    """

    ##    <(''<)  <( ' ' )>  (>'')>

    # Class-level shared annotation store — keyed by audio path, then class name or 'global'
    # Persisted to disk via annotation_io.save_global_point_annotations()
    global_point_annotations = {}

    def __init__(self, root):
        """Initialize BaseLayer state and build the UI."""
        self.root = root

        # Set window title only if root is a top-level Tk window
        if isinstance(root, tk.Tk):
            self.root.title("Base Annotator - YAAAT")

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # AUDIO STATE
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        # Ordered file list populated by file_nav.load_directory()
        self.audio_files = []
        self.current_file_idx = 0

        # Raw audio signal (mono float32) and sample rate
        self.y = None
        self.sr = None

        # Spectrogram output from audio_utils.compute_spectrogram_unified()
        self.S_db = None
        self.freqs = None
        self.times = None

        # Root directory of the loaded audio dataset
        self.base_audio_dir = None

        ##    <(''<)  <( ' ' )>  (>'')>
        # SPECTROGRAM PARAMETERS
        # All defaults sourced from CONFIG — change config.py to affect
        # all tabs without touching BaseLayer source.
        ##    <(''<)  <( ' ' )>  (>'')>

        self.n_fft      = tk.IntVar(value=CONFIG["n_fft"])
        self.hop_length = tk.IntVar(value=CONFIG["hop_length"])
        self.fmin_calc  = tk.IntVar(value=CONFIG["fmin_calc"])
        self.fmax_calc  = tk.IntVar(value=CONFIG["fmax_calc"])

        # 'linear' or 'mel' — controls compute_spectrogram_unified() scale param
        self.y_scale    = tk.StringVar(value=CONFIG["y_scale"])

        # Display frequency limits (may differ from computation limits)
        self.fmin_display = tk.IntVar(value=CONFIG["fmin_display"])
        self.fmax_display = tk.IntVar(value=CONFIG["fmax_display"])


        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # PLAYBACK STATE
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.playback_gain = tk.DoubleVar(value=CONFIG["playback_gain"])
        self.loop_enabled = False

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # INTERACTION STATE
        # Owned here; read by interaction.py functions via explicit args
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        # Stack of (xlim, ylim) tuples for zoom undo; right-click undo only unwinds scroll-zooms, not drag-zooms, after bbox update
        self.zoom_stack = []

        # Drag origin for bounding box
        self.drag_start = None

        # Active matplotlib Rectangle patch for bounding box preview
        self.drag_rect = None

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # ANNOTATION AND PERSISTENCE STATE
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        # Dirty flag — True when unsaved changes exist
        self.changes_made = False

        # Directory where annotation JSON files are written
        self.annotation_dir = None

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # DISPLAY STATE
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        # Cached matplotlib AxesImage — None forces full spectrogram redraw
        self.spec_image = None

        # Waveform overlay controls
        self.show_waveform = tk.BooleanVar(value=False)
        self.waveform_alpha = tk.DoubleVar(value=CONFIG["waveform_alpha"])

        # Twin y-axis for waveform overlay — None when not active
        self.waveform_ax = None

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # NAVIGATION REPEAT TIMER
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        # after() id for continuous prev/next navigation while button held
        self.repeat_id = None

        # Tab-sync flag — suppresses auto-save during programmatic file switches
        self._syncing_tabs = False

        # Skip-reload flag — suppresses full audio reload on tab switch
        self._skip_reload = False

        self.setup_ui()

        # Delay auto-load slightly to allow UI to fully render before disk access
        self.root.after(222, self._auto_load_on_startup)

    ##    <(''<)  <( ' ' )>  (>'')>
    # UI COMPOSITION
    ##    <(''<)  <( ' ' )>  (>'')>

    def setup_ui(self):
        """Build the primary two-panel UI: scrollable control panel (left) and spectrogram panel (right)."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # LEFT CONTROL PANEL — scrollable canvas wrapping a frame
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

        ctrl_canvas = tk.Canvas(control_frame)
        ctrl_scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=ctrl_canvas.yview)
        scrollable_frame = ttk.Frame(ctrl_canvas)

        ctrl_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ctrl_canvas.configure(yscrollcommand=ctrl_scrollbar.set)
        ctrl_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))
        )

        # Bind mousewheel only while cursor is inside the control panel
        ctrl_canvas.bind("<Enter>", lambda e: ctrl_canvas.bind_all(
            "<MouseWheel>", lambda ev: ctrl_canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")))
        ctrl_canvas.bind("<Leave>", lambda e: ctrl_canvas.unbind_all("<MouseWheel>"))

        # Expose scrollable frame for subclass control injection via setup_custom_controls()
        self.control_panel = scrollable_frame

        # Header
        ttk.Label(scrollable_frame, text="Audio Annotator", font=('', 10, 'bold')).pack(pady=(0, 2))
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # FILE MANAGEMENT AND PLAYBACK ROW
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        file_buttons_frame = ttk.Frame(scrollable_frame)
        file_buttons_frame.pack(fill=tk.X, pady=2)

        # Left column — file management buttons
        load_frame = ttk.Frame(file_buttons_frame)
        load_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        ttk.Label(load_frame, text="File Management:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Button(load_frame, text="Load Audio Directory", command=self.load_directory).pack(anchor=tk.W, pady=2)
        ttk.Button(load_frame, text="Load Test Audio", command=self.load_test_audio).pack(anchor=tk.W, pady=2)

        ttk.Separator(file_buttons_frame, orient=tk.VERTICAL).grid(row=0, column=1, sticky='ns', padx=10)

        # Right column — playback controls
        play_frame = ttk.Frame(file_buttons_frame)
        play_frame.grid(row=0, column=2, sticky='nsew', padx=(5, 0))

        ttk.Label(play_frame, text="Playback Controls:", font=('', 9, 'bold')).pack(anchor=tk.CENTER, pady=(0, 2))

        controls_container = ttk.Frame(play_frame)
        controls_container.pack(anchor=tk.CENTER)

        buttons_row = ttk.Frame(controls_container)
        buttons_row.pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(buttons_row, text="▶", command=self.play_audio,
                  bg='lightgreen', font=('', 12, 'bold'), width=2, height=1).pack(side=tk.LEFT, padx=2)

        self.pause_button = tk.Button(buttons_row, text="⏸", command=self.pause_audio,
                                      bg='yellow', font=('', 12, 'bold'), width=2, height=1)
        self.pause_button.pack(side=tk.LEFT, padx=2)

        tk.Button(buttons_row, text="⏹", command=self.stop_audio,
                  bg='lightcoral', font=('', 12, 'bold'), width=2, height=1).pack(side=tk.LEFT, padx=2)

        self.loop_button = tk.Button(buttons_row, text="⟳", command=self.toggle_loop,
                                     bg='lightblue', font=('', 12, 'bold'), width=2, height=1)
        self.loop_button.pack(side=tk.LEFT, padx=2)

        # Vertical gain fader
        gain_frame = ttk.Frame(controls_container)
        gain_frame.pack(side=tk.LEFT)

        ttk.Label(gain_frame, text="Gain", font=('', 7)).pack()
        ttk.Scale(gain_frame, from_=2.0, to=0.0, variable=self.playback_gain,
                  orient=tk.VERTICAL, length=60,
                  command=lambda v: self.update_gain_label()).pack()
        self.gain_label = ttk.Label(gain_frame, text="100%", font=('', 7))
        self.gain_label.pack()

        file_buttons_frame.columnconfigure(0, weight=1)
        file_buttons_frame.columnconfigure(2, weight=1)

        # Current filename display
        self.file_label = ttk.Label(scrollable_frame, text="No files loaded", wraplength=400, font=('', 8))
        self.file_label.pack(fill=tk.X, pady=2)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # SPECTROGRAM PARAMETERS
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(scrollable_frame, text="Spectrogram:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        # n_fft preset buttons
        ttk.Label(scrollable_frame, text="n_fft:", font=('', 8)).pack(anchor=tk.W)
        nfft_frame = ttk.Frame(scrollable_frame)
        nfft_frame.pack(fill=tk.X, pady=2)

        self.nfft_buttons = []
        for nfft in [128, 256, 512, 1024, 2048, 4096]:
            btn = tk.Button(nfft_frame, text=str(nfft), width=5,
                            command=lambda n=nfft: self.change_nfft(n))
            btn.pack(side=tk.LEFT, padx=2)
            self.nfft_buttons.append((btn, nfft))

        # hop_length preset buttons
        ttk.Label(scrollable_frame, text="hop:", font=('', 8)).pack(anchor=tk.W)
        hop_frame = ttk.Frame(scrollable_frame)
        hop_frame.pack(fill=tk.X, pady=2)

        self.hop_buttons = []
        for hop in [16, 32, 64, 128, 256, 512]:
            btn = tk.Button(hop_frame, text=str(hop), width=5,
                            command=lambda h=hop: self.change_hop(h))
            btn.pack(side=tk.LEFT, padx=2)
            self.hop_buttons.append((btn, hop))

        # Frequency display range — independent of computation range
        freq_frame = ttk.Frame(scrollable_frame)
        freq_frame.pack(fill=tk.X, pady=2)
        ttk.Label(freq_frame, text="Freq (Hz):", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.fmin_display, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(freq_frame, text="-", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.fmax_display, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(freq_frame, text="↻", width=2, command=self.update_display_range).pack(side=tk.LEFT, padx=2)

        # Linear / mel scale toggle
        scale_frame = ttk.Frame(scrollable_frame)
        scale_frame.pack(fill=tk.X, pady=2)
        ttk.Label(scale_frame, text="Scale:", font=('', 9)).pack(side=tk.LEFT)
        self.scale_button = tk.Button(scale_frame, text="Linear", width=8,
                                      command=self.toggle_scale, bg='lightgreen')
        self.scale_button.pack(side=tk.LEFT, padx=2)

        # Waveform overlay controls
        waveform_frame = ttk.LabelFrame(scrollable_frame, text="Waveform", padding=3)
        waveform_frame.pack(fill=tk.X, pady=2)

        ttk.Checkbutton(waveform_frame, text="Show waveform",
                        variable=self.show_waveform,
                        command=lambda: self.update_display()).pack(anchor=tk.W)

        wf_alpha_frame = ttk.Frame(waveform_frame)
        wf_alpha_frame.pack(fill=tk.X, pady=1)
        ttk.Label(wf_alpha_frame, text="Transparency:", font=('', 8)).pack(side=tk.LEFT)
        self.waveform_alpha_scale = ttk.Scale(
            wf_alpha_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self.waveform_alpha, length=150,
            command=self.on_waveform_alpha_change)
        self.waveform_alpha_scale.pack(side=tk.LEFT, padx=4)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # CUSTOM CONTROLS — populated by subclass via setup_custom_controls()
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(scrollable_frame, text="Custom Controls:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.setup_custom_controls()

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # ACTIONS
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(scrollable_frame, text="Actions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        button_grid = ttk.Frame(scrollable_frame)
        button_grid.pack(pady=2)

        for i, (text, command) in enumerate([
            ("Reset Zoom", self.reset_zoom),
            ("Next File",  self.next_file),
            ("Prev File",  self.previous_file),
        ]):
            ttk.Button(button_grid, text=text, command=command, width=12).grid(
                row=i // 3, column=i % 3, padx=2, pady=2, sticky='ew')

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # RIGHT SPECTROGRAM PANEL
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Top navigation bar
        nav_frame = ttk.Frame(self.plot_frame)
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

        ttk.Label(nav_frame,
                  text="[Click+Drag: select bbox | Ctrl+Wheel: zoom horizontal | Right-click: undo zoom]",
                  font=('', 8, 'italic')).pack(side=tk.RIGHT, padx=10)

        # Matplotlib figure and canvas
        self.fig = Figure(figsize=(10, 6))
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom navigation bar with hold-to-repeat buttons
        nav_bottom_frame = ttk.Frame(self.plot_frame)
        nav_bottom_frame.pack(fill=tk.X)

        button_center_frame = ttk.Frame(nav_bottom_frame)
        button_center_frame.pack(anchor=tk.CENTER)

        prev_btn = tk.Button(button_center_frame, text="◄ Previous", width=12, font=('', 8))
        prev_btn.pack(side=tk.LEFT, padx=3)
        prev_btn.bind('<ButtonPress-1>',   lambda e: self.start_continuous_nav('prev'))
        prev_btn.bind('<ButtonRelease-1>', lambda e: self.stop_continuous_nav())

        next_btn = tk.Button(button_center_frame, text="Next ►", width=12, font=('', 8))
        next_btn.pack(side=tk.LEFT, padx=3)
        next_btn.bind('<ButtonPress-1>',   lambda e: self.start_continuous_nav('next'))
        next_btn.bind('<ButtonRelease-1>', lambda e: self.stop_continuous_nav())

        # Zoom region info display (updated during drag)
        self.zoom_info_label = ttk.Label(self.plot_frame, text="", font=('', 8), foreground='blue')
        self.zoom_info_label.pack(pady=(2, 0))

        # Canvas event bindings — delegate to interaction module
        self.canvas.mpl_connect('button_press_event',   self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event',  self.on_motion)
        self.canvas.mpl_connect('scroll_event',         self.on_scroll)

        # Initial empty plot state
        self.ax.set_xlabel('Time (s)', fontsize=8)
        self.ax.set_ylabel('Frequency (Hz)', fontsize=8)
        self.ax.set_title('Load audio files to begin')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

        self.update_button_highlights()

    ##    <(''<)  <( ' ' )>  (>'')>
    # SUBCLASS HOOKS
    # Override in tab subclasses to inject tab-specific behavior
    ##    <(''<)  <( ' ' )>  (>'')>

    def setup_custom_controls(self):
        """Add tab-specific controls to the scrollable control panel."""
        ttk.Label(self.control_panel, text="No custom controls", font=('', 8, 'italic')).pack(pady=2)

    def process_audio(self):
        """Run tab-specific detection or analysis after audio is loaded."""
        pass

    def load_custom_data(self):
        """Load tab-specific annotation data for the current file."""
        pass

    def save_custom_data(self):
        """Save tab-specific annotation data for the current file."""
        pass

    def draw_custom_overlays(self):
        """Draw tab-specific matplotlib overlays onto self.ax."""
        pass

    def on_custom_press(self, event):
        """Handle tab-specific mouse press. Return True if event is consumed."""
        return False

    def on_custom_motion(self, event):
        """Handle tab-specific mouse motion. Return True if event is consumed."""
        return False

    def on_custom_release(self, event):
        """Handle tab-specific mouse release. Return True if event is consumed."""
        return False
    
    def on_bounding_box_selected(self, t_min, t_max, f_min, f_max):
        """Handle a committed drag-selected bounding box. No-op in base.

        Region-aware tabs override to append the box to their annotation
        list. Coordinates are pre-sorted (t_min < t_max, f_min < f_max).
        Inert on viewer tabs — drag selects nothing.
        """
        pass

    ##    <(''<)  <( ' ' )>  (>'')>
    # SPECTROGRAM COMPUTATION
    ##    <(''<)  <( ' ' )>  (>'')>

    def compute_spectrogram(self):
        """Compute spectrogram from current audio using current parameter tk vars."""
        self.S_db, self.freqs, self.times = audio_utils.compute_spectrogram_unified(
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

    def recompute_spectrogram(self):
        """Recompute spectrogram and redraw, preserving current zoom limits."""
        if self.y is None:
            return

        # Preserve zoom state across recompute
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.compute_spectrogram()
        self.spec_image = None
        self.process_audio()
        self.update_display(recompute_spec=True)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def change_nfft(self, new_nfft):
        """Update n_fft parameter and recompute spectrogram."""
        # NOTE: subclasses with grid caches (e.g. GridLayer) must override
        # and call super() after clearing their cache
        self.n_fft.set(new_nfft)
        self.update_button_highlights()
        if self.y is not None:
            self.recompute_spectrogram()

    def change_hop(self, new_hop):
        """Update hop_length parameter and recompute spectrogram."""
        # NOTE: subclasses with grid caches (e.g. GridLayer) must override
        # and call super() after clearing their cache
        self.hop_length.set(new_hop)
        self.update_button_highlights()
        if self.y is not None:
            self.recompute_spectrogram()

    def toggle_scale(self):
        """Toggle between linear and mel frequency scale and recompute."""
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
        """Convert Hz frequency limits to the current display scale (linear or mel)."""
        if self.y_scale.get() == 'mel':
            return audio_utils.hz_to_mel(fmin_hz), audio_utils.hz_to_mel(fmax_hz)
        return fmin_hz, fmax_hz

    ##    <(''<)  <( ' ' )>  (>'')>
    # DISPLAY
    # Delegates rendering to visualization module
    ##    <(''<)  <( ' ' )>  (>'')>

    def update_display(self, recompute_spec=False):
        """Redraw the spectrogram panel. Delegates to visualization.update_display()."""
        visualization.update_display(self, recompute_spec=recompute_spec)

    def update_display_range(self):
        """Apply current fmin/fmax display limits to the y-axis without recomputing."""
        visualization.update_display_range(self)

    def reset_zoom(self):
        """Pop zoom stack or reset to full view if stack is empty."""
        visualization.reset_zoom(self)

    def on_waveform_alpha_change(self, value):
        """Redraw waveform overlay when transparency slider moves."""
        if self.show_waveform.get():
            self.update_display()

    ##    <(''<)  <( ' ' )>  (>'')>
    # BUTTON HIGHLIGHTS
    ##    <(''<)  <( ' ' )>  (>'')>

    def update_button_highlights(self):
        """Highlight the currently active n_fft and hop_length buttons."""
        for btn, val in self.nfft_buttons:
            if val == self.n_fft.get():
                btn.config(bg='lightgreen', relief=tk.SUNKEN)
            else:
                btn.config(bg='SystemButtonFace', relief=tk.RAISED)

        for btn, val in self.hop_buttons:
            if val == self.hop_length.get():
                btn.config(bg='lightblue', relief=tk.SUNKEN)
            else:
                btn.config(bg='SystemButtonFace', relief=tk.RAISED)

    ##    <(''<)  <( ' ' )>  (>'')>
    # INTERACTION
    # Delegates to interaction module; subclass hooks called first
    ##    <(''<)  <( ' ' )>  (>'')>

    def on_press(self, event):
        """Route mouse press to interaction module after subclass hook check."""
        interaction.on_press(self, event)

    def on_motion(self, event):
        """Route mouse motion to interaction module after subclass hook check."""
        interaction.on_motion(self, event)

    def on_release(self, event):
        """Route mouse release to interaction module after subclass hook check."""
        interaction.on_release(self, event)

    def on_scroll(self, event):
        """Route scroll event to interaction module for zoom/pan handling."""
        interaction.on_scroll(self, event)

    ##    <(''<)  <( ' ' )>  (>'')>
    # PLAYBACK
    ##    <(''<)  <( ' ' )>  (>'')>

    def play_audio(self):
        """Play current audio signal at current gain setting."""
        if self.y is not None:
            pysoniq.set_gain(self.playback_gain.get())
            pysoniq.play(self.y, self.sr)

    def pause_audio(self):
        """Toggle pause/resume on current playback."""
        if pysoniq.is_paused():
            pysoniq.resume()
            if hasattr(self, 'pause_button'):
                self.pause_button.config(bg='yellow')
        else:
            pysoniq.pause()
            if hasattr(self, 'pause_button'):
                self.pause_button.config(bg='orange')

    def stop_audio(self):
        """Stop current playback."""
        pysoniq.stop()

    def toggle_loop(self):
        """Toggle loop mode on pysoniq playback."""
        self.loop_enabled = not self.loop_enabled
        pysoniq.set_loop(self.loop_enabled)

        if self.loop_enabled:
            self.loop_button.config(bg='orange', relief=tk.SUNKEN)
        else:
            self.loop_button.config(bg='lightblue', relief=tk.RAISED)

    def update_gain_label(self):
        """Sync gain label text and pysoniq gain to current slider value."""
        gain = self.playback_gain.get()
        self.gain_label.config(text=f"{int(gain * 100)}%")
        pysoniq.set_gain(gain)

    ##    <(''<)  <( ' ' )>  (>'')>
    # FILE NAVIGATION
    # Delegates to file_nav module
    ##    <(''<)  <( ' ' )>  (>'')>

    def load_directory(self):
        """Open directory dialog and load .wav files. Delegates to file_nav."""
        file_nav.load_directory(self)

    def load_test_audio(self):
        """Load bundled test audio files. Delegates to file_nav."""
        file_nav.load_test_audio(self)

    def _auto_load_on_startup(self):
        """Auto-load last directory or fall back to test audio on startup."""
        file_nav.auto_load_directory(self)

    def load_current_file(self):
        """Load audio and annotations for the current file index."""
        file_nav.load_current_file(self)

    def next_file(self):
        """Advance to the next file, auto-saving if changes exist."""
        file_nav.next_file(self)

    def previous_file(self):
        """Go back to the previous file, auto-saving if changes exist."""
        file_nav.previous_file(self)

    def jump_to_file(self):
        """Jump to the file number entered in the nav entry widget."""
        file_nav.jump_to_file(self)

    def start_continuous_nav(self, direction):
        """Begin hold-to-repeat navigation in the given direction."""
        file_nav.start_continuous_nav(self, direction)

    def continue_nav(self, direction):
        """Continue hold-to-repeat navigation — called by after() timer."""
        file_nav.continue_nav(self, direction)

    def stop_continuous_nav(self):
        """Cancel the hold-to-repeat navigation timer."""
        file_nav.stop_continuous_nav(self)

    def update_progress(self):
        """Update the file number entry and total label in the nav bar."""
        file_nav.update_progress(self)

    ##    <(''<)  <( ' ' )>  (>'')>
    # ANNOTATION I/O
    # Delegates to annotation_io module
    ##    <(''<)  <( ' ' )>  (>'')>

    def save_global_point_annotations(self):
        """Persist the class-level global_point_annotations dict to disk."""
        annotation_io.save_global_point_annotations(
            BaseLayer.global_point_annotations, self.annotation_dir)

    def load_global_point_annotations(self):
        """Load global_point_annotations from disk into the class-level dict."""
        BaseLayer.global_point_annotations = annotation_io.load_global_point_annotations(
            self.annotation_dir)

    def _get_point_bucket(self, scope="class"):
        """Return the annotation list for the current file and scope."""
        return annotation_io.get_point_bucket(
            BaseLayer.global_point_annotations,
            self.audio_files,
            self.current_file_idx,
            scope,
            self.__class__.__name__
        )

    def add_annotation_point(self, time_s, freq_hz, label=None, scope="class"):
        """Append a point annotation to the appropriate bucket and trigger redraw."""
        annotation_io.add_annotation_point(
            BaseLayer.global_point_annotations,
            self.audio_files,
            self.current_file_idx,
            time_s, freq_hz, label, scope,
            self.__class__.__name__
        )
        self.changes_made = True
        self.update_display()

    ##    <(''<)  <( ' ' )>  (>'')>
    # ENTRY POINT
    ##    <(''<)  <( ' ' )>  (>'')>

def main():
    """Launch BaseLayer as a standalone viewer tab."""
    root = tk.Tk()
    app = BaseLayer(root)
    root.geometry("1400x800")
    root.mainloop()


if __name__ == "__main__":
    main()

# U S A G I
# from yaaat.core.base_layer import BaseLayer
# root = tk.Tk(); app = BaseLayer(root); root.geometry("1400x800"); root.mainloop()