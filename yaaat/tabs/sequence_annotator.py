import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import json
from natsort import natsorted
import pysoniq

# try: 
#     from yaaat import utils
# except ImportError:
#     print("/utils subdir does not exist")
#     import utils

from utils import utils 

class SequenceAnnotator:
    """Interactive tool for annotating audio sequences with start/stop times."""
    
    def __init__(self, root):
        self.root = root
        if isinstance(root, tk.Tk):
            self.root.title("Sequence Annotator")
        
        # Audio and file management
        self.audio_files = []
        self.current_file_idx = 0
        self.y = None
        self.sr = None
        self.S_db = None
        self.freqs = None
        self.times = None
        
        # Sequences: list of (start_time, end_time) tuples
        self.sequences = []
        
        # Interaction state
        self.temp_start = None
        
        # Spectrogram parameters
        self.n_fft = tk.IntVar(value=2048)
        self.hop_length = tk.IntVar(value=512)
        self.fmin_display = tk.IntVar(value=0)
        self.fmax_display = tk.IntVar(value=12000)
        
        # State tracking
        self.changes_made = False
        self.annotation_dir = None
        self.base_audio_dir = None
        
        # Statistics
        self.total_sequences_across_files = 0
        self.total_files_annotated = 0
        
        # Playback state
        self.playback_gain = tk.DoubleVar(value=1.0)
        self.loop_enabled = False
        
        # Setup UI
        self.setup_ui()
        
        # Auto-load last directory or test audio
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

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            bind_to_mousewheel(scrollable_frame)

        scrollable_frame.bind("<Configure>", on_frame_configure)

        # ===== HEADER =====
        ttk.Label(scrollable_frame, text="Sequence Annotator", font=('', 10, 'bold')).pack(pady=(0, 2))
        ttk.Label(scrollable_frame, text="Mark sequence start/stop times for audio slicing", 
                  wraplength=400, font=('', 8, 'italic')).pack(padx=5, pady=(0, 3))

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # ===== INSTRUCTIONS =====
        ttk.Label(scrollable_frame, text="Instructions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        instructions = ttk.Label(scrollable_frame, 
                                text="• Click: mark start\n• Click again: mark end\n• Creates sequence region", 
                                wraplength=400, font=('', 8))
        instructions.pack(padx=5, pady=(0, 5))

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # ===== FILE MANAGEMENT AND PLAYBACK CONTROLS =====
        file_buttons_frame = ttk.Frame(scrollable_frame)
        file_buttons_frame.pack(fill=tk.X, pady=2)

        # Left side - File Management
        load_buttons_frame = ttk.Frame(file_buttons_frame)
        load_buttons_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        ttk.Label(load_buttons_frame, text="File Management:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Button(load_buttons_frame, text="Load Audio Directory", command=self.load_directory).pack(anchor=tk.W, pady=2)
        ttk.Button(load_buttons_frame, text="Load Test Audio", command=self.load_test_audio).pack(anchor=tk.W, pady=2)

        # Vertical separator
        ttk.Separator(file_buttons_frame, orient=tk.VERTICAL).grid(row=0, column=1, sticky='ns', padx=10)

        # Right side - Playback Controls
        play_controls_frame = ttk.Frame(file_buttons_frame)
        play_controls_frame.grid(row=0, column=2, sticky='nsew', padx=(5, 0))

        ttk.Label(play_controls_frame, text="Playback Controls:", font=('', 9, 'bold')).pack(anchor=tk.CENTER, pady=(0, 2))

        controls_container = ttk.Frame(play_controls_frame)
        controls_container.pack(anchor=tk.CENTER)

        # Buttons row
        buttons_row = ttk.Frame(controls_container)
        buttons_row.pack(side=tk.LEFT, padx=(0, 10))

        play_button = tk.Button(buttons_row, text="▶", command=self.play_audio, bg='lightgreen', 
                            font=('', 12, 'bold'), width=2, height=1, relief=tk.RAISED, cursor='hand2')
        play_button.pack(side=tk.LEFT, padx=2)

        self.pause_button = tk.Button(buttons_row, text="⏸", command=self.pause_audio, bg='yellow', 
                                font=('', 12, 'bold'), width=2, height=1, relief=tk.RAISED, cursor='hand2')
        self.pause_button.pack(side=tk.LEFT, padx=2)

        stop_button = tk.Button(buttons_row, text="⏹", command=self.stop_audio, bg='lightcoral', 
                            font=('', 12, 'bold'), width=2, height=1, relief=tk.RAISED, cursor='hand2')
        stop_button.pack(side=tk.LEFT, padx=2)

        self.loop_button = tk.Button(buttons_row, text="⟳", command=self.toggle_loop, bg='lightblue', 
                            font=('', 12, 'bold'), width=2, height=1, relief=tk.RAISED, cursor='hand2')
        self.loop_button.pack(side=tk.LEFT, padx=2)

        # Vertical gain fader
        gain_frame = ttk.Frame(controls_container)
        gain_frame.pack(side=tk.LEFT)

        ttk.Label(gain_frame, text="Gain", font=('', 7)).pack()

        gain_scale = ttk.Scale(gain_frame, from_=2.0, to=0.0, variable=self.playback_gain,
                            orient=tk.VERTICAL, length=60, command=lambda v: self.update_gain_label())
        gain_scale.pack()

        self.gain_label = ttk.Label(gain_frame, text="100%", font=('', 7))
        self.gain_label.pack()

        file_buttons_frame.columnconfigure(0, weight=1)
        file_buttons_frame.columnconfigure(2, weight=1)

        self.file_label = ttk.Label(scrollable_frame, text="No files loaded", wraplength=400, font=('', 8))
        self.file_label.pack(fill=tk.X, pady=2)
        
        # Annotation file button
        self.annotationfile_button = tk.Button(scrollable_frame, text="No annotation file", 
                                        font=('', 8), relief=tk.FLAT, fg='blue', cursor='hand2',
                                        command=self.open_annotation_location, anchor='w')
        self.annotationfile_button.pack(fill=tk.X, pady=2)

        # Save directory button
        self.save_dir_button = tk.Button(scrollable_frame, text="No save directory", 
                                         font=('', 8), relief=tk.FLAT, fg='blue', cursor='hand2',
                                         command=self.open_save_location, anchor='w')
        self.save_dir_button.pack(anchor=tk.W, pady=2)

        # Sequence count info
        self.sequence_info = ttk.Label(scrollable_frame, text="Sequences: 0 | Total: 0", wraplength=600, font=('', 8))
        self.sequence_info.pack(pady=2)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # ===== SPECTROGRAM PARAMETERS =====
        ttk.Label(scrollable_frame, text="Spectrogram Parameters:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        # n_fft
        ttk.Label(scrollable_frame, text="n_fft:", font=('', 8)).pack(anchor=tk.W)
        nfft_frame = ttk.Frame(scrollable_frame)
        nfft_frame.pack(fill=tk.X, pady=2)
        for nfft in [512, 1024, 2048, 4096]:
            ttk.Button(nfft_frame, text=str(nfft), width=8,
                      command=lambda n=nfft: self.change_nfft(n)).pack(side=tk.LEFT, padx=2)
        
        # hop_length
        ttk.Label(scrollable_frame, text="hop:", font=('', 8)).pack(anchor=tk.W)
        hop_frame = ttk.Frame(scrollable_frame)
        hop_frame.pack(fill=tk.X, pady=2)
        for hop in [128, 256, 512, 1024]:
            ttk.Button(hop_frame, text=str(hop), width=8,
                      command=lambda h=hop: self.change_hop(h)).pack(side=tk.LEFT, padx=2)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # ===== DISPLAY PARAMETERS =====
        ttk.Label(scrollable_frame, text="Display Range:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        freq_frame = ttk.Frame(scrollable_frame)
        freq_frame.pack(fill=tk.X, pady=2)
        ttk.Label(freq_frame, text="Freq (Hz):", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.fmin_display, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(freq_frame, text="-", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.fmax_display, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(freq_frame, text="↻", width=2, command=self.recompute_display).pack(side=tk.LEFT, padx=2)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # ===== SEQUENCES LIST =====
        ttk.Label(scrollable_frame, text="Sequences:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        self.sequences_frame = ttk.Frame(scrollable_frame)
        self.sequences_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # ===== ACTIONS =====
        ttk.Label(scrollable_frame, text="Actions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        button_grid = ttk.Frame(scrollable_frame)
        button_grid.pack(pady=2)

        buttons = [
            ("Load Audio", self.load_directory),
            ("Remove Last", self.remove_last_sequence),
            ("Next File", self.next_file),
            ("Export TSV", self.export_tsv),
            
            ("Play Audio", self.play_audio),
            ("Clear All", self.clear_all_sequences),
            ("Prev File", self.previous_file),
            ("Save Anno", self.save_annotations),
        ]

        for i, (text, command) in enumerate(buttons):
            row = i // 4
            col = i % 4
            ttk.Button(button_grid, text=text, command=command, width=12).grid(
                row=row, column=col, padx=2, pady=2, sticky='ew')

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # ===== STATISTICS =====
        ttk.Label(scrollable_frame, text="Statistics:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.stats_label = ttk.Label(scrollable_frame, text="No sequences annotated", 
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

        self.file_number_entry.bind('<Return>', lambda e: self.jump_to_file())

        ttk.Label(nav_frame, text="[Click to mark start, click again to mark end]", 
                  font=('', 8, 'italic')).pack(side=tk.RIGHT, padx=10)
        
        # Spectrogram canvas
        self.fig = Figure(figsize=(10, 6))
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initialize empty plot
        self.ax.set_xlabel('Time (s)', fontsize=8)
        self.ax.set_ylabel('Frequency (Hz)', fontsize=8)
        self.ax.set_title('Load audio files to begin')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()
    
    # ===== PARAMETER CHANGE METHODS =====
    
    def change_nfft(self, new_nfft):
        """Change n_fft and recompute"""
        self.n_fft.set(new_nfft)
        if self.y is not None:
            self.recompute_display()
    
    def change_hop(self, new_hop):
        """Change hop_length and recompute"""
        self.hop_length.set(new_hop)
        if self.y is not None:
            self.recompute_display()
    
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

        # Ask user where to save annotations
        response = messagebox.askyesnocancel(
            "Annotation Save Location",
            "Where do you want to save sequence annotations?\n\n"
            "Yes = Choose existing directory\n"
            "No = Create new directory\n"
            "Cancel = Use default location"
        )

        if response is True:
            save_dir = filedialog.askdirectory(title="Select Annotation Save Directory")
            if save_dir:
                self.annotation_dir = Path(save_dir)
            else:
                return
        elif response is False:
            save_dir = filedialog.askdirectory(title="Select Parent Directory for New Folder")
            if save_dir:
                dataset_name = Path(directory).name
                self.annotation_dir = Path(save_dir) / f"{dataset_name}_sequence_annotations"
                self.annotation_dir.mkdir(exist_ok=True)
            else:
                return
        else:
            dataset_name = Path(directory).name
            default_dir = Path.home() / "yaaat_annotations" / f"{dataset_name}_sequences"
            default_dir.mkdir(parents=True, exist_ok=True)
            self.annotation_dir = default_dir

        self.annotation_dir.mkdir(exist_ok=True)
        print(f"Sequence annotations will be saved to: {self.annotation_dir}")
        
        self.save_dir_button.config(text=f"📁 {self.annotation_dir}")
        
        self.current_file_idx = 0
        self.count_total_sequences()
        self.load_current_file()
        
        print(f"✓ Loaded {len(self.audio_files)} files")
        utils.save_last_directory(self.base_audio_dir)

    def load_test_audio(self):
        """Load bundled test audio files"""
        test_audio_dir = Path(__file__).parent / 'test_files' / 'test_audio' / 'sequencces' / 'jelatik'
        
        if not test_audio_dir.exists():
            messagebox.showinfo("No Test Data", "Test audio files not found in package")
            return
        
        self.audio_files = natsorted(test_audio_dir.rglob('*.wav'))
        self.base_audio_dir = test_audio_dir
        
        if not self.audio_files:
            messagebox.showwarning("No Files", "No .wav files found in test directory")
            return
        
        # Set up default save directory
        default_dir = Path.home() / "yaaat_annotations" / "test_audio_sequences"
        default_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_dir = default_dir
        
        self.save_dir_button.config(text=f"📁 {self.annotation_dir}")
        
        self.current_file_idx = 0
        self.count_total_sequences()
        self.load_current_file()
        
        print(f"✓ Loaded {len(self.audio_files)} test files")
        utils.save_last_directory(self.base_audio_dir)

    def auto_load_directory(self):
        """Auto-load last directory or default test audio on startup"""
        last_dir = utils.load_last_directory()
        if last_dir and last_dir.exists():
            print(f"Auto-loading last directory: {last_dir}")
            self.audio_files = natsorted(last_dir.rglob('*.wav'))
            self.base_audio_dir = last_dir
            if self.audio_files:
                dataset_name = last_dir.name
                self.annotation_dir = Path.home() / "yaaat_annotations" / f"{dataset_name}_sequences"
                self.annotation_dir.mkdir(parents=True, exist_ok=True)
                self.save_dir_button.config(text=f"📁 {self.annotation_dir}")
                self.current_file_idx = 0
                self.count_total_sequences()
                self.load_current_file()
                return
        
        self.load_test_audio()

    def load_current_file(self):
        """Load the current audio file and its sequence annotations"""
        if not self.audio_files:
            return
        
        audio_file = self.audio_files[self.current_file_idx]
        print(f"Loading {audio_file.name}...")
        
        # Load audio using pysoniq
        self.y, self.sr = pysoniq.load(str(audio_file))
        if self.y.ndim > 1:
            self.y = np.mean(self.y, axis=1)
        
        # Set default fmax if needed
        if self.fmax_display.get() == 0 or self.fmax_display.get() > self.sr / 2:
            self.fmax_display.set(int(self.sr / 2))
        
        # Compute spectrogram
        self.compute_spectrogram()
        
        # Clear sequences
        self.sequences = []
        self.temp_start = None
        
        # Load existing annotations
        relative_path = audio_file.relative_to(self.base_audio_dir).parent
        filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
        if filename_prefix and filename_prefix != '.':
            annotation_file = self.annotation_dir / f"{filename_prefix}_{audio_file.stem}_sequence_annotations.json"
        else:
            annotation_file = self.annotation_dir / f"{audio_file.stem}_sequence_annotations.json"

        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                self.sequences = [tuple(seq) for seq in data.get('sequences', [])]
                print(f"✓ Loaded {len(self.sequences)} sequences")
        else:
            print(f"No annotation file found")
        
        self.changes_made = False
        self.update_display()
        self.update_progress()
        
        # Update annotation file button
        if annotation_file.exists() and self.sequences:
            self.annotationfile_button.config(text=f"✓ {annotation_file.name}", foreground='green')
        else:
            self.annotationfile_button.config(text=f"→ {annotation_file.name}", foreground='blue')
    
    def count_total_sequences(self):
        """Count total sequences across all annotation files"""
        self.total_sequences_across_files = 0
        self.total_files_annotated = 0
        
        for audio_file in self.audio_files:
            relative_path = audio_file.relative_to(self.base_audio_dir).parent
            filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
            if filename_prefix and filename_prefix != '.':
                annotation_file = self.annotation_dir / f"{filename_prefix}_{audio_file.stem}_sequence_annotations.json"
            else:
                annotation_file = self.annotation_dir / f"{audio_file.stem}_sequence_annotations.json"
            
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    file_seqs = len(data.get('sequences', []))
                    if file_seqs > 0:
                        self.total_sequences_across_files += file_seqs
                        self.total_files_annotated += 1

        print(f"Total sequences: {self.total_sequences_across_files} across {self.total_files_annotated} files")
    
    def open_file_location(self, path):
        """Open file or directory location in system file explorer"""
        if path is None:
            messagebox.showinfo("No Location", "No location set yet.")
            return
        
        import subprocess
        import sys
        import os
        
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

    def open_annotation_location(self):
        """Open current annotation file location"""
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            messagebox.showinfo("No File", "Load audio files first.")
            return
        
        audio_file = self.audio_files[self.current_file_idx]
        relative_path = audio_file.relative_to(self.base_audio_dir).parent
        filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
        
        if filename_prefix and filename_prefix != '.':
            annotation_file = self.annotation_dir / f"{filename_prefix}_{audio_file.stem}_sequence_annotations.json"
        else:
            annotation_file = self.annotation_dir / f"{audio_file.stem}_sequence_annotations.json"
        
        self.open_file_location(annotation_file)

    def open_save_location(self):
        """Open save directory location"""
        self.open_file_location(self.annotation_dir)
    
    # ===== DISPLAY METHODS =====
    
    def compute_spectrogram(self):
        """Compute spectrogram for display"""
        if self.y is None:
            return
        
        self.S_db, self.freqs, self.times = utils.compute_spectrogram_unified(
            self.y,
            self.sr,
            nfft=self.n_fft.get(),
            hop=self.hop_length.get(),
            fmin=self.fmin_display.get(),
            fmax=self.fmax_display.get(),
            scale='linear',
            orientation='horizontal'
        )
        
        print(f"Computed: S_db shape={self.S_db.shape}, freqs={len(self.freqs)}, times={len(self.times)}")
    
    def recompute_display(self):
        """Recompute spectrogram with current parameters"""
        if self.y is None:
            return
        
        self.compute_spectrogram()
        self.update_display()
    
    def update_display(self):
        """Update the spectrogram display with sequences"""
        if self.y is None:
            return
        
        self.ax.clear()
        
        # Plot spectrogram
        im = self.ax.pcolormesh(self.times, self.freqs, self.S_db, 
                                shading='gouraud', cmap='magma', vmin=-80, vmax=0)
        
        # Plot existing sequences
        for i, (start, end) in enumerate(self.sequences):
            # Shaded region
            self.ax.axvspan(start, end, alpha=0.2, color='red', zorder=5)
            # Boundary lines
            self.ax.axvline(start, color='red', linestyle='--', linewidth=1, alpha=0.7, zorder=6)
            self.ax.axvline(end, color='red', linestyle='--', linewidth=1, alpha=0.7, zorder=6)
            # Start/end markers (small dots)
            marker_freq = self.freqs[int(len(self.freqs) * 0.95)]
            self.ax.plot(start, marker_freq, 'ro', markersize=4, zorder=7)
            self.ax.plot(end, marker_freq, 'rs', markersize=4, zorder=7)
        
        # Plot temporary start marker if exists
        if self.temp_start is not None:
            self.ax.axvline(self.temp_start, color='orange', linestyle=':', linewidth=2, zorder=6)
            marker_freq = self.freqs[int(len(self.freqs) * 0.95)]
            self.ax.plot(self.temp_start, marker_freq, 'o', color='orange', markersize=6, zorder=7)
        
        self.ax.set_xlabel('Time (s)', fontsize=8)
        self.ax.set_ylabel('Frequency (Hz)', fontsize=8)
        self.ax.set_xlim(0, self.times[-1] if len(self.times) > 0 else 1)
        self.ax.set_ylim(self.freqs[0], self.freqs[-1])
        
        # Update title
        filename = self.audio_files[self.current_file_idx].name
        save_marker = "✓ " if not self.changes_made else ""
        self.ax.set_title(f'{save_marker}{filename} | {len(self.sequences)} sequences | '
                         f'n_fft={self.n_fft.get()} hop={self.hop_length.get()}', fontsize=9)
        
        self.canvas.draw()
        
        # Update sequence list and stats
        self.update_sequence_list()
        self.update_stats()
        self.update_sequence_info()
    
    def update_sequence_list(self):
        """Update the control panel sequence list"""
        for widget in self.sequences_frame.winfo_children():
            widget.destroy()
        
        if not self.sequences:
            ttk.Label(self.sequences_frame, text="No sequences yet", 
                     foreground='gray').pack(pady=5)
            return
        
        for i, (start, end) in enumerate(self.sequences):
            seq_frame = ttk.Frame(self.sequences_frame)
            seq_frame.pack(fill=tk.X, pady=2)
            
            label_text = f"{i+1}. {start:.3f} - {end:.3f} s ({end-start:.3f}s)"
            ttk.Label(seq_frame, text=label_text, font=('', 8)).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(seq_frame, text="Delete", 
                      command=lambda idx=i: self.delete_sequence(idx),
                      width=8).pack(side=tk.RIGHT, padx=5)
    
    def update_stats(self):
        """Update sequence statistics display"""
        if not self.sequences:
            self.stats_label.config(text="No sequences annotated")
            return
        
        durations = [end - start for start, end in self.sequences]
        mean_dur = np.mean(durations)
        std_dur = np.std(durations)
        min_dur = np.min(durations)
        max_dur = np.max(durations)
        total_dur = np.sum(durations)
        
        self.stats_label.config(
            text=f"Duration: μ={mean_dur:.3f}s, σ={std_dur:.3f}s\n"
                 f"Range: {min_dur:.3f} - {max_dur:.3f}s\n"
                 f"Total: {total_dur:.2f}s"
        )
    
    def update_sequence_info(self):
        """Update sequence count info label"""
        self.sequence_info.config(
            text=f"Sequences: {len(self.sequences)} | Total: {self.total_sequences_across_files}"
        )
    
    def update_progress(self):
        """Update file progress indicator"""
        self.file_number_entry.delete(0, tk.END)
        self.file_number_entry.insert(0, str(self.current_file_idx + 1))
        self.file_total_label.config(text=f"/ {len(self.audio_files)}")
        self.file_label.config(text=self.audio_files[self.current_file_idx].name)
    
    # ===== MOUSE EVENT HANDLERS =====
    
    def on_click(self, event):
        """Handle mouse clicks on spectrogram"""
        if event.inaxes != self.ax or self.y is None:
            return
        
        if event.xdata is None:
            return
        
        click_time = event.xdata
        
        if self.temp_start is None:
            # First click: set temporary start
            self.temp_start = click_time
            print(f"Sequence start: {click_time:.3f} s")
        else:
            # Second click: finalize sequence
            start = min(self.temp_start, click_time)
            end = max(self.temp_start, click_time)
            
            # Add sequence
            self.sequences.append((start, end))
            self.sequences.sort()  # Keep sequences in time order
            
            print(f"Sequence created: {start:.3f} - {end:.3f} s")
            
            # Clear temporary start
            self.temp_start = None
            self.changes_made = True
        
        # Update display
        self.update_display()
    
    # ===== SEQUENCE MANAGEMENT =====
    
    def delete_sequence(self, index):
        """Delete a sequence by index"""
        if 0 <= index < len(self.sequences):
            start, end = self.sequences[index]
            self.sequences.pop(index)
            self.changes_made = True
            print(f"Deleted sequence: {start:.3f} - {end:.3f} s")
            self.update_display()
    
    def remove_last_sequence(self):
        """Remove the last annotated sequence"""
        if self.sequences:
            removed = self.sequences.pop()
            self.changes_made = True
            print(f"- Removed sequence: {removed[0]:.3f} - {removed[1]:.3f} s")
            self.update_display()
    
    def clear_all_sequences(self):
        """Clear all sequences after confirmation"""
        if not self.sequences:
            return
        
        if messagebox.askyesno("Clear All", "Delete all sequences?"):
            self.sequences = []
            self.temp_start = None
            self.changes_made = True
            self.update_display()
            print("All sequences cleared")
    
    # ===== PLAYBACK CONTROLS =====
    
    def play_audio(self):
        """Play the current audio file with gain applied"""
        if self.y is not None:
            pysoniq.set_gain(self.playback_gain.get())
            pysoniq.play(self.y, self.sr)
    
    def pause_audio(self):
        """Pause audio playback"""
        if pysoniq.is_paused():
            pysoniq.resume()
            if hasattr(self, 'pause_button'):
                self.pause_button.config(bg='yellow')
            import pysoniq.pause as pause_module
            if pause_module.was_looping() and hasattr(self, 'loop_button'):
                self.loop_button.config(bg='orange', relief=tk.SUNKEN)
                self.loop_enabled = True
        else:
            pysoniq.pause()
            if hasattr(self, 'pause_button'):
                self.pause_button.config(bg='orange')
    
    def stop_audio(self):
        """Stop audio playback"""
        pysoniq.stop()
    
    def toggle_loop(self):
        """Toggle loop mode"""
        self.loop_enabled = not self.loop_enabled
        pysoniq.set_loop(self.loop_enabled)
        
        if self.loop_enabled:
            self.loop_button.config(bg='orange', relief=tk.SUNKEN)
        else:
            self.loop_button.config(bg='lightblue', relief=tk.RAISED)
        
        print(f"Loop {'enabled' if self.loop_enabled else 'disabled'}")
    
    def update_gain_label(self):
        """Update gain label and master gain when slider moves"""
        gain = self.playback_gain.get()
        gain_percent = int(gain * 100)
        self.gain_label.config(text=f"{gain_percent}%")
        pysoniq.set_gain(gain)
    
    # ===== SAVE AND EXPORT =====
    
    def save_annotations(self):
        """Save sequences to JSON"""
        try:
            if not self.audio_files or self.annotation_dir is None:
                return
            
            audio_file = self.audio_files[self.current_file_idx]
            relative_path = audio_file.relative_to(self.base_audio_dir).parent
            filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
            
            if filename_prefix and filename_prefix != '.':
                annotation_file = self.annotation_dir / f"{filename_prefix}_{audio_file.stem}_sequence_annotations.json"
            else:
                annotation_file = self.annotation_dir / f"{audio_file.stem}_sequence_annotations.json"
            
            # Calculate statistics
            if self.sequences:
                durations = [end - start for start, end in self.sequences]
                sequence_stats = {
                    'num_sequences': len(self.sequences),
                    'mean_duration': float(np.mean(durations)),
                    'std_duration': float(np.std(durations)),
                    'min_duration': float(np.min(durations)),
                    'max_duration': float(np.max(durations)),
                    'total_duration': float(np.sum(durations))
                }
            else:
                sequence_stats = {}
            
            data = {
                'audio_file': str(audio_file),
                'sequences': [[start, end] for start, end in self.sequences],
                'sequence_stats': sequence_stats,
                'spec_params': {
                    'n_fft': self.n_fft.get(),
                    'hop_length': self.hop_length.get(),
                    'fmin_display': self.fmin_display.get(),
                    'fmax_display': self.fmax_display.get()
                }
            }
            
            with open(annotation_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.changes_made = False
            self.count_total_sequences()
            self.update_display()
            self.update_sequence_info()
            print(f"✓ Saved {len(self.sequences)} sequences to {annotation_file.name}")
            
        except Exception as e:
            print(f"ERROR saving annotations: {e}")
            import traceback
            traceback.print_exc()
    
    def export_tsv(self):
        """Export sequences to TSV format for tranche pipeline"""
        if not self.sequences:
            messagebox.showwarning("Warning", "No sequences to export")
            return
        
        if not self.audio_files:
            messagebox.showwarning("Warning", "No audio file loaded")
            return
        
        audio_file = self.audio_files[self.current_file_idx]
        tsv_file = audio_file.with_suffix('.seq.tsv')
        
        try:
            with open(tsv_file, 'w') as f:
                for start, end in self.sequences:
                    f.write(f"{start:.6f}\t{end:.6f}\n")
            
            messagebox.showinfo("Success", f"Exported to {tsv_file.name}")
            print(f"✓ Exported {len(self.sequences)} sequences to TSV: {tsv_file.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    # ===== NAVIGATION =====
    
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


# ===== MAIN ENTRY POINT =====

def main():
    """Entry point for the sequence annotator application"""
    root = tk.Tk()
    root.title("YAAAT - Sequence Annotator")
    root.geometry("1400x800")
    app = SequenceAnnotator(root)
    root.mainloop()


if __name__ == "__main__":
    main()