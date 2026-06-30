"""
Grid Binary Annotator - Built on BaseLayer
Visual grid-based binary annotation for audio slices
"""

from .base_layer import BaseLayer
from .contour_layer import ContourLayer

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pysoniq
from yaaat.audio_utils import compute_spectrogram_unified

from scipy import signal


class BinaryAnnotationLayer(BaseLayer):
    """Grid view for binary classification of audio files"""
    
    def __init__(self, root):
        # Grid-specific data
        self.grid_size = 25  # files per page
        self.current_page = 0
        self.selected_files = set()
        self.binary_columns = {}  # {column_name: {filepath: bool}}
        self.grid_spectrograms = {}  # cache
        
        # Grid display
        self.grid_fig = None
        self.grid_canvas = None
        self.grid_axes = []
        
        # Contour detection
        self.show_contour = tk.BooleanVar(value=False)
        self.contour_processor = None

        # Active annotation mode
        self.active_column = tk.StringVar(value='')
        
        super().__init__(root)
        
        if isinstance(root, tk.Tk):
            self.root.title("Grid Binary Annotator - YAAAT")
    
    
    def setup_custom_controls(self):
        """Add grid-specific controls"""
        # Column management
        ttk.Label(self.control_panel, text="Binary Annotation:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        column_frame = ttk.Frame(self.control_panel)
        column_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(column_frame, text="Column:").pack(side=tk.LEFT, padx=2)
        self.column_entry = ttk.Entry(column_frame, width=15)
        self.column_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(column_frame, text="Add", command=self.add_column, width=5).pack(side=tk.LEFT, padx=2)
        
        # Column list
        self.column_listbox = tk.Listbox(self.control_panel, height=5, font=('', 8))
        self.column_listbox.pack(fill=tk.X, pady=2)
        self.column_listbox.bind('<<ListboxSelect>>', self.select_column)
        
        ttk.Button(self.control_panel, text="Remove Column", command=self.remove_column).pack(pady=2)
        
        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # Grid controls
        ttk.Label(self.control_panel, text="Grid Controls:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        size_frame = ttk.Frame(self.control_panel)
        size_frame.pack(fill=tk.X, pady=2)
        ttk.Label(size_frame, text="Grid size:").pack(side=tk.LEFT, padx=2)
        
        for size in [16, 25, 36, 49]:
            ttk.Button(size_frame, text=str(size), width=4, 
                      command=lambda s=size: self.change_grid_size(s)).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.control_panel, text="Refresh View", command=self.update_display).pack(pady=5)

        # Contour calculation buttons etc
        ttk.Checkbutton( self.control_panel, text="Show contour overlay", variable=self.show_contour, command=self.update_display).pack(anchor=tk.W, pady=2)

        self.auto_recompute_contours = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            self.control_panel,
            text="Recompute contours on param change",
            variable=self.auto_recompute_contours
        ).pack(anchor=tk.W, pady=2)

        # Page navigation
        nav_frame = ttk.Frame(self.control_panel)
        nav_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(nav_frame, text="◄ Prev", command=self.prev_page).pack(side=tk.LEFT, padx=2)
        self.page_label = ttk.Label(nav_frame, text="Page 1/1", font=('', 8))
        self.page_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next ►", command=self.next_page).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        

        # Selection info
        self.selection_info = ttk.Label(self.control_panel, text="Selected: 0", font=('', 8))
        self.selection_info.pack(pady=2)

        # Selected files list
        ttk.Label(self.control_panel, text="Selected Files:", font=('', 8, 'bold')).pack(anchor=tk.W, pady=(5, 2))

        selection_frame = ttk.Frame(self.control_panel, height=100)
        selection_frame.pack(fill=tk.X, pady=2)
        selection_frame.pack_propagate(False)

        sel_canvas = tk.Canvas(selection_frame, highlightthickness=0)
        sel_scrollbar = ttk.Scrollbar(selection_frame, orient="vertical", command=sel_canvas.yview)
        self.selection_list_frame = ttk.Frame(sel_canvas)

        sel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        sel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sel_canvas.configure(yscrollcommand=sel_scrollbar.set)

        sel_canvas.create_window((0, 0), window=self.selection_list_frame, anchor="nw")
        self.selection_list_frame.bind("<Configure>", 
            lambda e: sel_canvas.configure(scrollregion=sel_canvas.bbox("all")))
        
        self.selection_info.pack(pady=2)
        
        # Batch actions
        ttk.Label(self.control_panel, text="Batch Actions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        batch_frame = ttk.Frame(self.control_panel)
        batch_frame.pack(pady=2)
        
        ttk.Button(batch_frame, text="Mark True", command=lambda: self.batch_annotate(True), 
                  width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(batch_frame, text="Mark False", command=lambda: self.batch_annotate(False), 
                  width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(batch_frame, text="Clear", command=self.clear_selection, 
                  width=10).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # Export
        ttk.Button(self.control_panel, text="Export to CSV", command=self.export_csv).pack(pady=2)
    
    def setup_ui(self):
        """Override to add grid view"""
        super().setup_ui()
        
        # Replace plot_frame with grid view
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Frame) and hasattr(self, 'fig'):
                        # Found the old plot frame
                        child.destroy()  # remove old figure
                        self.setup_grid_view(widget)  # insert new grid
                        break

    def setup_grid_view(self, parent):
        """Create scrollable grid spectrogram view"""
        grid_frame = ttk.Frame(parent)
        grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Page navigation
        nav_frame = ttk.Frame(grid_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(nav_frame, text="◄ Previous Page", command=self.prev_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next Page ►", command=self.next_page).pack(side=tk.LEFT, padx=5)
        
        self.page_info = ttk.Label(nav_frame, text="Page 1/1", font=('', 10))
        self.page_info.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(nav_frame, text="Click: toggle | Shift+Click: range select", 
                font=('', 8, 'italic')).pack(side=tk.RIGHT, padx=10)
        
        # Scrollable container
        scroll_container = ttk.Frame(grid_frame)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        
        self.grid_canvas_tk = tk.Canvas(scroll_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=self.grid_canvas_tk.yview)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.grid_canvas_tk.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.grid_canvas_tk.configure(yscrollcommand=scrollbar.set)
        
        # Inner frame for matplotlib figure
        self.grid_inner_frame = ttk.Frame(self.grid_canvas_tk)
        self.grid_canvas_tk.create_window((0, 0), window=self.grid_inner_frame, anchor="nw")
        
        # Matplotlib figure
        self.grid_fig = Figure()
        self.grid_canvas = FigureCanvasTkAgg(self.grid_fig, master=self.grid_inner_frame)
        self.grid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.grid_canvas.mpl_connect('button_press_event', self.on_grid_click)
        
        # Update scroll region when figure resizes
        def update_scroll_region(event=None):
            self.grid_canvas_tk.configure(scrollregion=self.grid_canvas_tk.bbox("all"))
        
        self.grid_inner_frame.bind("<Configure>", update_scroll_region)
        
        # Mousewheel scrolling
        def on_mousewheel(event):
            self.grid_canvas_tk.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        self.grid_canvas_tk.bind("<MouseWheel>", on_mousewheel)
        self.grid_canvas.get_tk_widget().bind("<MouseWheel>", on_mousewheel)
        
        self.root.after_idle(self.update_display)

    
    def process_audio(self):
        """Precompute spectrograms for current page"""
        if not self.audio_files:
            return
        
        start_idx = self.current_page * self.grid_size
        end_idx = min(start_idx + self.grid_size, len(self.audio_files))
        
        for i in range(start_idx, end_idx):
            filepath = self.audio_files[i]
            
            if str(filepath) in self.grid_spectrograms:
                continue
            
            try:
                y, sr = pysoniq.load_audio(str(filepath))
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                
                # Highpass filter
                sos = signal.butter(5, 800, btype='highpass', fs=sr, output='sos')
                y_filtered = signal.sosfilt(sos, y)
                
                # Compute spectrogram
                mel_db, freqs, times = compute_spectrogram_unified(
                    y=y_filtered,
                    sr=sr,
                    nfft=self.n_fft.get(),
                    hop=self.hop_length.get(),
                    scale='mel',
                    n_mels=64
                )
                # Note: already in dB, no power_to_db needed
                
                # Per-file standardization
                mel_standardized = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
                mel_norm = np.clip((mel_standardized + 3) / 6, 0, 1)
                
                self.grid_spectrograms[str(filepath)] = mel_norm
                
            except Exception as e:
                print(f"Error processing {filepath.name}: {e}")
                self.grid_spectrograms[str(filepath)] = None

        if self.show_contour.get():
            from .contour_layer import process_audio_file
            from contours.config import DEFAULT_CONFIG
            # Cache contour results per file
            if not hasattr(self, 'contour_cache'):
                self.contour_cache = {}


    def update_selection_display(self):
        """Update the selected files list in control panel"""
        for widget in self.selection_list_frame.winfo_children():
            widget.destroy()
        
        for filepath in sorted(self.selected_files, key=lambda p: p.name):
            ttk.Label(self.selection_list_frame, text=filepath.stem, 
                    font=('', 7), foreground='red').pack(anchor=tk.W, padx=2)


    def update_display(self, recompute_spec=False):
        """Update grid display"""

        # print(f"DEBUG: canvas_width={self.grid_canvas_tk.winfo_width()}, fig_size={self.grid_fig.get_size_inches()}")

        if not self.audio_files:
            return
        
        # self.grid_fig.clear()
        
        # Clear figure completely
        self.grid_fig.clf()
        
        # Set background color and transparency
        # self.grid_fig.patch.set_facecolor('#ffcccc') # pinkish
        self.grid_fig.patch.set_color((1.0, 0.8, 0.8, 0.5))  # RGBA tuple
        self.grid_fig.patch.set_alpha(0.5)

        # Clear canvas widget before redraw
        self.grid_canvas.draw_idle()
        
        start_idx = self.current_page * self.grid_size
        end_idx = min(start_idx + self.grid_size, len(self.audio_files))
        page_files = self.audio_files[start_idx:end_idx]

        # Calculate grid dimensions
        n_files = len(page_files)
        cols = int(np.sqrt(self.grid_size))
        rows = int(np.ceil(n_files / cols))

        # # Let figure fill canvas - get canvas size
        # canvas_width = self.grid_canvas.get_tk_widget().winfo_width()
        # canvas_height = self.grid_canvas.get_tk_widget().winfo_height()

        # Check if canvas is initialized
        canvas_width = self.grid_canvas_tk.winfo_width()
        if canvas_width < 100:
            canvas_width = 800

        fig_width = canvas_width / self.grid_fig.dpi
        cell_size = fig_width / cols
        fig_height = cell_size * rows

        self.grid_fig.set_size_inches(fig_width, fig_height, forward=True)
        self.grid_canvas.get_tk_widget().configure(width=int(canvas_width), height=int(fig_height * self.grid_fig.dpi))

        self.grid_axes = []

        for idx, filepath in enumerate(page_files):
            ax = self.grid_fig.add_subplot(rows, cols, idx + 1)
            
            self.grid_axes.append((ax, filepath))
            
            spec = self.grid_spectrograms.get(str(filepath))
            
            if spec is not None:
                # Display spectrogram
                extent = [0, 1, 0, 1]  # normalized coordinates
                im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest', extent=extent)

                # Add shading to indicate selection
                is_selected = filepath in self.selected_files
                if is_selected:
                    overlay = np.ones_like(spec) * 0.5
                    ax.imshow(overlay, aspect='auto', origin='lower', cmap='gray', alpha=0.3, interpolation='nearest', extent=extent)
                
                # Border highlighting
                border_color = None
                border_width = 2
                
                # Check annotation state
                if self.active_column.get() and self.active_column.get() in self.binary_columns:
                    col_data = self.binary_columns[self.active_column.get()]
                    if filepath in col_data:
                        border_color = 'lime' if col_data[filepath] else 'orange'
                        border_width = 2
                
                # Selection gets red border (overrides annotation color)
                if is_selected:
                    border_color = 'red'
                    border_width = 4
                
                # Apply border - set axis back on and configure spines
                ax.set_xticks([])
                ax.set_yticks([])
                if border_color:
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(border_width)
                else:
                    ax.axis('off')
                
                # Filename above plots
                text_color = 'black'
                ax.text(0.5, 1.05, filepath.stem, ha='center', va='bottom', 
                    transform=ax.transAxes, fontsize=6, color=text_color, weight='bold' if is_selected else 'normal')
                
            else:
                ax.text(0.5, 0.5, 'ERROR', ha='center', va='center')
                ax.axis('off')
            
            if self.show_contour.get():
                contour_result = self._get_contour_for_file(filepath)
                if contour_result and contour_result['component'].sum() > 0:
                    ax.imshow(contour_result['component'], aspect='auto', origin='lower', cmap='hot', alpha=0.5, interpolation='nearest', extent=extent)

                # # Debug to figure out why contour appears at the correct size and resolution but spectrogram shrinks
                # if contour_result and contour_result['component'].sum() > 0:
                #     spec = self.grid_spectrograms.get(str(filepath))
                #     print(f"DEBUG: {filepath.name}: spec shape={spec.shape}, contour shape={contour_result['component'].shape}")

        self.grid_fig.tight_layout()
        self.grid_canvas.draw()
        
        # Update page info
        total_pages = int(np.ceil(len(self.audio_files) / self.grid_size))
        self.page_info.config(text=f"Page {self.current_page + 1}/{total_pages}")
        self.page_label.config(text=f"Page {self.current_page + 1}/{total_pages}")
        
        # Update selection info
        self.selection_info.config(text=f"Selected: {len(self.selected_files)}")


    def _get_contour_for_file(self, filepath):
        if not hasattr(self, 'contour_cache'):
            self.contour_cache = {}
        
        key = str(filepath)
        if key not in self.contour_cache:
            try:
                from contours.processing import process_audio_file
                import pysoniq
                y, sr = pysoniq.load_audio(str(filepath))
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                self.contour_cache[key] = process_audio_file(str(filepath), y=y, sr=sr)
            except Exception as e:
                print(f"Contour error {filepath.name}: {e}")
                self.contour_cache[key] = None
        
        return self.contour_cache.get(key)

        
    def on_grid_click(self, event):
        """Handle grid click for selection"""
        if event.inaxes is None:
            return
        
        # Find which file was clicked
        clicked_file = None
        for ax, filepath in self.grid_axes:
            if ax == event.inaxes:
                clicked_file = filepath
                break
        
        if clicked_file is None:
            return
        
        # Check for shift-click (range select)
        import sys
        is_shift = False
        if sys.platform == 'win32':
            import ctypes
            is_shift = bool(ctypes.windll.user32.GetKeyState(0x10) & 0x8000)
        
        if is_shift and self.selected_files:
            # Range select
            last_selected = list(self.selected_files)[-1]
            try:
                idx1 = self.audio_files.index(last_selected)
                idx2 = self.audio_files.index(clicked_file)
                start, end = sorted([idx1, idx2])
                
                for i in range(start, end + 1):
                    self.selected_files.add(self.audio_files[i])
            except ValueError:
                pass
        else:
            # Toggle selection
            if clicked_file in self.selected_files:
                self.selected_files.remove(clicked_file)
            else:
                self.selected_files.add(clicked_file)
        
        self.update_selection_display()
        self.update_display()
    
    def add_column(self):
        """Add new binary annotation column"""
        col_name = self.column_entry.get().strip()
        
        if not col_name:
            messagebox.showwarning("Empty Name", "Enter column name")
            return
        
        if col_name in self.binary_columns:
            messagebox.showwarning("Exists", f"Column '{col_name}' already exists")
            return
        
        self.binary_columns[col_name] = {}
        self.column_listbox.insert(tk.END, col_name)
        self.column_entry.delete(0, tk.END)
        
        print(f"✓ Added column: {col_name}")
    
    def remove_column(self):
        """Remove selected column"""
        selection = self.column_listbox.curselection()
        if not selection:
            return
        
        col_name = self.column_listbox.get(selection[0])
        
        if messagebox.askyesno("Remove", f"Remove column '{col_name}'?"):
            del self.binary_columns[col_name]
            self.column_listbox.delete(selection[0])
            if self.active_column.get() == col_name:
                self.active_column.set('')
            self.update_display()
    
    def select_column(self, event):
        """Select active annotation column"""
        selection = self.column_listbox.curselection()
        if not selection:
            return
        
        col_name = self.column_listbox.get(selection[0])
        self.active_column.set(col_name)
        self.update_display()
        print(f"✓ Active column: {col_name}")
    
    def batch_annotate(self, value):
        """Annotate all selected files"""
        if not self.active_column.get():
            messagebox.showwarning("No Column", "Select annotation column first")
            return
        
        if not self.selected_files:
            messagebox.showwarning("No Selection", "Select files first")
            return
        
        col_name = self.active_column.get()
        for filepath in self.selected_files:
            self.binary_columns[col_name][filepath] = value
        
        self.changes_made = True
        self.update_display()
        print(f"✓ Marked {len(self.selected_files)} files as {value} in '{col_name}'")
    
    def clear_selection(self):
        """Clear current selection"""
        self.selected_files.clear()
        self.update_selection_display()
        self.update_display()
    
    def change_grid_size(self, new_size):
        """Change grid size"""
        self.grid_size = new_size
        self.current_page = 0
        self.process_audio()
        self.root.after_idle(self.update_display)
    
    def prev_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.process_audio()
            self.update_display()
    
    def next_page(self):
        """Go to next page"""
        total_pages = int(np.ceil(len(self.audio_files) / self.grid_size))
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.process_audio()
            self.update_display()
    
    def export_csv(self):
        """Export annotations to CSV"""
        if not self.audio_files:
            messagebox.showwarning("No Data", "Load audio files first")
            return
        
        # Build dataframe
        from natsort import natsorted
        import re
        
        rows = []
        for filepath in self.audio_files:
            path = Path(filepath)
            
            # Extract index
            match = re.search(r'slice_(\d+)\.wav', path.name)
            file_index = int(match.group(1)) if match else None
            
            row = {
                'parent_directory': path.parent.name,
                'filename': path.name,
                'index': file_index,
                'filepath': str(filepath),
                'grandparent_directory': path.parent.parent.name,
            }
            
            # Add binary columns
            for col_name, col_data in self.binary_columns.items():
                row[col_name] = col_data.get(filepath, None)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Natural sort
        df = df.sort_values(
            by=['parent_directory', 'index'],
            key=lambda x: pd.Series(natsorted(x)) if x.name == 'parent_directory' else x
        )
        df = df.reset_index(drop=True)
        
        # Save
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="binary_annotations.csv"
        )
        
        if filepath:
            df.to_csv(filepath, index=False)
            print(f"✓ Exported to {filepath}")
            messagebox.showinfo("Exported", f"Saved to:\n{filepath}")
    
    def save_custom_data(self):
        """Save binary annotations"""
        if not self.annotation_dir:
            return
        
        import json
        
        # Convert Path objects to strings for JSON
        serializable_data = {}
        for col_name, col_data in self.binary_columns.items():
            serializable_data[col_name] = {str(k): v for k, v in col_data.items()}
        
        save_file = self.annotation_dir / "grid_binary_annotations.json"
        with open(save_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        self.changes_made = False
        print(f"✓ Saved annotations to {save_file}")
    
    def load_custom_data(self):
        """Load binary annotations"""
        if not self.annotation_dir:
            return
        
        import json
        
        load_file = self.annotation_dir / "grid_binary_annotations.json"
        if load_file.exists():
            with open(load_file, 'r') as f:
                data = json.load(f)
                
                # Convert strings back to Path objects
                self.binary_columns = {}
                for col_name, col_data in data.items():
                    self.binary_columns[col_name] = {
                        Path(k): v for k, v in col_data.items()
                    }
                
                # Update column listbox
                self.column_listbox.delete(0, tk.END)
                for col_name in self.binary_columns.keys():
                    self.column_listbox.insert(tk.END, col_name)
                
                print(f"✓ Loaded annotations from {load_file}")


def main():
    root = tk.Tk()
    app = BinaryAnnotationLayer(root)
    root.geometry("1400x900")
    root.mainloop()


if __name__ == "__main__":
    main()
