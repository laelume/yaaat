"""
tabs/binary_annotator.py

Binary annotator tab for YAAAT.
Inherits from GridLayer — paginated multi-file grid spectrogram view.

Responsibilities:
    - Display audio files as a paginated grid of spectrograms
    - Click to toggle file selection, Shift+Click for range selection
    - Define named binary annotation columns (e.g. 'has_noise', 'has_bifurcation')
    - Batch annotate selected files True/False per column
    - Optional contour overlay per grid cell (from contours.processing)
    - Export annotations to CSV
    - Save/load per-audio _binary.json via annotation_io merge-write
    - Save/load column schema to _binary_columns.json (dataset-level)

File schema (Option B — hybrid):
    _binary_columns.json  — dataset-level column definitions (names, order)
        Written once when columns are added/removed.
        Lives in annotation_dir, not per-audio.
    {prefix}_{stem}_binary.json — per-audio label values
        {'binary_labels': {'has_noise': true, ...}, spec_params, psd_params}
        null = not yet labeled (distinct from false).

GridLayer responsibilities (inherited):
    - grid_fig, grid_canvas, grid_axes
    - grid_spectrograms cache (cleared on n_fft/hop change via change_nfft/change_hop)
    - change_grid_size(), next_page(), previous_page()
    - setup_grid_view() — called from GridLayer.setup_custom_controls()

Contour overlay:
    Optional overlay from contours.processing.process_audio_file().
    Cached per file in self.contour_cache.
    Enabled via show_contour checkbox.
    Recomputed on param change if auto_recompute_contours is set.

Spectrogram computation:
    Uses highpass filter (800 Hz Butterworth 5th order) before spectrogram.
    Mel scale, n_mels=64, per-file standardization and normalization.
    Stored in grid_spectrograms cache keyed by str(filepath).

Annotation file: {prefix}_{stem}_binary.json
    Written via annotation_io.merge_and_save().
    Carries contour_source key pointing to _changepoints.json.
"""

import logging
import traceback
import re

import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
from natsort import natsorted

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pysoniq
from yaaat.core.grid_layer import GridLayer
from yaaat.core import audio_utils
from yaaat.core import annotation_io
from yaaat.core.annotation_io import (
    SUFFIX_BINARY,
    SUFFIX_CHANGEPOINTS,
    resolve_annotation_path,
    merge_and_save,
    load_and_check_params,
    build_spec_params,
    build_psd_params,
)

logger = logging.getLogger(__name__)


# (つ -' _ '- )つ    (つ -' _ '- )つ
# COLUMN SCHEMA FILE
# Single file per annotation directory — dataset-level, not per-audio.
# (つ -' _ '- )つ    (つ -' _ '- )つ

_BINARY_COLUMNS_FILENAME = "_binary_columns.json"

# (つ -' _ '- )つ    (つ -' _ '- )つ
# HIGHPASS FILTER PARAMETERS
# Applied before spectrogram computation to suppress low-frequency noise.
# 800 Hz cutoff, 5th order Butterworth.
# (つ -' _ '- )つ    (つ -' _ '- )つ

_HIGHPASS_CUTOFF_HZ = 800
_HIGHPASS_ORDER     = 5

# (つ -' _ '- )つ    (つ -' _ '- )つ
# GRID SPECTROGRAM PARAMETERS
# Mel scale with 64 bands — sufficient resolution for binary annotation.
# Per-file standardization: (S - mean) / std, clipped to [-3, 3], rescaled to [0, 1].
# (つ -' _ '- )つ    (つ -' _ '- )つ

_GRID_N_MELS         = 64
_GRID_STANDARDIZE_STD = 3.0


##    <(''<)  <( ' ' )>  (>'')>

class BinaryAnnotator(GridLayer):
    """Binary annotation tab — paginated grid view for dataset-level labeling.

    Each grid cell shows a mel spectrogram for one audio file.
    Files are selected and batch-annotated True/False per named column.
    """

    def __init__(self, root):
        """Initialize binary annotation state before calling GridLayer.__init__."""

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # SELECTION STATE
        # selected_files: set of Path objects currently selected in grid
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.selected_files = set()

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # BINARY ANNOTATION DATA
        # binary_columns: {column_name: {Path: bool or None}}
        #   None = not yet labeled, distinct from False
        # active_column: currently selected column for batch annotation
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.binary_columns = {}
        self.active_column  = tk.StringVar(value='')

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # CONTOUR OVERLAY STATE
        # Contour overlay is optional — computationally expensive.
        # contour_cache: {str(filepath): contour_result or None}
        # auto_recompute_contours: if True, clears cache on param change
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.show_contour            = tk.BooleanVar(value=False)
        self.auto_recompute_contours = tk.BooleanVar(value=False)
        self.contour_cache           = {}

        super().__init__(root)

        if isinstance(root, tk.Tk):
            self.root.title("Binary Annotator - YAAAT")

    ##    <(''<)  <( ' ' )>  (>'')>
    # GRID CONTROLS — injected into GridLayer.setup_custom_controls()
    ##    <(''<)  <( ' ' )>  (>'')>

    def setup_grid_controls(self):
        """Add binary-annotation-specific controls below grid size selector."""

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # COLUMN MANAGEMENT
        # Columns define what binary labels exist in the dataset.
        # Column schema is saved to _binary_columns.json — dataset-level.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Binary Annotation:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        column_frame = ttk.Frame(self.control_panel)
        column_frame.pack(fill=tk.X, pady=2)

        ttk.Label(column_frame, text="Column:").pack(side=tk.LEFT, padx=2)
        self.column_entry = ttk.Entry(column_frame, width=15)
        self.column_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(column_frame, text="Add", command=self._add_column,
                   width=5).pack(side=tk.LEFT, padx=2)

        self.column_listbox = tk.Listbox(self.control_panel, height=5,
                                         font=('', 8))
        self.column_listbox.pack(fill=tk.X, pady=2)
        self.column_listbox.bind('<<ListboxSelect>>', self._select_column)

        ttk.Button(self.control_panel, text="Remove Column",
                   command=self._remove_column).pack(pady=2)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # CONTOUR OVERLAY
        # Contour detection is expensive — disabled by default.
        # auto_recompute_contours clears cache when spectrogram params change.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Checkbutton(self.control_panel, text="Show Contour Overlay",
                        variable=self.show_contour,
                        command=self.update_grid_display).pack(
                            anchor=tk.W, pady=2)

        ttk.Checkbutton(self.control_panel,
                        text="Recompute contours on param change",
                        variable=self.auto_recompute_contours).pack(
                            anchor=tk.W, pady=2)

        ttk.Button(self.control_panel, text="Refresh View",
                   command=self.update_grid_display).pack(pady=5)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # SELECTION INFO AND LIST
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.selection_info = ttk.Label(self.control_panel,
                                        text="Selected: 0", font=('', 8))
        self.selection_info.pack(pady=2)

        ttk.Label(self.control_panel, text="Selected Files:",
                  font=('', 8, 'bold')).pack(anchor=tk.W, pady=(5, 2))

        selection_frame = ttk.Frame(self.control_panel, height=100)
        selection_frame.pack(fill=tk.X, pady=2)
        selection_frame.pack_propagate(False)

        sel_canvas    = tk.Canvas(selection_frame, highlightthickness=0)
        sel_scrollbar = ttk.Scrollbar(selection_frame, orient="vertical",
                                      command=sel_canvas.yview)
        self.selection_list_frame = ttk.Frame(sel_canvas)

        sel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        sel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sel_canvas.configure(yscrollcommand=sel_scrollbar.set)
        sel_canvas.create_window((0, 0), window=self.selection_list_frame,
                                 anchor="nw")
        self.selection_list_frame.bind(
            "<Configure>",
            lambda e: sel_canvas.configure(
                scrollregion=sel_canvas.bbox("all")))

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # BATCH ACTIONS
        # Mark True/False/Clear for all currently selected files.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Batch Actions:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        batch_frame = ttk.Frame(self.control_panel)
        batch_frame.pack(pady=2)

        ttk.Button(batch_frame, text="Mark True",
                   command=lambda: self._batch_annotate(True),
                   width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(batch_frame, text="Mark False",
                   command=lambda: self._batch_annotate(False),
                   width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(batch_frame, text="Clear",
                   command=self._clear_selection,
                   width=10).pack(side=tk.LEFT, padx=2)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=3)

        ttk.Button(self.control_panel, text="Export to CSV",
                   command=self._export_csv).pack(pady=2)

        ttk.Button(self.control_panel, text="Save Annotations",
                   command=self.save_custom_data).pack(pady=2)

    ##    <(''<)  <( ' ' )>  (>'')>
    # GRID SPECTROGRAM COMPUTATION
    # Overrides GridLayer.process_grid_page() to populate grid_spectrograms
    # cache with mel spectrograms for the current page.
    ##    <(''<)  <( ' ' )>  (>'')>

    def process_grid_page(self):
        """Compute and cache mel spectrograms for files on the current page.

        Applies 800 Hz highpass filter before spectrogram computation.
        Uses mel scale with 64 bands for compact grid display.
        Per-file standardization: clips to ±3 std, rescales to [0, 1].
        Skips files already in cache.
        """
        if not self.audio_files:
            return

        start_idx = self.current_page * self.grid_size
        end_idx   = min(start_idx + self.grid_size, len(self.audio_files))

        for i in range(start_idx, end_idx):
            filepath = self.audio_files[i]

            if str(filepath) in self.grid_spectrograms:
                continue

            try:
                y, sr = pysoniq.load_audio(str(filepath))
                if y.ndim > 1:
                    y = np.mean(y, axis=1)

                # (つ -' _ '- )つ    (つ -' _ '- )つ
                # Highpass filter — suppresses low-frequency noise before
                # mel spectrogram computation. Butterworth 5th order, 800 Hz.
                # (つ -' _ '- )つ    (つ -' _ '- )つ
                sos        = signal.butter(
                    _HIGHPASS_ORDER, _HIGHPASS_CUTOFF_HZ,
                    btype='highpass', fs=sr, output='sos')
                y_filtered = signal.sosfilt(sos, y)

                mel_db, freqs, times = audio_utils.compute_spectrogram_unified(
                    y=y_filtered, sr=sr,
                    nfft=self.n_fft.get(),
                    hop=self.hop_length.get(),
                    scale='mel',
                    n_mels=_GRID_N_MELS
                )

                # (つ -' _ '- )つ    (つ -' _ '- )つ
                # Per-file standardization — normalizes each file independently
                # so grid cells are visually comparable regardless of amplitude.
                # Clip to ±3 std, rescale to [0, 1].
                # (つ -' _ '- )つ    (つ -' _ '- )つ
                mel_std  = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
                mel_norm = np.clip(
                    (mel_std + _GRID_STANDARDIZE_STD) / (2 * _GRID_STANDARDIZE_STD),
                    0, 1)

                self.grid_spectrograms[str(filepath)] = mel_norm

            except Exception as e:
                logger.error("Error processing %s: %s", filepath.name, e)
                self.grid_spectrograms[str(filepath)] = None

    ##    <(''<)  <( ' ' )>  (>'')>
    # GRID CELL RENDERING
    ##    <(''<)  <( ' ' )>  (>'')>

    def draw_grid_overlays(self, ax, file_idx):
        """Draw selection highlight, annotation border, and optional contour overlay.

        Border colors:
            red (4px)  — file is selected
            lime (2px) — labeled True in active column
            orange (2px) — labeled False in active column
            none — unlabeled

        Contour overlay drawn on top if show_contour is enabled.
        """
        filepath  = self.audio_files[file_idx]
        is_selected = filepath in self.selected_files

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # Selection overlay — semi-transparent gray over selected cells
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        if is_selected:
            spec = self.grid_spectrograms.get(str(filepath))
            if spec is not None:
                overlay = np.ones_like(spec) * 0.5
                ax.imshow(overlay, aspect='auto', origin='lower',
                          cmap='gray', alpha=0.3,
                          interpolation='nearest', extent=[0, 1, 0, 1])

        # Spine border color reflects annotation state
        border_color = None
        border_width = 2

        if (self.active_column.get() and
                self.active_column.get() in self.binary_columns):
            col_data = self.binary_columns[self.active_column.get()]
            val      = col_data.get(filepath)
            if val is True:
                border_color = 'lime'
            elif val is False:
                border_color = 'orange'
            # None = unlabeled, no border color

        # Selection overrides annotation border
        if is_selected:
            border_color = 'red'
            border_width = 4

        ax.set_xticks([])
        ax.set_yticks([])

        if border_color:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)
        else:
            ax.axis('off')

        # Filename label above cell
        ax.text(0.5, 1.05, filepath.stem,
                ha='center', va='bottom',
                transform=ax.transAxes,
                fontsize=6,
                color='black',
                weight='bold' if is_selected else 'normal')

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # Contour overlay — drawn on top of spectrogram if enabled.
        # Uses cached result from _get_contour_for_file().
        # Displayed as hot colormap at 0.5 alpha over normalized [0,1] extent.
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        if self.show_contour.get():
            contour_result = self._get_contour_for_file(filepath)
            if (contour_result is not None and
                    contour_result.get('component') is not None and
                    contour_result['component'].sum() > 0):
                ax.imshow(contour_result['component'],
                          aspect='auto', origin='lower',
                          cmap='hot', alpha=0.5,
                          interpolation='nearest',
                          extent=[0, 1, 0, 1])

    def _get_contour_for_file(self, filepath):
        """Return cached contour result for filepath, computing if needed.

        Uses contours.processing.process_audio_file() — optional dependency.
        Returns None if the contours package is not available or processing fails.

        # FALLBACK: contour computation falls back to None if contours package
        # is unavailable. Track this fallback to ensure contour overlay is
        # implemented when contours package is confirmed available.
        """
        key = str(filepath)
        if key not in self.contour_cache:
            try:
                from contours.processing import process_audio_file
                y, sr = pysoniq.load_audio(str(filepath))
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                self.contour_cache[key] = process_audio_file(
                    str(filepath), y=y, sr=sr)
            except Exception as e:
                logger.error("Contour error %s: %s", filepath.name, e)
                self.contour_cache[key] = None

        return self.contour_cache.get(key)

    ##    <(''<)  <( ' ' )>  (>'')>
    # GRID CLICK HANDLER
    ##    <(''<)  <( ' ' )>  (>'')>

    def on_grid_click(self, file_idx, event):
        """Toggle or range-select files on grid click.

        Shift+Click: range select from last selected to clicked file.
        Click: toggle selection of clicked file.
        Updates selection display and redraws grid.
        """
        filepath = self.audio_files[file_idx]

        # Detect Shift key state — Windows-native, fallback for other platforms
        is_shift = False
        try:
            import ctypes
            import sys
            if sys.platform == 'win32':
                is_shift = bool(
                    ctypes.windll.user32.GetKeyState(0x10) & 0x8000)
        except Exception:
            pass

        if is_shift and self.selected_files:
            # Range select from last selected to clicked
            last_selected = list(self.selected_files)[-1]
            try:
                idx1         = self.audio_files.index(last_selected)
                idx2         = file_idx
                start, end   = sorted([idx1, idx2])
                for i in range(start, end + 1):
                    self.selected_files.add(self.audio_files[i])
            except ValueError:
                pass
        else:
            # Toggle
            if filepath in self.selected_files:
                self.selected_files.remove(filepath)
            else:
                self.selected_files.add(filepath)

        self._update_selection_display()

    ##    <(''<)  <( ' ' )>  (>'')>
    # SPECTROGRAM PARAM OVERRIDES
    # Clear contour cache on param change if auto_recompute_contours is set.
    ##    <(''<)  <( ' ' )>  (>'')>

    def change_nfft(self, new_nfft):
        """Clear contour cache if auto-recompute enabled, then delegate to GridLayer."""
        if self.auto_recompute_contours.get():
            # FALLBACK: contour cache clear on param change — experimental.
            # Track this fallback until contour recompute is fully validated.
            self.contour_cache.clear()
        super().change_nfft(new_nfft)

    def change_hop(self, new_hop):
        """Clear contour cache if auto-recompute enabled, then delegate to GridLayer."""
        if self.auto_recompute_contours.get():
            # FALLBACK: contour cache clear on param change — experimental.
            # Track this fallback until contour recompute is fully validated.
            self.contour_cache.clear()
        super().change_hop(new_hop)

    ##    <(''<)  <( ' ' )>  (>'')>
    # COLUMN MANAGEMENT
    ##    <(''<)  <( ' ' )>  (>'')>

    def _add_column(self):
        """Add a new binary annotation column and save column schema."""
        col_name = self.column_entry.get().strip()

        if not col_name:
            messagebox.showwarning("Empty Name", "Enter column name")
            return

        if col_name in self.binary_columns:
            messagebox.showwarning(
                "Exists", f"Column '{col_name}' already exists")
            return

        # Initialize all files to None (unlabeled)
        self.binary_columns[col_name] = {
            f: None for f in self.audio_files}

        self.column_listbox.insert(tk.END, col_name)
        self.column_entry.delete(0, tk.END)

        # Save column schema immediately — dataset-level file
        self._save_column_schema()

        logger.info("Added column: %s", col_name)

    def _remove_column(self):
        """Remove selected column and update column schema."""
        selection = self.column_listbox.curselection()
        if not selection:
            return

        col_name = self.column_listbox.get(selection[0])

        if messagebox.askyesno("Remove", f"Remove column '{col_name}'?"):
            del self.binary_columns[col_name]
            self.column_listbox.delete(selection[0])
            if self.active_column.get() == col_name:
                self.active_column.set('')
            self._save_column_schema()
            self.update_grid_display()

    def _select_column(self, event):
        """Set active annotation column from listbox selection."""
        selection = self.column_listbox.curselection()
        if not selection:
            return
        col_name = self.column_listbox.get(selection[0])
        self.active_column.set(col_name)
        self.update_grid_display()
        logger.debug("Active column: %s", col_name)

    ##    <(''<)  <( ' ' )>  (>'')>
    # COLUMN SCHEMA PERSISTENCE
    # _binary_columns.json — dataset-level, one per annotation directory.
    # Stores column names and order. Written on add/remove column.
    ##    <(''<)  <( ' ' )>  (>'')>

    def _save_column_schema(self):
        """Write column names and order to _binary_columns.json.

        This is the authoritative column definition file for the dataset.
        Written to annotation_dir — not per-audio.
        """
        if self.annotation_dir is None:
            return

        schema_path = Path(self.annotation_dir) / _BINARY_COLUMNS_FILENAME
        schema = {
            "columns":     list(self.binary_columns.keys()),
            "description": "Binary annotation column schema for YAAAT binary annotator",
        }

        try:
            import json
            with open(schema_path, 'w') as f:
                import json
                json.dump(schema, f, indent=2)
            logger.debug("Saved column schema: %s", schema_path)
        except Exception as e:
            logger.error("Failed to save column schema: %s", e)

    def _load_column_schema(self):
        """Load column names from _binary_columns.json.

        Returns list of column names in order, or empty list if not found.
        """
        if self.annotation_dir is None:
            return []

        schema_path = Path(self.annotation_dir) / _BINARY_COLUMNS_FILENAME
        if not schema_path.exists():
            return []

        try:
            import json
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            return schema.get('columns', [])
        except Exception as e:
            logger.error("Failed to load column schema: %s", e)
            return []

    ##    <(''<)  <( ' ' )>  (>'')>
    # BATCH ANNOTATION
    ##    <(''<)  <( ' ' )>  (>'')>

    def _batch_annotate(self, value):
        """Set binary label for all selected files in the active column.

        value: True, False, or None (clear label).
        Triggers per-audio save for each annotated file.
        """
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

        # Save per-audio annotation files for all affected files
        for filepath in self.selected_files:
            self._save_file_annotation(filepath)

        self.update_grid_display()
        logger.info("Marked %d files as %s in '%s'",
                    len(self.selected_files), value, col_name)

    def _clear_selection(self):
        """Clear current selection without changing annotations."""
        self.selected_files.clear()
        self._update_selection_display()
        self.update_grid_display()

    ##    <(''<)  <( ' ' )>  (>'')>
    # SELECTION DISPLAY
    ##    <(''<)  <( ' ' )>  (>'')>

    def _update_selection_display(self):
        """Update the scrollable selected files list in control panel."""
        for widget in self.selection_list_frame.winfo_children():
            widget.destroy()

        for filepath in sorted(self.selected_files, key=lambda p: p.name):
            ttk.Label(self.selection_list_frame, text=filepath.stem,
                      font=('', 7), foreground='red').pack(
                          anchor=tk.W, padx=2)

        if hasattr(self, 'selection_info'):
            self.selection_info.config(
                text=f"Selected: {len(self.selected_files)}")

    ##    <(''<)  <( ' ' )>  (>'')>
    # CSV EXPORT
    ##    <(''<)  <( ' ' )>  (>'')>

    def _export_csv(self):
        """Export all binary annotations to a CSV file.

        Reads column order from _binary_columns.json (authoritative).
        Rows are sorted by parent directory then natural file index.
        Unlabeled files have None/NaN in label columns.
        """
        if not self.audio_files:
            messagebox.showwarning("No Data", "Load audio files first")
            return

        # Load column order from schema file — preserves addition order
        column_names = self._load_column_schema()
        if not column_names:
            column_names = list(self.binary_columns.keys())

        rows = []
        for filepath in self.audio_files:
            path  = Path(filepath)
            match = re.search(r'slice_(\d+)\.wav', path.name)
            file_index = int(match.group(1)) if match else None

            row = {
                'parent_directory':      path.parent.name,
                'grandparent_directory': path.parent.parent.name,
                'filename':              path.name,
                'index':                 file_index,
                'filepath':              str(filepath),
            }

            for col_name in column_names:
                col_data    = self.binary_columns.get(col_name, {})
                row[col_name] = col_data.get(filepath)

            rows.append(row)

        df = pd.DataFrame(rows)

        # Natural sort by parent directory then file index
        df = df.sort_values(
            by=['parent_directory', 'index'],
            key=lambda x: (pd.Series(natsorted(x))
                           if x.name == 'parent_directory' else x)
        ).reset_index(drop=True)

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="binary_annotations.csv"
        )

        if save_path:
            df.to_csv(save_path, index=False)
            logger.info("Exported to %s", save_path)
            messagebox.showinfo("Exported", f"Saved to:\n{save_path}")

    ##    <(''<)  <( ' ' )>  (>'')>
    # SAVE / LOAD
    # Per-audio: {prefix}_{stem}_binary.json via merge_and_save
    # Dataset-level: _binary_columns.json via _save_column_schema
    ##    <(''<)  <( ' ' )>  (>'')>

    def _save_file_annotation(self, filepath):
        """Save binary labels for a single audio file to _binary.json.

        Writes all column values for this file.
        null in JSON = None in Python = unlabeled (distinct from false).
        contour_source points to _changepoints.json.
        """
        if self.annotation_dir is None:
            return

        path = resolve_annotation_path(
            filepath, self.base_audio_dir,
            self.annotation_dir, SUFFIX_BINARY)

        changepoints_path = resolve_annotation_path(
            filepath, self.base_audio_dir,
            self.annotation_dir, SUFFIX_CHANGEPOINTS)

        # Collect label values for all columns for this file
        binary_labels = {}
        for col_name, col_data in self.binary_columns.items():
            val = col_data.get(filepath)
            # Store None as JSON null — explicitly unlabeled
            binary_labels[col_name] = val

        tab_data = {
            "audio_file":     str(filepath),
            "contour_source": changepoints_path.name,
            "binary_labels":  binary_labels,
            "spec_params":    build_spec_params(self, orientation="horizontal"),
            "psd_params":     build_psd_params(self),
            "skip":           False,
            "skip_reason":    "",
        }

        merge_and_save(path, tab_data)
        logger.debug("Saved binary annotation: %s", path.name)

    def save_custom_data(self):
        """Save binary annotations for all files that have been labeled.

        Saves per-audio _binary.json for every file with at least one
        non-None label value. Also saves column schema.
        """
        if self.annotation_dir is None:
            return

        saved_count = 0
        for filepath in self.audio_files:
            # Check if this file has any non-None labels
            has_labels = any(
                col_data.get(filepath) is not None
                for col_data in self.binary_columns.values()
            )
            if has_labels:
                self._save_file_annotation(filepath)
                saved_count += 1

        self._save_column_schema()
        self.changes_made = False
        logger.info("Saved binary annotations for %d files", saved_count)

    def load_custom_data(self):
        """Load binary annotations for the current page from _binary.json files.

        Also loads column schema from _binary_columns.json on first call.
        Per-audio files are loaded for all files on the current page.
        Files not yet annotated produce None values for all columns.
        """
        if self.annotation_dir is None:
            return

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # Load column schema first — defines what columns exist and their order.
        # Without this, per-audio files with no labels would have no columns.
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        column_names = self._load_column_schema()

        if column_names:
            # Initialize columns from schema if not already present
            for col_name in column_names:
                if col_name not in self.binary_columns:
                    self.binary_columns[col_name] = {}

            # Sync listbox to loaded schema
            self.column_listbox.delete(0, tk.END)
            for col_name in column_names:
                self.column_listbox.insert(tk.END, col_name)

        # Load per-audio annotations for all files on current page
        start_idx = self.current_page * self.grid_size
        end_idx   = min(start_idx + self.grid_size, len(self.audio_files))

        for i in range(start_idx, end_idx):
            filepath = self.audio_files[i]
            path     = resolve_annotation_path(
                filepath, self.base_audio_dir,
                self.annotation_dir, SUFFIX_BINARY)

            data = load_and_check_params(
                path, self, SUFFIX_BINARY, self.annotation_dir)

            if not data:
                continue

            binary_labels = data.get('binary_labels', {})
            for col_name, val in binary_labels.items():
                if col_name not in self.binary_columns:
                    self.binary_columns[col_name] = {}
                self.binary_columns[col_name][filepath] = val

        logger.debug("Loaded binary annotations for page %d",
                     self.current_page)

    ##    <(''<)  <( ' ' )>  (>'')>
    # PROCESS AUDIO OVERRIDE
    # GridLayer expects process_grid_page() — process_audio() triggers it.
    ##    <(''<)  <( ' ' )>  (>'')>

    def process_audio(self):
        """Trigger grid page spectrogram computation on file load."""
        self.process_grid_page()
        if self.annotation_dir:
            self.load_custom_data()

    ##    <(''<)  <( ' ' )>  (>'')>
    # LOAD DIRECTORY HOOK
    # Called after audio files are loaded — triggers column schema load
    # and initial grid page computation.
    ##    <(''<)  <( ' ' )>  (>'')>

    def load_current_file(self):
        """Override to trigger grid page load instead of single file load."""
        if not self.audio_files:
            return

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # Binary annotator does not load a single file — it loads a page.
        # BaseLayer.load_current_file() is bypassed in favor of process_audio()
        # which calls process_grid_page().
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        self.process_audio()
        self.update_grid_display()
        self.update_progress()


##    <(''<)  <( ' ' )>  (>'')>

def main():
    """Launch BinaryAnnotator as a standalone tab."""
    root = tk.Tk()
    app  = BinaryAnnotator(root)
    root.geometry("1400x900")
    root.mainloop()


if __name__ == "__main__":
    main()

# U S A G I
# from yaaat.tabs.binary_annotator import BinaryAnnotator
# root = tk.Tk(); app = BinaryAnnotator(root); root.geometry("1400x900"); root.mainloop()