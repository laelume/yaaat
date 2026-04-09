"""
tabs/changepoint_annotator.py

Changepoint annotator tab for YAAAT.
Inherits from BaseLayer — single-file spectrogram view.

Responsibilities:
    - Click-to-add contour points on the spectrogram
    - Lasso selection (Ctrl+drag) to group points into contours
    - Ctrl+Click dual-endpoint marking to extract contour from point range
    - Finish contour — sorts points by time, assigns onset_idx/offset_idx
    - Skip file with reason dialog
    - Find next skipped or unannotated file
    - Harmonic bounding boxes with nudge controls
    - Sequence mode display
    - Auto-finish contour on file navigation
    - Count total contours and skipped files across dataset
    - Save/load to _changepoints.json via annotation_io merge-write

Contour data structure:
    {
        'id':         str  — 'c{idx}' assigned at save time
        'points':     [{'time': float, 'freq': float}, ...]
        'onset_idx':  int  — index of onset point in points list
        'offset_idx': int  — index of offset point in points list
    }

finish_contour() implementation:
    - Requires minimum 2 points in current_contour
    - Sorts points by time
    - Appends as dict with onset_idx=0, offset_idx=len-1
    - Takes silent=False parameter for programmatic calls during navigation
    - Returns True on success, False on failure
    - This is the merged implementation from ChangepointLayer (document 6)
      and the second finish_contour definition in ChangepointAnnotator
      (document 7). The first definition in document 7 is discarded.

Old format conversion:
    Contours saved as bare lists (old format) are converted inline on load
    to the dict format with onset_idx/offset_idx. No separate conversion
    function — handled directly in load_custom_data().

Annotation file: {prefix}_{stem}_changepoints.json
    Written via annotation_io.merge_and_save() — canonical geometry file.
    All other tab files carry a contour_source key pointing to this file.
"""

import json
import logging
import traceback
import sys

import numpy as np
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from yaaat.core.base_layer import BaseLayer
from yaaat.core import annotation_io
from yaaat.core.annotation_io import (
    SUFFIX_CHANGEPOINTS,
    resolve_annotation_path,
    merge_and_save,
    load_and_check_params,
    mark_skip,
    is_skipped,
    compute_contour_metrics,
    build_spec_params,
    build_psd_params,
)

logger = logging.getLogger(__name__)

from yaaat.config import CONFIG


# (つ -' _ '- )つ    (つ -' _ '- )つ
# ANNOTATION LABEL COLORS
# onset=green, offset=magenta, changepoint=cyan — consistent across all views
# (つ -' _ '- )つ    (つ -' _ '- )つ

_LABEL_COLORS = {
    'onset':       'green',
    'offset':      'magenta',
    'changepoint': 'cyan',
}


##    <(''<)  <( ' ' )>  (>'')>

class ChangepointAnnotator(BaseLayer):
    """Changepoint annotation tab — click-to-add contour points with full
    lasso, endpoint marking, skip, and harmonic bounding box support.
    """

    def __init__(self, root):
        """Initialize changepoint state before calling BaseLayer.__init__."""

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # CONTOUR STATE
        # current_contour: points being built, not yet finished
        # contours: list of finished contour dicts
        # annotations: flat list rebuilt from contours for display
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.current_contour = []
        self.contours        = []
        self.annotations     = []
        self.changepoints    = []

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # ANNOTATION MODE
        # 'contour' — click points, finish contour
        # 'sequence' — mark syllable boundaries
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.annotation_mode = tk.StringVar(value='contour')

        # Ctrl+Click endpoint marking state
        # pending_onset_idx: first Ctrl+Click info, waiting for second click
        self.pending_onset_idx = None

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # LASSO STATE
        # lasso_mode: True while drag-lasso is active
        # lasso_points: list of (x, y) vertices
        # lasso_lines: matplotlib Line2D objects for visual feedback
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.lasso_mode   = False
        self.lasso_points = []
        self.lasso_lines  = []

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # GUIDE AND BOUNDING BOX STATE
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.show_time_guides    = tk.BooleanVar(value=False)
        self.show_freq_guides    = tk.BooleanVar(value=False)
        self.show_all_guides_var = tk.BooleanVar(value=False)
        self.hide_text           = tk.BooleanVar(value=False)
        self.show_bounding_box   = tk.BooleanVar(value=False)
        self.bounding_box_shape  = tk.StringVar(value='rectangle')

        # Harmonic bounding boxes — 2nd and 3rd harmonic overlays
        # Each entry: {'multiplier': DoubleVar, 'show': BooleanVar,
        #              'label': ttk.Label or None, 'color': str, 'name': str}
        self.harmonics = [
            {
                'multiplier': tk.DoubleVar(value=2.0),
                'show':       tk.BooleanVar(value=False),
                'label':      None,
                'color':      'cyan',
                'name':       '2nd'
            },
            {
                'multiplier': tk.DoubleVar(value=3.0),
                'show':       tk.BooleanVar(value=False),
                'label':      None,
                'color':      'orange',
                'name':       '3rd'
            },
        ]
        self.harmonic_repeat_ids = {}
        self.dragging_harmonic   = None

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # TRACKING ACROSS FILES
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.total_contours_across_files = 0
        self.total_skipped_files         = 0
        self.file_was_annotated          = False

        super().__init__(root)

        if isinstance(root, tk.Tk):
            self.root.title("Changepoint Annotator - YAAAT")

    ##    <(''<)  <( ' ' )>  (>'')>
    # CUSTOM CONTROLS
    ##    <(''<)  <( ' ' )>  (>'')>

    def setup_custom_controls(self):
        """Build changepoint-specific controls in the scrollable control panel."""

        # Instructions
        ttk.Label(self.control_panel, text="Instructions:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(
            self.control_panel,
            text=(
                "• Click: add point\n"
                "• Click near point: remove\n"
                "• Ctrl+Drag: lasso selection\n"
                "• Ctrl+Click+Click: mark endpoints\n"
                "• Right-click: undo zoom"
            ),
            wraplength=400, font=('', 8)
        ).pack(padx=5, pady=(0, 5))

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # ANNOTATION MODE
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Mode:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        mode_frame = ttk.Frame(self.control_panel)
        mode_frame.pack(fill=tk.X, pady=2)

        ttk.Radiobutton(mode_frame, text="Syllable Mode",
                        variable=self.annotation_mode, value='contour',
                        command=self.switch_annotation_mode).pack(
                            side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Sequence Mode",
                        variable=self.annotation_mode, value='sequence',
                        command=self.switch_annotation_mode).pack(
                            side=tk.LEFT, padx=5)

        self.mode_instructions = ttk.Label(
            self.control_panel,
            text="Syllable Mode: Click points → Finish Contour",
            wraplength=400, font=('', 8, 'italic'), foreground='blue'
        )
        self.mode_instructions.pack(pady=2)

        # Sequence display — hidden in syllable mode
        self.sequence_display_frame = ttk.LabelFrame(
            self.control_panel, text="Contour Sequence", padding=5)
        self.sequence_display_frame.pack(fill=tk.X, pady=5)
        self.sequence_display_frame.pack_forget()

        seq_header = ttk.Frame(self.sequence_display_frame)
        seq_header.pack(fill=tk.X)
        for label, width in [('#', 3), ('t_on', 7), ('f_on', 7),
                              ('t_off', 7), ('f_off', 7), ('Δt', 6), ('pts', 4)]:
            ttk.Label(seq_header, text=label,
                      font=('', 8, 'bold'), width=width).pack(side=tk.LEFT)

        seq_canvas    = tk.Canvas(self.sequence_display_frame,
                                  height=200, highlightthickness=0)
        seq_scrollbar = ttk.Scrollbar(self.sequence_display_frame,
                                      orient="vertical", command=seq_canvas.yview)
        self.sequence_inner_frame = ttk.Frame(seq_canvas)

        seq_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        seq_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        seq_canvas.configure(yscrollcommand=seq_scrollbar.set)
        seq_canvas.create_window((0, 0), window=self.sequence_inner_frame,
                                 anchor="nw")
        self.sequence_inner_frame.bind(
            "<Configure>",
            lambda e: seq_canvas.configure(
                scrollregion=seq_canvas.bbox("all")))

        # Contour info
        self.contour_info = ttk.Label(
            self.control_panel,
            text="Unsaved Points: 0 | Saved Points: 0 | Contours: 0",
            wraplength=600, font=('', 8), justify=tk.LEFT
        )
        self.contour_info.pack(pady=2)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # CONTOUR ACTIONS
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Contour Actions:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        button_grid = ttk.Frame(self.control_panel)
        button_grid.pack(pady=2)

        actions = [
            ("Finish Contour", self.finish_contour),
            ("Clear Last",     self.clear_last),
            ("Clear All",      self.clear_all),
            ("Skip File",      self.skip_file),
            ("Save Anno",      self.save_custom_data),
            ("Find Skipped",   self.find_next_skipped),
        ]

        for i, (text, command) in enumerate(actions):
            ttk.Button(button_grid, text=text, command=command,
                       width=12).grid(
                row=i // 3, column=i % 3,
                padx=2, pady=2, sticky='ew')

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # GUIDES AND BOUNDING BOX
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Guides:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        guides_grid = ttk.Frame(self.control_panel)
        guides_grid.pack(fill=tk.X, pady=2)

        ttk.Checkbutton(guides_grid, text="Time Lines",
                        variable=self.show_time_guides,
                        command=self._toggle_guides).grid(
                            row=0, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Freq Lines",
                        variable=self.show_freq_guides,
                        command=self._toggle_guides).grid(
                            row=0, column=1, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Show All",
                        variable=self.show_all_guides_var,
                        command=self._toggle_show_all).grid(
                            row=0, column=2, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Hide Text",
                        variable=self.hide_text,
                        command=self._toggle_guides).grid(
                            row=0, column=3, sticky=tk.W, padx=2, pady=2)

        ttk.Checkbutton(guides_grid, text="Bounding Box",
                        variable=self.show_bounding_box,
                        command=self._toggle_guides).grid(
                            row=1, column=0, sticky=tk.W, padx=2, pady=2)

        for col, (val, label) in enumerate(
                [('rectangle', 'Rectangle'), ('polygon', 'Polygon'),
                 ('ellipse', 'Ellipse')], start=1):
            ttk.Radiobutton(guides_grid, text=label,
                            variable=self.bounding_box_shape, value=val,
                            command=self._toggle_guides).grid(
                                row=1, column=col, sticky=tk.W, padx=2, pady=2)

        # Harmonic bounding box nudge controls
        for i, harmonic in enumerate(self.harmonics):
            row = 2 + i
            ttk.Checkbutton(
                guides_grid,
                text=f"Bound {harmonic['name']} Harmonic",
                variable=harmonic['show'],
                command=self._toggle_guides
            ).grid(row=row, column=1, sticky=tk.W, padx=2, pady=2)

            nudge_frame = ttk.Frame(guides_grid)
            nudge_frame.grid(row=row, column=2, columnspan=2,
                             sticky=tk.W, padx=2, pady=2)

            down_btn = tk.Button(nudge_frame, text="▼", width=3, font=('', 8))
            down_btn.pack(side=tk.LEFT, padx=1)
            down_btn.bind('<ButtonPress-1>',
                          lambda e, idx=i: self._start_continuous_harmonic(idx, 'down'))
            down_btn.bind('<ButtonRelease-1>',
                          lambda e, idx=i: self._stop_continuous_harmonic(idx))

            harmonic['label'] = ttk.Label(
                nudge_frame,
                text=f"{harmonic['multiplier'].get():.2f}x",
                width=5, font=('', 8))
            harmonic['label'].pack(side=tk.LEFT, padx=2)

            up_btn = tk.Button(nudge_frame, text="▲", width=3, font=('', 8))
            up_btn.pack(side=tk.LEFT, padx=1)
            up_btn.bind('<ButtonPress-1>',
                        lambda e, idx=i: self._start_continuous_harmonic(idx, 'up'))
            up_btn.bind('<ButtonRelease-1>',
                        lambda e, idx=i: self._stop_continuous_harmonic(idx))

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # STATISTICS
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Statistics:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.stats_label = ttk.Label(self.control_panel, text="No annotations",
                                     justify=tk.LEFT, font=('', 8))
        self.stats_label.pack(fill=tk.X, pady=2)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # ANNOTATIONS TABLE
        # Scrollable per-contour summary: onset, offset, fmin, fmax
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Contours:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        header_frame = ttk.Frame(self.control_panel)
        header_frame.pack(fill=tk.X, pady=2)
        for label, width in [('#', 3), ('t_onset', 7), ('t_offset', 7),
                              ('f_min', 7), ('f_max', 7)]:
            ttk.Label(header_frame, text=label,
                      font=('', 8, 'bold'), width=width).pack(side=tk.LEFT)

        table_frame = ttk.Frame(self.control_panel, height=150)
        table_frame.pack(fill=tk.X, pady=2)
        table_frame.pack_propagate(False)

        ann_canvas    = tk.Canvas(table_frame, highlightthickness=0)
        ann_scrollbar = ttk.Scrollbar(table_frame, orient="vertical",
                                      command=ann_canvas.yview)
        self.annotations_inner_frame = ttk.Frame(ann_canvas)

        ann_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ann_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ann_canvas.configure(yscrollcommand=ann_scrollbar.set)
        ann_canvas.create_window((0, 0), window=self.annotations_inner_frame,
                                 anchor="nw")
        self.annotations_inner_frame.bind(
            "<Configure>",
            lambda e: ann_canvas.configure(
                scrollregion=ann_canvas.bbox("all")))

        def on_ann_mousewheel(event):
            ann_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        ann_canvas.bind("<Enter>",
                        lambda e: ann_canvas.bind_all(
                            "<MouseWheel>", on_ann_mousewheel))
        ann_canvas.bind("<Leave>",
                        lambda e: ann_canvas.unbind_all("<MouseWheel>"))

    ##    <(''<)  <( ' ' )>  (>'')>
    # ANNOTATION MODE
    ##    <(''<)  <( ' ' )>  (>'')>

    def switch_annotation_mode(self):
        """Switch between syllable and sequence annotation modes."""
        mode = self.annotation_mode.get()
        if mode == 'sequence':
            self.sequence_display_frame.pack(fill=tk.X, pady=5)
            self.mode_instructions.config(
                text="Sequence Mode: Define contour boundaries")
        else:
            self.sequence_display_frame.pack_forget()
            self.mode_instructions.config(
                text="Syllable Mode: Click points → Finish Contour")
        self._update_sequence_display()

    ##    <(''<)  <( ' ' )>  (>'')>
    # FINISH CONTOUR
    # Merged implementation from ChangepointLayer (document 6) and
    # the second finish_contour definition in ChangepointAnnotator (document 7).
    # - Requires minimum 2 points (from document 7 second definition)
    # - Sorts points by time (from both)
    # - Appends as dict with onset_idx/offset_idx (from document 6)
    # - silent parameter suppresses dialogs for programmatic calls (document 7)
    # - Returns bool for programmatic callers (document 7)
    # The first definition in document 7 is discarded.
    ##    <(''<)  <( ' ' )>  (>'')>

    def finish_contour(self, silent=False):
        """Mark current contour as complete and start a new one.

        Sorts current_contour points by time, assigns onset_idx=0 and
        offset_idx=len-1, and appends the dict to self.contours.

        Args:
            silent: bool — if True, suppresses warning dialogs and print output.
                    Used by auto-finish on navigation.

        Returns:
            bool — True if contour was finished, False if not enough points.
        """
        if len(self.current_contour) < 2:
            if not silent:
                messagebox.showwarning(
                    "Need More Points",
                    "Need at least 2 points to finish contour")
            return False

        sorted_points = sorted(self.current_contour, key=lambda x: x['time'])

        self.contours.append({
            'points':     sorted_points,
            'onset_idx':  0,
            'offset_idx': len(sorted_points) - 1,
        })

        self.current_contour = []
        self.changes_made    = True

        self.rebuild_annotations()
        self.update_display(recompute_spec=False)

        if not silent:
            logger.info("Contour complete (%d total)", len(self.contours))

        return True

    ##    <(''<)  <( ' ' )>  (>'')>
    # ANNOTATION REBUILD
    # Derives the flat annotations list from contours + current_contour.
    # This is the source of truth for display and stats.
    ##    <(''<)  <( ' ' )>  (>'')>

    def rebuild_annotations(self):
        """Rebuild flat annotations list from contours and current_contour.

        Handles both new dict format (with onset_idx/offset_idx) and
        old list format (converted inline — onset=first, offset=last by time).
        Updates contour_info label, stats, sequence display, and table.
        """
        self.annotations = []

        for contour in self.contours:
            if isinstance(contour, dict):
                points     = contour['points']
                onset_idx  = contour.get('onset_idx', 0)
                offset_idx = contour.get('offset_idx', len(points) - 1)

                for i, point in enumerate(points):
                    if i == onset_idx:
                        label = 'onset'
                    elif i == offset_idx:
                        label = 'offset'
                    else:
                        label = 'changepoint'

                    self.annotations.append({
                        'time':  point['time'],
                        'freq':  point['freq'],
                        'label': label,
                    })
            else:
                # (つ -' _ '- )つ    (つ -' _ '- )つ
                # FALLBACK: old list format — convert inline
                # onset = first point by time, offset = last point by time
                # This handles files saved before the dict format was adopted.
                # TODO: flag as fallback for tracking legacy file conversion
                # (つ -' _ '- )つ    (つ -' _ '- )つ
                sorted_points = sorted(contour, key=lambda x: x['time'])
                for i, point in enumerate(sorted_points):
                    label = ('onset' if i == 0
                             else 'offset' if i == len(sorted_points) - 1
                             else 'changepoint')
                    self.annotations.append({
                        'time':  point['time'],
                        'freq':  point['freq'],
                        'label': label,
                    })

        # Add current (unsaved) contour points as changepoints
        for point in self.current_contour:
            self.annotations.append({
                'time':  point['time'],
                'freq':  point['freq'],
                'label': 'changepoint',
            })

        # Update info label
        unsaved     = len(self.current_contour)
        saved_pts   = sum(
            len(c['points'] if isinstance(c, dict) else c)
            for c in self.contours)
        self.contour_info.config(
            text=(f"Unsaved Points: {unsaved} | "
                  f"Saved Points: {saved_pts} | "
                  f"Contours: {len(self.contours)} | "
                  f"Total: {self.total_contours_across_files}"))

        self._update_stats()
        self._update_annotations_table()
        self._update_sequence_display()

    ##    <(''<)  <( ' ' )>  (>'')>
    # MOUSE INTERACTION HOOKS
    ##    <(''<)  <( ' ' )>  (>'')>

    def on_custom_press(self, event):
        """Handle Ctrl+Click for lasso or endpoint marking.

        Ctrl+Click near existing point → endpoint marking mode.
        Ctrl+Click elsewhere → start lasso.
        """
        if event.button != 1:
            return False

        is_ctrl = self._get_ctrl_state()

        if is_ctrl and (self.current_contour or self.contours):
            near = self._find_nearest_point(
                event.xdata, event.ydata, 0.02, 100)

            if near:
                # Near existing point — endpoint marking mode
                self.drag_start = (event.xdata, event.ydata)
                return True
            else:
                # Start lasso
                self.lasso_mode   = True
                self.lasso_points = [(event.xdata, event.ydata)]
                self.drag_start   = (event.xdata, event.ydata)

                if self.drag_rect is not None:
                    self.drag_rect.remove()
                    self.drag_rect = None

                return True

        return False

    def on_custom_motion(self, event):
        """Draw lasso path during Ctrl+drag."""
        if self.lasso_mode and self.drag_start is not None:
            self.lasso_points.append((event.xdata, event.ydata))
            self._draw_lasso_preview()
            return True
        return False

    def on_custom_release(self, event):
        """Handle lasso finish or point add/remove on click release."""
        if self.lasso_mode:
            if len(self.lasso_points) >= 2:
                self._finish_lasso_selection()
            else:
                self._cancel_lasso()
            return True

        if event.xdata is None or event.ydata is None:
            return False

        drag_dist = 0.0
        if self.drag_start:
            x0, y0    = self.drag_start
            drag_dist = np.sqrt(
                (event.xdata - x0) ** 2 + (event.ydata - y0) ** 2)

        if drag_dist < 0.05:
            is_ctrl = self._get_ctrl_state()

            if is_ctrl and (self.current_contour or self.contours):
                self._handle_ctrl_click(event.xdata, event.ydata)
                return True

            # Normal click — add or remove point
            if self._remove_nearby_annotation(event.xdata, event.ydata):
                logger.debug("Removed point")
            else:
                self.current_contour.append({
                    'time': float(event.xdata),
                    'freq': float(event.ydata),
                })
                self.changes_made = True
                self.rebuild_annotations()
                self.update_display(recompute_spec=False)
                logger.debug("Added point: t=%.3f f=%.0f",
                             event.xdata, event.ydata)
            return True

        return False

    ##    <(''<)  <( ' ' )>  (>'')>
    # CTRL+CLICK ENDPOINT MARKING
    # First Ctrl+Click marks a point. Second Ctrl+Click extracts all points
    # between the two clicked times into a new contour.
    ##    <(''<)  <( ' ' )>  (>'')>

    def _handle_ctrl_click(self, x, y):
        """Handle Ctrl+Click for dual-endpoint contour extraction."""
        if self.pending_onset_idx is None:
            info = self._find_point_info(
                x, y,
                CONFIG["changepoint_ctrl_time_thresh_s"],
                CONFIG["changepoint_ctrl_freq_thresh_hz"]
            )
            if info is None:
                return
            self.pending_onset_idx = info
            logger.debug("First endpoint marked: t=%.3f", info['point']['time'])
        else:
            second_info = self._find_point_info(x, y, 0.05, 200)
            if second_info is None:
                self.pending_onset_idx = None
                return

            first_info  = self.pending_onset_idx
            onset_time  = min(first_info['point']['time'],
                              second_info['point']['time'])
            offset_time = max(first_info['point']['time'],
                              second_info['point']['time'])

            self._extract_contour_by_time_range(onset_time, offset_time)
            self.pending_onset_idx = None

    def _extract_contour_by_time_range(self, onset_time, offset_time):
        """Extract all points between onset_time and offset_time into a new contour.

        Removes extracted points from their source contours or current_contour.
        Creates a new dict-format contour and appends to self.contours.
        """
        extracted  = []
        source_map = []

        for i, point in enumerate(self.current_contour):
            if onset_time <= point['time'] <= offset_time:
                extracted.append(point)
                source_map.append(('current', i))

        for ci, contour in enumerate(self.contours):
            points = contour['points'] if isinstance(contour, dict) else contour
            for pi, point in enumerate(points):
                if onset_time <= point['time'] <= offset_time:
                    extracted.append(point)
                    source_map.append(('contour', ci, pi))

        if len(extracted) < 2:
            logger.debug("Not enough points in time range for extraction")
            return

        # Sort extracted by time
        sorted_pairs  = sorted(zip(extracted, source_map),
                                key=lambda x: x[0]['time'])
        extracted     = [p for p, _ in sorted_pairs]
        source_map    = [s for _, s in sorted_pairs]

        # Remove extracted points from sources
        current_remove = {s[1] for s in source_map if s[0] == 'current'}
        self.current_contour = [
            p for i, p in enumerate(self.current_contour)
            if i not in current_remove]

        contour_remove = {}
        for s in source_map:
            if s[0] == 'contour':
                contour_remove.setdefault(s[1], set()).add(s[2])

        for ci in sorted(contour_remove.keys(), reverse=True):
            contour = self.contours[ci]
            points  = contour['points'] if isinstance(contour, dict) else contour
            remaining = [p for pi, p in enumerate(points)
                         if pi not in contour_remove[ci]]
            if len(remaining) < 2:
                self.contours.pop(ci)
            else:
                sorted_rem = sorted(range(len(remaining)),
                                    key=lambda i: remaining[i]['time'])
                self.contours[ci] = {
                    'points':     remaining,
                    'onset_idx':  sorted_rem[0],
                    'offset_idx': sorted_rem[-1],
                }

        # Append new contour
        self.contours.append({
            'points':     extracted,
            'onset_idx':  0,
            'offset_idx': len(extracted) - 1,
        })

        self.changes_made = True
        self.rebuild_annotations()
        self.update_display(recompute_spec=False)

    ##    <(''<)  <( ' ' )>  (>'')>
    # LASSO SELECTION
    ##    <(''<)  <( ' ' )>  (>'')>

    def _draw_lasso_preview(self):
        """Draw lasso path as a yellow line during drag."""
        for line in self.lasso_lines:
            try:
                line.remove()
            except Exception:
                pass
        self.lasso_lines = []

        if len(self.lasso_points) < 2:
            return

        xs   = [p[0] for p in self.lasso_points]
        ys   = [p[1] for p in self.lasso_points]
        line = self.ax.plot(xs, ys, 'y-', linewidth=2, alpha=0.7)[0]
        self.lasso_lines.append(line)

        circle = self.ax.plot(
            self.lasso_points[0][0], self.lasso_points[0][1],
            'yo', markersize=10,
            markeredgecolor='black', markeredgewidth=2)[0]
        self.lasso_lines.append(circle)

        self.canvas.draw_idle()

    def _finish_lasso_selection(self):
        """Extract all points inside the lasso polygon into a new contour."""
        try:
            if len(self.lasso_points) < 3:
                self._cancel_lasso()
                return

            extracted  = []
            source_map = []

            for i, point in enumerate(self.current_contour):
                if self._point_in_polygon(
                        point['time'], point['freq'], self.lasso_points):
                    extracted.append(point)
                    source_map.append(('current', i))

            for ci, contour in enumerate(self.contours):
                points = (contour['points']
                          if isinstance(contour, dict) else contour)
                for pi, point in enumerate(points):
                    if self._point_in_polygon(
                            point['time'], point['freq'], self.lasso_points):
                        extracted.append(point)
                        source_map.append(('contour', ci, pi))

            if len(extracted) < 2:
                self._cancel_lasso()
                return

            sorted_pairs = sorted(zip(extracted, source_map),
                                   key=lambda x: x[0]['time'])
            extracted    = [p for p, _ in sorted_pairs]
            source_map   = [s for _, s in sorted_pairs]

            current_remove = {s[1] for s in source_map if s[0] == 'current'}
            self.current_contour = [
                p for i, p in enumerate(self.current_contour)
                if i not in current_remove]

            contour_remove = {}
            for s in source_map:
                if s[0] == 'contour':
                    contour_remove.setdefault(s[1], set()).add(s[2])

            for ci in sorted(contour_remove.keys(), reverse=True):
                contour   = self.contours[ci]
                points    = (contour['points']
                             if isinstance(contour, dict) else contour)
                remaining = [p for pi, p in enumerate(points)
                             if pi not in contour_remove[ci]]
                if len(remaining) < 2:
                    self.contours.pop(ci)
                else:
                    sorted_rem = sorted(range(len(remaining)),
                                        key=lambda i: remaining[i]['time'])
                    self.contours[ci] = {
                        'points':     remaining,
                        'onset_idx':  sorted_rem[0],
                        'offset_idx': sorted_rem[-1],
                    }

            self.contours.append({
                'points':     extracted,
                'onset_idx':  0,
                'offset_idx': len(extracted) - 1,
            })

            self.changes_made = True
            self._cancel_lasso()
            self.rebuild_annotations()
            self.update_display(recompute_spec=False)

        except Exception as e:
            logger.error("ERROR in lasso selection: %s", e)
            logger.debug(traceback.format_exc())
            self._cancel_lasso()

    def _cancel_lasso(self):
        """Cancel lasso and clean up visual feedback."""
        self.lasso_mode   = False
        self.lasso_points = []
        for line in self.lasso_lines:
            try:
                line.remove()
            except Exception:
                pass
        self.lasso_lines = []
        self.canvas.draw_idle()

    def _point_in_polygon(self, x, y, polygon):
        """Ray casting algorithm for point-in-polygon test."""
        n      = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = ((y - p1y) * (p2x - p1x)
                                       / (p2y - p1y) + p1x)
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    ##    <(''<)  <( ' ' )>  (>'')>
    # POINT LOOKUP HELPERS
    ##    <(''<)  <( ' ' )>  (>'')>

    def _find_nearest_point(self, x, y, time_thresh, freq_thresh):
        """Return first point within thresholds, or None."""
        for point in self.current_contour:
            if (abs(point['time'] - x) < time_thresh and
                    abs(point['freq'] - y) < freq_thresh):
                return point
        for contour in self.contours:
            points = contour['points'] if isinstance(contour, dict) else contour
            for point in points:
                if (abs(point['time'] - x) < time_thresh and
                        abs(point['freq'] - y) < freq_thresh):
                    return point
        return None

    def _find_point_info(self, x, y, time_thresh, freq_thresh):
        """Return dict with point and source info for Ctrl+Click matching."""
        for i, point in enumerate(self.current_contour):
            if (abs(point['time'] - x) < time_thresh and
                    abs(point['freq'] - y) < freq_thresh):
                return {'point': point, 'contour_idx': -1, 'point_idx': i}
        for ci, contour in enumerate(self.contours):
            points = contour['points'] if isinstance(contour, dict) else contour
            for pi, point in enumerate(points):
                if (abs(point['time'] - x) < time_thresh and
                        abs(point['freq'] - y) < freq_thresh):
                    return {'point': point, 'contour_idx': ci, 'point_idx': pi}
        return None

    def _remove_nearby_annotation(self, x, y):
        """Remove the closest point within 50ms / 100Hz threshold.

        Returns True if a point was removed.
        """
        time_thresh = CONFIG["changepoint_time_thresh_s"]
        freq_thresh = CONFIG["changepoint_freq_thresh_hz"]
        min_dist    = float('inf')
        closest     = None
        source      = None

        for i, point in enumerate(self.current_contour):
            dist = np.sqrt(
                ((point['time'] - x) / time_thresh) ** 2 +
                ((point['freq'] - y) / freq_thresh) ** 2)
            if dist < 1.0 and dist < min_dist:
                min_dist = dist
                closest  = i
                source   = 'current'

        for ci, contour in enumerate(self.contours):
            points = contour['points'] if isinstance(contour, dict) else contour
            for pi, point in enumerate(points):
                dist = np.sqrt(
                    ((point['time'] - x) / time_thresh) ** 2 +
                    ((point['freq'] - y) / freq_thresh) ** 2)
                if dist < 1.0 and dist < min_dist:
                    min_dist = dist
                    closest  = (ci, pi)
                    source   = 'contour'

        if source == 'current':
            self.current_contour.pop(closest)
            self.changes_made = True
            self.rebuild_annotations()
            self.update_display(recompute_spec=False)
            return True

        if source == 'contour':
            ci, pi   = closest
            contour  = self.contours[ci]
            points   = (contour['points']
                        if isinstance(contour, dict) else contour)
            points.pop(pi)
            if len(points) < 2:
                self.contours.pop(ci)
            else:
                if isinstance(contour, dict):
                    onset_idx  = contour.get('onset_idx', 0)
                    offset_idx = contour.get('offset_idx', len(points))
                    contour['onset_idx']  = max(0, onset_idx  - (1 if pi <= onset_idx  else 0))
                    contour['offset_idx'] = max(0, offset_idx - (1 if pi <= offset_idx else 0))
            self.changes_made = True
            self.rebuild_annotations()
            self.update_display(recompute_spec=False)
            return True

        return False

    ##    <(''<)  <( ' ' )>  (>'')>
    # CLEAR ACTIONS
    ##    <(''<)  <( ' ' )>  (>'')>

    def clear_last(self):
        """Remove last point from current_contour, or restore last finished contour."""
        if self.current_contour:
            self.current_contour.pop()
            self.changes_made = True
            self.rebuild_annotations()
            self.update_display(recompute_spec=False)
        elif self.contours:
            last = self.contours.pop()
            self.current_contour = (last['points'] if isinstance(last, dict)
                                    else last)
            self.changes_made = True
            self.rebuild_annotations()
            self.update_display(recompute_spec=False)

    def clear_all(self):
        """Clear all annotations after confirmation."""
        if (self.annotations or self.current_contour):
            if messagebox.askyesno("Clear", "Remove all annotations?"):
                self.annotations     = []
                self.current_contour = []
                self.contours        = []
                self.changes_made    = True
                self.update_display(recompute_spec=False)

    ##    <(''<)  <( ' ' )>  (>'')>
    # SKIP FILE
    ##    <(''<)  <( ' ' )>  (>'')>

    def skip_file(self):
        """Mark current file as skipped with reason dialog."""
        if not self.audio_files:
            return

        dialog     = tk.Toplevel(self.root)
        dialog.title("Skip File")
        dialog.geometry("300x150")

        ttk.Label(dialog, text="Reason for skipping:",
                  font=('', 10, 'bold')).pack(pady=10)

        reason_var = tk.StringVar(value="Noisy")
        for reason in ["Noisy", "Truncated", "Other"]:
            ttk.Radiobutton(dialog, text=reason,
                            variable=reason_var, value=reason).pack(
                                anchor=tk.W, padx=20)

        result = {'confirmed': False, 'reason': None}

        def on_ok():
            result['confirmed'] = True
            result['reason']    = reason_var.get()
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="OK",     command=on_ok).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(
            side=tk.LEFT, padx=5)

        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)

        if not result['confirmed']:
            return

        path = resolve_annotation_path(
            self.audio_files[self.current_file_idx],
            self.base_audio_dir,
            self.annotation_dir,
            SUFFIX_CHANGEPOINTS
        )
        mark_skip(path, result['reason'], self)
        self._count_skipped_files()
        self.next_file()

    ##    <(''<)  <( ' ' )>  (>'')>
    # FIND NEXT SKIPPED
    ##    <(''<)  <( ' ' )>  (>'')>

    def find_next_skipped(self):
        """Jump to the next file that is skipped or has no annotations."""
        if not self.audio_files:
            messagebox.showinfo("No Audio", "Load audio files first")
            return

        start_idx = (self.current_file_idx + 1) % len(self.audio_files)

        for i in range(len(self.audio_files)):
            check_idx = (start_idx + i) % len(self.audio_files)
            path      = resolve_annotation_path(
                self.audio_files[check_idx],
                self.base_audio_dir,
                self.annotation_dir,
                SUFFIX_CHANGEPOINTS
            )

            needs_annotation = False

            if not path.exists():
                needs_annotation = True
            else:
                data = annotation_io.load_annotation_file(path)
                if data.get('skip', False):
                    needs_annotation = True
                elif not data.get('contours'):
                    needs_annotation = True

            if needs_annotation:
                if self.changes_made:
                    self.save_custom_data()
                self.current_file_idx = check_idx
                self.load_current_file()
                return

        messagebox.showinfo("All Annotated",
                            "No skipped or unannotated files found.")

    ##    <(''<)  <( ' ' )>  (>'')>
    # OVERLAYS
    ##    <(''<)  <( ' ' )>  (>'')>

    def draw_custom_overlays(self):
        """Draw annotation points, guide lines, and bounding boxes."""

        # Annotation points colored by label
        for ann in self.annotations:
            self.ax.scatter(
                ann['time'], ann['freq'],
                c=_LABEL_COLORS[ann['label']],
                marker='.', s=100, linewidths=1, zorder=10
            )

        # Guide lines and text
        if self.annotations and (self.show_time_guides.get() or
                                  self.show_freq_guides.get()):
            times = [a['time'] for a in self.annotations]
            freqs = [a['freq'] for a in self.annotations]

            if self.show_time_guides.get():
                for t in times:
                    self.ax.axvline(x=t, color='lime', linestyle='--',
                                    linewidth=1.5, alpha=0.5)
                    if not self.hide_text.get():
                        self.ax.text(
                            t, self.ax.get_ylim()[1] * 0.95,
                            f"{t:.3f}s", color='lime', fontsize=9,
                            rotation=90, va='top', ha='right',
                            family='monospace', alpha=0.9)

            if self.show_freq_guides.get():
                for f in freqs:
                    self.ax.axhline(y=f, color='yellow', linestyle='--',
                                    linewidth=1.5, alpha=0.5)
                    if not self.hide_text.get():
                        self.ax.text(
                            self.ax.get_xlim()[0] + 0.01, f,
                            f"{f:.1f}Hz", color='yellow', fontsize=9,
                            va='center', ha='left',
                            family='monospace', alpha=0.9)

        # Bounding box and harmonic overlays
        if self.show_bounding_box.get() and self.annotations:
            times  = [a['time'] for a in self.annotations]
            freqs  = [a['freq'] for a in self.annotations]
            t_min, t_max = min(times), max(times)
            f_min, f_max = min(freqs), max(freqs)

            self._draw_shape(t_min, t_max, f_min, f_max,
                             color='white', linewidth=2.5, alpha=0.8)

            for harmonic in self.harmonics:
                if harmonic['show'].get():
                    mult    = harmonic['multiplier'].get()
                    h_f_min = f_min * mult
                    h_f_max = f_max * mult
                    self._draw_shape(t_min, t_max, h_f_min, h_f_max,
                                     color=harmonic['color'],
                                     linewidth=2, alpha=0.6)

        # Shared point annotations
        from yaaat.core.visualization import draw_shared_point_annotations
        draw_shared_point_annotations(self)

    def _draw_shape(self, t_min, t_max, f_min, f_max,
                    color, linewidth, alpha):
        """Draw rectangle, ellipse, or polygon bounding shape."""
        shape_type = self.bounding_box_shape.get()

        if shape_type == 'rectangle':
            self.ax.add_patch(plt.Rectangle(
                (t_min, f_min), t_max - t_min, f_max - f_min,
                fill=False, edgecolor=color,
                linewidth=linewidth, alpha=alpha, zorder=11))

        elif shape_type == 'ellipse':
            from matplotlib.patches import Ellipse
            self.ax.add_patch(Ellipse(
                ((t_min + t_max) / 2, (f_min + f_max) / 2),
                t_max - t_min, f_max - f_min,
                fill=False, edgecolor=color,
                linewidth=linewidth, alpha=alpha, zorder=11))

        elif shape_type == 'polygon':
            from matplotlib.patches import Polygon
            pts = [(a['time'], a['freq']) for a in self.annotations]
            self.ax.add_patch(Polygon(
                pts, closed=True,
                fill=False, edgecolor=color,
                linewidth=linewidth, alpha=alpha, zorder=11))

    ##    <(''<)  <( ' ' )>  (>'')>
    # GUIDE TOGGLE HELPERS
    ##    <(''<)  <( ' ' )>  (>'')>

    def _toggle_guides(self):
        """Redraw when any guide toggle changes."""
        if self.y is not None:
            self.update_display(recompute_spec=False)

    def _toggle_show_all(self):
        """Sync individual guide checkboxes to Show All state."""
        state = self.show_all_guides_var.get()
        self.show_time_guides.set(state)
        self.show_freq_guides.set(state)
        self._toggle_guides()

    ##    <(''<)  <( ' ' )>  (>'')>
    # HARMONIC NUDGE CONTROLS
    ##    <(''<)  <( ' ' )>  (>'')>

    def _nudge_harmonic(self, harmonic_index, direction):
        """Nudge harmonic multiplier by 0.01."""
        harmonic = self.harmonics[harmonic_index]
        current  = harmonic['multiplier'].get()
        if direction == 'up':
            harmonic['multiplier'].set(current + 0.01)
        elif direction == 'down' and current > 0.02:
            harmonic['multiplier'].set(current - 0.01)
        if harmonic['label']:
            harmonic['label'].config(
                text=f"{harmonic['multiplier'].get():.2f}x")
        if harmonic['show'].get():
            self._toggle_guides()

    def _start_continuous_harmonic(self, harmonic_index, direction):
        """Begin hold-to-repeat harmonic nudge."""
        self._nudge_harmonic(harmonic_index, direction)
        self.harmonic_repeat_ids[harmonic_index] = self.root.after(
            200, self._continue_harmonic, harmonic_index, direction)

    def _continue_harmonic(self, harmonic_index, direction):
        """Continue hold-to-repeat harmonic nudge at 50ms intervals."""
        self._nudge_harmonic(harmonic_index, direction)
        self.harmonic_repeat_ids[harmonic_index] = self.root.after(
            50, self._continue_harmonic, harmonic_index, direction)

    def _stop_continuous_harmonic(self, harmonic_index):
        """Cancel hold-to-repeat harmonic nudge."""
        if harmonic_index in self.harmonic_repeat_ids:
            self.root.after_cancel(self.harmonic_repeat_ids[harmonic_index])
            del self.harmonic_repeat_ids[harmonic_index]

    ##    <(''<)  <( ' ' )>  (>'')>
    # STATISTICS AND TABLE
    ##    <(''<)  <( ' ' )>  (>'')>

    def _update_stats(self):
        """Update statistics label with onset/offset timing and frequency range."""
        if not self.annotations:
            self.stats_label.config(text="No annotations")
            return

        onset  = next((a for a in self.annotations if a['label'] == 'onset'),  None)
        offset = next((a for a in self.annotations if a['label'] == 'offset'), None)

        if onset and offset:
            duration    = offset['time'] - onset['time']
            freqs       = [a['freq'] for a in self.annotations]
            delta_freq  = max(freqs) - min(freqs)
            self.stats_label.config(
                text=(f"Duration: {duration:.3f}s | "
                      f"ΔFreq: {delta_freq:.1f} Hz"))
        else:
            self.stats_label.config(text="Incomplete annotation")

    def _update_annotations_table(self):
        """Refresh the scrollable contour summary table."""
        for widget in self.annotations_inner_frame.winfo_children():
            widget.destroy()

        for idx, contour in enumerate(self.contours):
            points = contour['points'] if isinstance(contour, dict) else contour
            if len(points) < 1:
                continue

            sorted_pts = sorted(points, key=lambda x: x['time'])
            t_onset    = sorted_pts[0]['time']
            t_offset   = sorted_pts[-1]['time']
            all_freqs  = [p['freq'] for p in points]
            f_min      = min(all_freqs)
            f_max      = max(all_freqs)

            row = ttk.Frame(self.annotations_inner_frame)
            row.pack(fill=tk.X, pady=1)
            for text, width in [
                (f"{idx+1}",         3),
                (f"{t_onset:.4f}",   7),
                (f"{t_offset:.4f}",  7),
                (f"{f_min:.0f}",     7),
                (f"{f_max:.0f}",     7),
            ]:
                ttk.Label(row, text=text,
                          font=('', 8), width=width).pack(side=tk.LEFT)

        if self.current_contour:
            unsaved_frame = ttk.Frame(self.annotations_inner_frame)
            unsaved_frame.pack(fill=tk.X, pady=3)
            ttk.Label(
                unsaved_frame,
                text=f"→ Current: {len(self.current_contour)} pts (unsaved)",
                font=('', 8, 'italic'), foreground='orange'
            ).pack(padx=5)

    def _update_sequence_display(self):
        """Update sequence mode contour list."""
        for widget in self.sequence_inner_frame.winfo_children():
            widget.destroy()

        if self.annotation_mode.get() != 'sequence':
            return

        for i, contour in enumerate(self.contours):
            points = contour['points'] if isinstance(contour, dict) else contour
            if len(points) < 2:
                continue

            onset_idx  = contour.get('onset_idx', 0) if isinstance(contour, dict) else 0
            offset_idx = contour.get('offset_idx', len(points) - 1) if isinstance(contour, dict) else len(points) - 1

            onset  = points[onset_idx]
            offset = points[offset_idx]
            freqs  = [p['freq'] for p in points]

            row = ttk.Frame(self.sequence_inner_frame)
            row.pack(fill=tk.X, pady=1)
            for text, width in [
                (f"{i+1}",                     3),
                (f"{onset['time']:.3f}",        7),
                (f"{offset['time']:.3f}",       7),
                (f"{min(freqs):.0f}",           7),
                (f"{max(freqs):.0f}",           7),
                (f"{offset['time']-onset['time']:.3f}", 6),
                (f"{len(points)}",              4),
            ]:
                ttk.Label(row, text=text,
                          font=('', 8), width=width).pack(side=tk.LEFT)

    ##    <(''<)  <( ' ' )>  (>'')>
    # DATASET-LEVEL COUNTS
    ##    <(''<)  <( ' ' )>  (>'')>

    def _count_total_contours(self):
        """Count total contours across all annotation files in the dataset."""
        self.total_contours_across_files = 0
        if self.annotation_dir is None:
            return

        for audio_file in self.audio_files:
            path = resolve_annotation_path(
                audio_file, self.base_audio_dir,
                self.annotation_dir, SUFFIX_CHANGEPOINTS)
            if path.exists():
                data = annotation_io.load_annotation_file(path)
                self.total_contours_across_files += len(
                    data.get('contours', []))

    def _count_skipped_files(self):
        """Count total skipped files across the dataset."""
        self.total_skipped_files = 0
        if self.annotation_dir is None:
            return

        for audio_file in self.audio_files:
            path = resolve_annotation_path(
                audio_file, self.base_audio_dir,
                self.annotation_dir, SUFFIX_CHANGEPOINTS)
            if path.exists() and is_skipped(path):
                self.total_skipped_files += 1

    ##    <(''<)  <( ' ' )>  (>'')>
    # KEY STATE HELPER
    ##    <(''<)  <( ' ' )>  (>'')>

    def _get_ctrl_state(self):
        """Detect Ctrl key state. Windows-native via ctypes; fallback for others."""
        if sys.platform == 'win32':
            import ctypes
            return bool(ctypes.windll.user32.GetKeyState(0x11) & 0x8000)
        return False

    ##    <(''<)  <( ' ' )>  (>'')>
    # SAVE / LOAD
    ##    <(''<)  <( ' ' )>  (>'')>

    def save_custom_data(self):
        """Save changepoint annotations to _changepoints.json via merge-write.

        _changepoints.json is the canonical geometry file for the shared
        annotation schema. contour_metrics are computed at save time from
        the current contour geometry — not stored during annotation.
        contour IDs are assigned here as 'c{idx}'.
        """
        if not self.audio_files or self.annotation_dir is None:
            return

        path = resolve_annotation_path(
            self.audio_files[self.current_file_idx],
            self.base_audio_dir,
            self.annotation_dir,
            SUFFIX_CHANGEPOINTS
        )

        # Assign contour IDs at save time
        contours_with_ids = []
        for idx, contour in enumerate(self.contours):
            c = dict(contour) if isinstance(contour, dict) else {
                'points':     contour,
                'onset_idx':  0,
                'offset_idx': len(contour) - 1,
            }
            c['id'] = f"c{idx}"
            contours_with_ids.append(c)

        # Compute metrics at save time from current geometry
        metrics = compute_contour_metrics(contours_with_ids)

        # PSD params — changepoint annotator does not have its own PSD vars;
        # uses BaseLayer defaults via build_psd_params()
        tab_data = {
            "audio_file":      str(self.audio_files[self.current_file_idx]),
            "contours":        contours_with_ids,
            "contour_metrics": metrics,
            "spec_params":     build_spec_params(self, orientation="horizontal"),
            "psd_params":      build_psd_params(self),
            "skip":            False,
            "skip_reason":     "",
        }

        merge_and_save(path, tab_data)
        self.changes_made        = False
        self.file_was_annotated  = True
        self._count_total_contours()
        self.rebuild_annotations()
        self.update_display(recompute_spec=False)
        logger.info("Saved %d contours to %s",
                    len(self.contours), path.name)

    def load_custom_data(self):
        """Load changepoint annotations from _changepoints.json.

        Handles both new dict format (onset_idx/offset_idx) and old list
        format (converted inline — no separate conversion function needed).
        """
        if not self.audio_files or self.annotation_dir is None:
            return

        path = resolve_annotation_path(
            self.audio_files[self.current_file_idx],
            self.base_audio_dir,
            self.annotation_dir,
            SUFFIX_CHANGEPOINTS
        )

        self.annotations     = []
        self.current_contour = []
        self.contours        = []

        data = load_and_check_params(
            path, self, SUFFIX_CHANGEPOINTS, self.annotation_dir)

        if not data:
            self.file_was_annotated = False
            return

        raw_contours = data.get('contours', [])

        for raw in raw_contours:
            if isinstance(raw, dict) and 'points' in raw:
                # New dict format — use directly
                self.contours.append(raw)
            elif isinstance(raw, list) and len(raw) > 0:
                # (つ -' _ '- )つ    (つ -' _ '- )つ
                # FALLBACK: old list format — convert inline to dict format
                # onset_idx=0 (first by time), offset_idx=len-1 (last by time)
                # TODO: flag as fallback for tracking legacy file conversion
                # (つ -' _ '- )つ    (つ -' _ '- )つ
                sorted_pts = sorted(raw, key=lambda x: x['time'])
                self.contours.append({
                    'points':     sorted_pts,
                    'onset_idx':  0,
                    'offset_idx': len(sorted_pts) - 1,
                })

        self.file_was_annotated = bool(self.contours)

        if self.contours:
            self.rebuild_annotations()
            logger.info("Loaded %d contours from %s",
                        len(self.contours), path.name)

    ##    <(''<)  <( ' ' )>  (>'')>
    # NAVIGATION OVERRIDES
    # Auto-finish current contour before navigating if >= 2 points exist.
    ##    <(''<)  <( ' ' )>  (>'')>

    def next_file(self):
        """Auto-finish contour if in progress, then navigate to next file."""
        if len(self.current_contour) >= 2:
            self.finish_contour(silent=True)
        super().next_file()

    def previous_file(self):
        """Auto-finish contour if in progress, then navigate to previous file."""
        if len(self.current_contour) >= 2:
            self.finish_contour(silent=True)
        super().previous_file()


##    <(''<)  <( ' ' )>  (>'')>

def main():
    """Launch ChangepointAnnotator as a standalone tab."""
    root = tk.Tk()
    app  = ChangepointAnnotator(root)
    root.geometry("1400x800")
    root.mainloop()


if __name__ == "__main__":
    main()

# U S A G I
# from yaaat.tabs.changepoint_annotator import ChangepointAnnotator
# root = tk.Tk(); app = ChangepointAnnotator(root); root.geometry("1400x800"); root.mainloop()