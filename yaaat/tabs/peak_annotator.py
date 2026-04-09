"""
tabs/peak_annotator.py

Peak annotator tab for YAAAT.
Inherits from BaseLayer — single-file view with vertical spectrogram orientation.

Responsibilities:
    - Dual-resolution display: vertical spectrogram (temporal) + PSD overlay (frequency)
    - Click-to-mark nearest PSD peak within threshold
    - Click-near-existing-peak to remove
    - Auto-detect all prominent peaks above prominence threshold
    - Guide lines (time/freq), bounding boxes, harmonic overlays with nudge
    - Skip file with reason dialog
    - Count total peaks across dataset
    - Save/load to _peaks.json via annotation_io merge-write

Display orientation:
    Vertical spectrogram — frequency on x-axis, time on y-axis.
    This differs from all other tabs which use horizontal orientation.
    orientation='vertical' is recorded in spec_params to distinguish this
    file from horizontal-orientation files on param divergence check.

Dual parameter sets:
    spec_params: n_fft_spect, hop_spect — spectrogram temporal resolution
    psd_params:  n_fft_psd, hop_psd    — PSD frequency resolution
    Both are checked independently on load for param divergence.
    Peak annotator is the only tab that owns both independently.

PSD overlay:
    Plotted as a normalized amplitude curve over the vertical spectrogram.
    Amplitude is scaled to time-axis units (multiplied by len(times)) so
    it sits in the same coordinate space as the spectrogram display.

Annotation file: {prefix}_{stem}_peaks.json
    Written via annotation_io.merge_and_save().
    carries contour_source key pointing to _changepoints.json for
    cross-tab reference resolution.
"""

import json
import logging
import traceback
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections
from scipy.signal import find_peaks

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

from yaaat.core.base_layer import BaseLayer
from yaaat.core import audio_utils
from yaaat.core import annotation_io
from yaaat.core.annotation_io import (
    SUFFIX_PEAKS,
    SUFFIX_CHANGEPOINTS,
    resolve_annotation_path,
    merge_and_save,
    load_and_check_params,
    build_spec_params,
    build_psd_params,
    mark_skip,
)

logger = logging.getLogger(__name__)


##    <(''<)  <( ' ' )>  (>'')>

class PeakAnnotator(BaseLayer):
    """Peak annotation tab — dual-resolution spectrogram+PSD display with
    click-to-mark peak annotation and auto-detection support.
    """

    def __init__(self, root):
        """Initialize peak-specific state before calling BaseLayer.__init__."""

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # DUAL-RESOLUTION SPECTROGRAM PARAMETERS
        # Peak annotator uses separate parameter sets for spectrogram and PSD.
        # These are distinct from BaseLayer's n_fft/hop_length which are shared
        # across tabs. Peak-specific vars are initialized here before super().__init__()
        # so they exist when setup_ui() calls setup_custom_controls().
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.n_fft_spect = tk.IntVar(value=512)
        self.hop_spect   = tk.IntVar(value=32)
        self.n_fft_psd   = tk.IntVar(value=1024)
        self.hop_psd     = tk.IntVar(value=512)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # PSD DATA
        # Computed alongside the vertical spectrogram in compute_dual_view().
        # pfreqs: frequency array for PSD
        # ppsd: normalized PSD values [0, 1]
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.pfreqs = None
        self.ppsd   = None

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # PEAK ANNOTATIONS
        # peak_annotations: list of dicts per marked peak
        #   {'freq': float, 'amplitude_normalized': float,
        #    'prominence': float, 'auto_detected': bool}
        # auto_detected_peaks: cached scipy peak indices for current PSD
        #   Invalidated when prominence slider changes.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.peak_annotations   = []
        self.auto_detected_peaks = None

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # DETECTION PARAMETERS
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.peak_prominence         = tk.DoubleVar(value=0.1)
        self.peak_click_threshold_hz = tk.DoubleVar(value=100.0)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # DISPLAY OPTIONS
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.show_auto_peaks = tk.BooleanVar(value=True)
        self.show_psd        = tk.BooleanVar(value=True)
        self.show_freq_guides = tk.BooleanVar(value=False)
        self.show_time_guides = tk.BooleanVar(value=False)
        self.show_all_guides_var = tk.BooleanVar(value=False)
        self.hide_text        = tk.BooleanVar(value=False)
        self.show_bounding_box = tk.BooleanVar(value=False)
        self.bounding_box_shape = tk.StringVar(value='rectangle')

        # Harmonic bounding box overlays — same structure as changepoint tab
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

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # TRACKING ACROSS FILES
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.total_peaks_across_files  = 0
        self.total_files_annotated     = 0

        super().__init__(root)

        if isinstance(root, tk.Tk):
            self.root.title("Peak Annotator - YAAAT")

    ##    <(''<)  <( ' ' )>  (>'')>
    # CUSTOM CONTROLS
    ##    <(''<)  <( ' ' )>  (>'')>

    def setup_custom_controls(self):
        """Build peak-specific controls in the scrollable control panel."""

        # Instructions
        ttk.Label(self.control_panel, text="Instructions:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(
            self.control_panel,
            text=(
                "• Click near peak: mark\n"
                "• Click near marked peak: remove\n"
                "• Drag: zoom to region\n"
                "• Right-click: undo zoom"
            ),
            wraplength=400, font=('', 8)
        ).pack(padx=5, pady=(0, 5))

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # DUAL SPECTROGRAM + PSD PARAMETER BUTTONS
        # Two side-by-side columns: spectrogram (left) and PSD (right).
        # Peak annotator is the only tab with independent spectrogram+PSD params.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        spec_psd_container = ttk.Frame(self.control_panel)
        spec_psd_container.pack(fill=tk.X, pady=3)

        # Left column — spectrogram params
        spec_col = ttk.Frame(spec_psd_container)
        spec_col.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        ttk.Label(spec_col, text="Spectrogram:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        ttk.Label(spec_col, text="n_fft:", font=('', 8)).pack(anchor=tk.W)
        nfft_spect_frame = ttk.Frame(spec_col)
        nfft_spect_frame.pack(fill=tk.X, pady=2)

        self.nfft_spect_buttons = []
        for nfft in [256, 512, 1024, 2048]:
            btn = tk.Button(nfft_spect_frame, text=str(nfft), width=4,
                            command=lambda n=nfft: self._change_nfft_spect(n))
            btn.pack(side=tk.LEFT, padx=2)
            self.nfft_spect_buttons.append((btn, nfft))

        ttk.Label(spec_col, text="hop:", font=('', 8)).pack(anchor=tk.W)
        hop_spect_frame = ttk.Frame(spec_col)
        hop_spect_frame.pack(fill=tk.X, pady=2)

        self.hop_spect_buttons = []
        for hop in [16, 32, 64, 128]:
            btn = tk.Button(hop_spect_frame, text=str(hop), width=4,
                            command=lambda h=hop: self._change_hop_spect(h))
            btn.pack(side=tk.LEFT, padx=2)
            self.hop_spect_buttons.append((btn, hop))

        ttk.Separator(spec_psd_container, orient=tk.VERTICAL).grid(
            row=0, column=1, sticky='ns', padx=4)

        # Right column — PSD params
        psd_col = ttk.Frame(spec_psd_container)
        psd_col.grid(row=0, column=2, sticky='nsew')

        ttk.Label(psd_col, text="PSD:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        ttk.Label(psd_col, text="n_fft:", font=('', 8)).pack(anchor=tk.W)
        nfft_psd_frame = ttk.Frame(psd_col)
        nfft_psd_frame.pack(fill=tk.X, pady=2)

        self.nfft_psd_buttons = []
        for nfft in [512, 1024, 2048, 4096]:
            btn = tk.Button(nfft_psd_frame, text=str(nfft), width=4,
                            command=lambda n=nfft: self._change_nfft_psd(n))
            btn.pack(side=tk.LEFT, padx=2)
            self.nfft_psd_buttons.append((btn, nfft))

        ttk.Label(psd_col, text="hop:", font=('', 8)).pack(anchor=tk.W)
        hop_psd_frame = ttk.Frame(psd_col)
        hop_psd_frame.pack(fill=tk.X, pady=2)

        self.hop_psd_buttons = []
        for hop in [256, 512, 1024]:
            btn = tk.Button(hop_psd_frame, text=str(hop), width=4,
                            command=lambda h=hop: self._change_hop_psd(h))
            btn.pack(side=tk.LEFT, padx=2)
            self.hop_psd_buttons.append((btn, hop))

        ttk.Checkbutton(psd_col, text="Show PSD",
                        variable=self.show_psd,
                        command=lambda: self.update_display(
                            recompute_spec=False)).pack(anchor=tk.W, pady=2)

        spec_psd_container.columnconfigure(0, weight=1)
        spec_psd_container.columnconfigure(2, weight=1)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # PEAK DETECTION PARAMETERS
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Peak Detection:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        prom_frame = ttk.Frame(self.control_panel)
        prom_frame.pack(fill=tk.X, pady=2)
        ttk.Label(prom_frame, text="Prominence:", font=('', 8)).pack(
            side=tk.LEFT)
        ttk.Scale(prom_frame, from_=0.01, to=0.5,
                  variable=self.peak_prominence, orient=tk.HORIZONTAL,
                  command=self._on_prominence_change).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.prom_label = ttk.Label(prom_frame,
                                    text=f"{self.peak_prominence.get():.2f}",
                                    font=('', 8), width=5)
        self.prom_label.pack(side=tk.LEFT)

        thresh_frame = ttk.Frame(self.control_panel)
        thresh_frame.pack(fill=tk.X, pady=2)
        ttk.Label(thresh_frame, text="Click threshold (Hz):",
                  font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(thresh_frame, textvariable=self.peak_click_threshold_hz,
                  width=6).pack(side=tk.LEFT, padx=2)

        ttk.Checkbutton(self.control_panel, text="Show Auto-Detected Peaks",
                        variable=self.show_auto_peaks,
                        command=lambda: self.update_display(
                            recompute_spec=False)).pack(anchor=tk.W, pady=2)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # GUIDES AND BOUNDING BOX
        # Same structure as changepoint tab — shared UI pattern.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Guides:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(3, 2))

        guides_grid = ttk.Frame(self.control_panel)
        guides_grid.pack(fill=tk.X, pady=2)

        ttk.Checkbutton(guides_grid, text="Time Lines",
                        variable=self.show_time_guides,
                        command=self._toggle_guides).grid(
                            row=0, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Freq Lines",
                        variable=self.show_freq_guides,
                        command=self._toggle_guides).grid(
                            row=1, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Show All",
                        variable=self.show_all_guides_var,
                        command=self._toggle_show_all).grid(
                            row=2, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Hide Text",
                        variable=self.hide_text,
                        command=self._toggle_guides).grid(
                            row=3, column=0, sticky=tk.W, padx=2, pady=2)

        ttk.Checkbutton(guides_grid, text="Bounding Box",
                        variable=self.show_bounding_box,
                        command=self._toggle_guides).grid(
                            row=0, column=1, sticky=tk.W, padx=2, pady=2)

        for col, (val, label) in enumerate(
                [('rectangle', 'Rectangle'),
                 ('polygon',   'Polygon'),
                 ('ellipse',   'Ellipse')], start=1):
            ttk.Radiobutton(guides_grid, text=label,
                            variable=self.bounding_box_shape, value=val,
                            command=self._toggle_guides).grid(
                                row=1, column=col, sticky=tk.W,
                                padx=2, pady=2)

        # Harmonic nudge controls
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
                          lambda e, idx=i: self._start_continuous_harmonic(
                              idx, 'down'))
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
                        lambda e, idx=i: self._start_continuous_harmonic(
                            idx, 'up'))
            up_btn.bind('<ButtonRelease-1>',
                        lambda e, idx=i: self._stop_continuous_harmonic(idx))

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # ACTIONS
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Actions:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        button_grid = ttk.Frame(self.control_panel)
        button_grid.pack(pady=2)

        actions = [
            ("Autofill Peaks",  self.auto_detect_peaks),
            ("Remove Peak",     self._clear_last_peak),
            ("Clear Peaks",     self._clear_all_peaks),
            ("Save Anno",       self.save_custom_data),
            ("Reset Display",   self._recompute_display),
            ("Debug Info",      self._print_debug_info),
        ]

        for i, (text, command) in enumerate(actions):
            ttk.Button(button_grid, text=text, command=command,
                       width=12).grid(
                row=i // 3, column=i % 3,
                padx=2, pady=2, sticky='ew')

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # PEAK INFO AND STATISTICS
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.peak_info = ttk.Label(
            self.control_panel,
            text="Peaks: 0 | Total: 0",
            font=('', 8), justify=tk.LEFT)
        self.peak_info.pack(pady=2)

        ttk.Label(self.control_panel, text="Statistics:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.stats_label = ttk.Label(self.control_panel,
                                     text="No peaks annotated",
                                     justify=tk.LEFT, font=('', 8))
        self.stats_label.pack(fill=tk.X, pady=2)

        self._update_spect_button_highlights()
        self._update_psd_button_highlights()

    ##    <(''<)  <( ' ' )>  (>'')>
    # AUDIO PROCESSING HOOK
    ##    <(''<)  <( ' ' )>  (>'')>

    def process_audio(self):
        """Compute dual-resolution view after audio is loaded."""
        self._compute_dual_view()

    ##    <(''<)  <( ' ' )>  (>'')>
    # DUAL-RESOLUTION COMPUTATION
    # Vertical spectrogram — temporal resolution for display
    # PSD — frequency resolution for peak detection and overlay
    ##    <(''<)  <( ' ' )>  (>'')>

    def _compute_dual_view(self):
        """Compute vertical spectrogram and PSD from current audio.

        Uses peak-annotator-specific params (n_fft_spect/hop_spect for
        spectrogram, n_fft_psd/hop_psd for PSD) rather than BaseLayer
        shared params.

        Vertical orientation: frequency on x-axis, time on y-axis.
        This is the display convention for the peak annotator — all other
        tabs use horizontal orientation.
        """
        if self.y is None:
            return

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # Vertical spectrogram — uses peak-specific n_fft_spect/hop_spect
        # orientation='vertical' rotates S_db so freq is on x-axis
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        self.S_db, self.freqs, self.times = audio_utils.compute_spectrogram_unified(
            self.y, self.sr,
            nfft=self.n_fft_spect.get(),
            hop=self.hop_spect.get(),
            fmin=0,
            fmax=self.sr / 2,
            scale='linear',
            orientation='vertical'
        )

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # PSD — uses peak-specific n_fft_psd/hop_psd
        # ppsd is normalized to [0, 1] — scaled to time units for overlay
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        self.pfreqs, self.ppsd = audio_utils.compute_psd(
            self.y, self.sr,
            nfft_psd=self.n_fft_psd.get(),
            hop_psd=self.hop_psd.get()
        )

        # Invalidate auto-detected peaks cache on recompute
        self.auto_detected_peaks = None

        logger.debug("Dual view: S_db=%s freqs=%d times=%d psd=%d",
                     self.S_db.shape, len(self.freqs),
                     len(self.times), len(self.pfreqs))

    ##    <(''<)  <( ' ' )>  (>'')>
    # SPECTROGRAM OVERRIDE
    # BaseLayer.compute_spectrogram() uses shared params — peak annotator
    # overrides to use its own dual-resolution params instead.
    ##    <(''<)  <( ' ' )>  (>'')>

    def compute_spectrogram(self):
        """Override BaseLayer.compute_spectrogram() to use dual-view computation."""
        self._compute_dual_view()

    ##    <(''<)  <( ' ' )>  (>'')>
    # PEAK DETECTION
    ##    <(''<)  <( ' ' )>  (>'')>

    def auto_detect_peaks(self):
        """Auto-detect all prominent PSD peaks and add to peak_annotations.

        Uses current prominence threshold. Skips peaks already annotated
        (within 1 Hz of existing annotation). Caches detected peak indices.
        """
        if self.ppsd is None:
            return

        mask        = self.pfreqs <= self.fmax_display.get()
        ppsd_scaled = self.ppsd[mask] * len(self.times)

        peak_indices, properties = find_peaks(
            ppsd_scaled, prominence=self.peak_prominence.get())

        self.auto_detected_peaks = peak_indices
        added = 0

        for idx in peak_indices:
            freq = float(self.pfreqs[mask][idx])

            # Skip if already annotated within 1 Hz
            if any(abs(p['freq'] - freq) < 1.0
                   for p in self.peak_annotations):
                continue

            self.peak_annotations.append({
                'freq':                 freq,
                'amplitude_normalized': float(self.ppsd[mask][idx]),
                'prominence':           float(
                    properties['prominences'][
                        list(peak_indices).index(idx)]),
                'auto_detected':        True,
            })
            added += 1

        self.changes_made = True
        self.update_display(recompute_spec=False)
        logger.info("Auto-detected %d peaks (%d added)", len(peak_indices), added)

    def _mark_nearest_peak(self, click_freq, click_y):
        """Find and mark the nearest PSD peak to the click frequency.

        Searches for prominence-filtered PSD peaks within the display range.
        Snaps to the nearest peak within peak_click_threshold_hz.
        Does not add duplicates within 1 Hz of an existing annotation.
        """
        if self.ppsd is None:
            return

        mask        = ((self.pfreqs >= self.fmin_display.get()) &
                       (self.pfreqs <= self.fmax_display.get()))
        ppsd_scaled = self.ppsd[mask] * len(self.times)
        pfreqs_masked = self.pfreqs[mask]

        peak_indices, properties = find_peaks(
            ppsd_scaled, prominence=self.peak_prominence.get())

        if len(peak_indices) == 0:
            logger.debug("No peaks found near click")
            return

        peak_freqs = pfreqs_masked[peak_indices]
        distances  = np.abs(peak_freqs - click_freq)
        nearest    = np.argmin(distances)

        if distances[nearest] > self.peak_click_threshold_hz.get():
            logger.debug("Nearest peak %.1f Hz away, threshold %.1f Hz",
                         distances[nearest],
                         self.peak_click_threshold_hz.get())
            return

        peak_idx_in_masked = peak_indices[nearest]
        freq               = float(pfreqs_masked[peak_idx_in_masked])
        prominence         = float(properties['prominences'][nearest])

        if any(abs(p['freq'] - freq) < 1.0 for p in self.peak_annotations):
            logger.debug("Peak at %.1f Hz already annotated", freq)
            return

        self.peak_annotations.append({
            'freq':                 freq,
            'amplitude_normalized': float(self.ppsd[mask][peak_idx_in_masked]),
            'prominence':           prominence,
            'auto_detected':        False,
        })

        self.changes_made = True
        self.update_display(recompute_spec=False)
        logger.info("Marked peak: %.1f Hz (prominence: %.3f)", freq, prominence)

    def _remove_nearby_peak(self, click_freq):
        """Remove the closest annotated peak within click threshold.

        Returns True if a peak was removed.
        """
        threshold = self.peak_click_threshold_hz.get()
        min_dist  = float('inf')
        closest   = None

        for i, peak in enumerate(self.peak_annotations):
            dist = abs(peak['freq'] - click_freq)
            if dist < threshold and dist < min_dist:
                min_dist = dist
                closest  = i

        if closest is not None:
            removed = self.peak_annotations.pop(closest)
            self.changes_made = True
            self.update_display(recompute_spec=False)
            logger.info("Removed peak at %.1f Hz", removed['freq'])
            return True

        return False

    ##    <(''<)  <( ' ' )>  (>'')>
    # MOUSE INTERACTION
    # Peak annotator uses on_custom_release for click-to-mark.
    # Drag is handled by BaseLayer zoom logic.
    ##    <(''<)  <( ' ' )>  (>'')>

    def on_custom_release(self, event):
        """Mark or remove peak on click release.

        Small drag (< 50 display units) treated as click.
        Checks for nearby existing peak first — removes if found.
        Otherwise snaps to nearest PSD peak within threshold.
        """
        if event.xdata is None or event.ydata is None:
            return False

        drag_dist = 0.0
        if self.drag_start:
            x0, y0    = self.drag_start
            drag_dist = np.sqrt(
                (event.xdata - x0) ** 2 + (event.ydata - y0) ** 2)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # Threshold of 50 in display units — peak annotator uses vertical
        # orientation so x-axis is frequency (Hz), y-axis is time units.
        # A tighter threshold than changepoint tab which uses normalized coords.
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        if drag_dist < 50:
            # Click on x-axis = frequency in Hz (vertical orientation)
            click_freq = event.xdata

            if self._remove_nearby_peak(click_freq):
                return True

            self._mark_nearest_peak(click_freq, event.ydata)
            return True

        return False

    ##    <(''<)  <( ' ' )>  (>'')>
    # OVERLAYS
    ##    <(''<)  <( ' ' )>  (>'')>

    def draw_custom_overlays(self):
        """Draw PSD overlay, auto-detected peaks, annotated peaks, guides, boxes."""

        if self.ppsd is None:
            return

        mask        = self.pfreqs <= self.fmax_display.get()
        ppsd_scaled = self.ppsd[mask] * len(self.times)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # PSD OVERLAY
        # Plotted as a red curve over the spectrogram.
        # x-axis = frequency (Hz), y-axis = scaled amplitude (time units).
        # Scaling by len(times) places the curve in spectrogram coordinate space.
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        if self.show_psd.get():
            self.ax.plot(self.pfreqs[mask], ppsd_scaled,
                         color='red', linewidth=1.5, alpha=0.8,
                         label='PSD', zorder=5)

            # Auto-detected peaks overlay
            if self.show_auto_peaks.get():
                if self.auto_detected_peaks is None:
                    self.auto_detected_peaks, _ = find_peaks(
                        ppsd_scaled,
                        prominence=self.peak_prominence.get())
                self.ax.scatter(
                    self.pfreqs[mask][self.auto_detected_peaks],
                    ppsd_scaled[self.auto_detected_peaks],
                    color='orange', marker='x', s=50, linewidths=2,
                    label='Auto-detected', zorder=6)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # ANNOTATED PEAKS
        # Plotted as cyan circles at (freq, scaled_amplitude).
        # amplitude_normalized is re-scaled to time units for consistent display.
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        if self.peak_annotations:
            freqs = [p['freq'] for p in self.peak_annotations]
            amps  = [p.get('amplitude_normalized', 0.0) * len(self.times)
                     for p in self.peak_annotations]

            self.ax.scatter(freqs, amps,
                            c='cyan', marker='o', s=100,
                            edgecolors='white', linewidths=2,
                            label='Annotated', zorder=10)

            # Guide lines
            if self.show_freq_guides.get():
                for freq in freqs:
                    self.ax.axvline(x=freq, color='cyan', linestyle='--',
                                    linewidth=1, alpha=0.5, zorder=4)
                    if not self.hide_text.get():
                        self.ax.text(
                            freq, self.ax.get_ylim()[1] * 0.95,
                            f"{freq:.1f}Hz", color='cyan', fontsize=9,
                            rotation=90, va='top', ha='right',
                            family='monospace', alpha=0.9)

            if self.show_time_guides.get():
                for amp in amps:
                    self.ax.axhline(y=amp, color='lime', linestyle='--',
                                    linewidth=1, alpha=0.5, zorder=4)
                    if not self.hide_text.get():
                        self.ax.text(
                            self.ax.get_xlim()[1] * 0.95, amp,
                            f"{amp:.1f}", color='lime', fontsize=9,
                            va='center', ha='right',
                            family='monospace', alpha=0.9)

            # Bounding box and harmonic overlays
            if self.show_bounding_box.get():
                f_min, f_max = min(freqs), max(freqs)
                a_min, a_max = min(amps),  max(amps)
                self._draw_shape(f_min, f_max, a_min, a_max,
                                 color='white', linewidth=2.5, alpha=0.8)

                for harmonic in self.harmonics:
                    if harmonic['show'].get():
                        mult    = harmonic['multiplier'].get()
                        h_f_min = f_min * mult
                        h_f_max = f_max * mult
                        self._draw_shape(h_f_min, h_f_max, a_min, a_max,
                                         color=harmonic['color'],
                                         linewidth=2, alpha=0.6)

        # Shared point annotations
        from yaaat.core.visualization import draw_shared_point_annotations
        draw_shared_point_annotations(self)

    def _draw_shape(self, f_min, f_max, a_min, a_max,
                    color, linewidth, alpha):
        """Draw rectangle, ellipse, or polygon bounding shape in freq/time space."""
        shape_type = self.bounding_box_shape.get()

        if shape_type == 'rectangle':
            self.ax.add_patch(plt.Rectangle(
                (f_min, a_min), f_max - f_min, a_max - a_min,
                fill=False, edgecolor=color,
                linewidth=linewidth, alpha=alpha, zorder=11))

        elif shape_type == 'ellipse':
            from matplotlib.patches import Ellipse
            self.ax.add_patch(Ellipse(
                ((f_min + f_max) / 2, (a_min + a_max) / 2),
                f_max - f_min, a_max - a_min,
                fill=False, edgecolor=color,
                linewidth=linewidth, alpha=alpha, zorder=11))

        elif shape_type == 'polygon':
            from matplotlib.patches import Polygon
            if self.peak_annotations:
                freqs = [p['freq'] for p in self.peak_annotations]
                amps  = [p.get('amplitude_normalized', 0.0) * len(self.times)
                         for p in self.peak_annotations]
                pts = list(zip(freqs, amps))
                self.ax.add_patch(Polygon(
                    pts, closed=True,
                    fill=False, edgecolor=color,
                    linewidth=linewidth, alpha=alpha, zorder=11))

    ##    <(''<)  <( ' ' )>  (>'')>
    # UPDATE DISPLAY OVERRIDE
    # Peak annotator needs a different axis initialization than BaseLayer —
    # vertical orientation with freq on x-axis, time on y-axis.
    ##    <(''<)  <( ' ' )>  (>'')>

    def update_display(self, recompute_spec=False):
        """Override BaseLayer.update_display() for vertical orientation.

        Full redraw: plots vertical spectrogram with freq on x-axis.
        Overlay-only: removes scatter collections and PSD lines, redraws overlays.
        """
        try:
            if self.y is None:
                return

            if recompute_spec or self.spec_image is None:
                self.ax.clear()

                # (つ -' _ '- )つ    (つ -' _ '- )つ
                # Vertical spectrogram display.
                # S_db is already rotated (orientation='vertical') so:
                #   rows = time frames, cols = frequency bins
                # freq_mask clips display to fmax_display.
                # extent: [freq_min, freq_max, 0, n_time_frames]
                # (つ -' _ '- )つ    (つ -' _ '- )つ
                freq_mask     = self.freqs <= self.fmax_display.get()
                S_db_display  = self.S_db[:, freq_mask]
                freqs_display = self.freqs[freq_mask]

                extent = [
                    freqs_display.min(), freqs_display.max(),
                    0, len(self.times)
                ]

                self.spec_image = self.ax.imshow(
                    S_db_display, aspect='auto',
                    extent=extent, origin='lower', cmap='magma')

                self.ax.set_xlim(self.fmin_display.get(),
                                 self.fmax_display.get())
                self.ax.set_ylim(0, len(self.times))
                self.ax.set_xlabel('Frequency (Hz)', fontsize=8)
                self.ax.set_ylabel('Time (frames)', fontsize=8)

            else:
                # Remove scatter collections and PSD lines for overlay-only redraw
                collections_to_remove = [
                    c for c in self.ax.collections
                    if isinstance(c, matplotlib.collections.PathCollection)
                ]
                for c in collections_to_remove:
                    c.remove()

                for line in self.ax.lines[:]:
                    line.remove()

                for patch in self.ax.patches[:]:
                    patch.remove()

                for text in self.ax.texts[:]:
                    text.remove()

            # Draw all overlays
            self.draw_custom_overlays()

            # Title
            if self.audio_files:
                filename    = self.audio_files[self.current_file_idx].name
                save_marker = "" if self.changes_made else "✓ "
                self.ax.set_title(
                    f"{save_marker}{filename} | "
                    f"{len(self.peak_annotations)} peaks | "
                    f"spec n_fft={self.n_fft_spect.get()} "
                    f"hop={self.hop_spect.get()} | "
                    f"psd n_fft={self.n_fft_psd.get()}",
                    fontsize=9)

            self.canvas.draw()
            self._update_stats()
            self._update_peak_info()

        except Exception as e:
            logger.error("ERROR in update_display: %s", e)
            logger.debug(traceback.format_exc())

    ##    <(''<)  <( ' ' )>  (>'')>
    # DISPLAY RANGE OVERRIDE
    # Peak annotator x-axis is frequency — different from BaseLayer y-axis.
    ##    <(''<)  <( ' ' )>  (>'')>

    def update_display_range(self):
        """Update frequency display range on x-axis (vertical orientation)."""
        if self.y is None:
            return
        self.ax.set_xlim(self.fmin_display.get(), self.fmax_display.get())
        self.canvas.draw_idle()

    def reset_zoom(self):
        """Reset zoom to full frequency range and full time extent."""
        if self.y is None:
            return
        self.zoom_stack = []
        self.ax.set_xlim(self.fmin_display.get(), self.fmax_display.get())
        self.ax.set_ylim(0, len(self.times))
        self.canvas.draw_idle()

    ##    <(''<)  <( ' ' )>  (>'')>
    # PARAMETER CHANGE HANDLERS
    ##    <(''<)  <( ' ' )>  (>'')>

    def _change_nfft_spect(self, new_nfft):
        """Update spectrogram n_fft and recompute dual view."""
        self.n_fft_spect.set(new_nfft)
        self._update_spect_button_highlights()
        if self.y is not None:
            self._recompute_display()

    def _change_hop_spect(self, new_hop):
        """Update spectrogram hop and recompute dual view."""
        self.hop_spect.set(new_hop)
        self._update_spect_button_highlights()
        if self.y is not None:
            self._recompute_display()

    def _change_nfft_psd(self, new_nfft):
        """Update PSD n_fft and recompute dual view."""
        self.n_fft_psd.set(new_nfft)
        self._update_psd_button_highlights()
        if self.y is not None:
            self._recompute_display()

    def _change_hop_psd(self, new_hop):
        """Update PSD hop and recompute dual view."""
        self.hop_psd.set(new_hop)
        self._update_psd_button_highlights()
        if self.y is not None:
            self._recompute_display()

    def _on_prominence_change(self, value):
        """Invalidate auto-detected peaks cache and redraw."""
        self.prom_label.config(text=f"{self.peak_prominence.get():.2f}")
        self.auto_detected_peaks = None
        if self.show_auto_peaks.get() and self.y is not None:
            self.update_display(recompute_spec=False)

    def _recompute_display(self):
        """Recompute dual view and redraw, preserving zoom."""
        if self.y is None:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self._compute_dual_view()
        self.spec_image = None
        self.update_display(recompute_spec=True)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

    ##    <(''<)  <( ' ' )>  (>'')>
    # BUTTON HIGHLIGHTS
    ##    <(''<)  <( ' ' )>  (>'')>

    def _update_spect_button_highlights(self):
        """Highlight active spectrogram n_fft and hop buttons."""
        for btn, val in self.nfft_spect_buttons:
            btn.config(bg='lightgreen' if val == self.n_fft_spect.get()
                       else 'SystemButtonFace',
                       relief=tk.SUNKEN if val == self.n_fft_spect.get()
                       else tk.RAISED)
        for btn, val in self.hop_spect_buttons:
            btn.config(bg='lightblue' if val == self.hop_spect.get()
                       else 'SystemButtonFace',
                       relief=tk.SUNKEN if val == self.hop_spect.get()
                       else tk.RAISED)

    def _update_psd_button_highlights(self):
        """Highlight active PSD n_fft and hop buttons."""
        for btn, val in self.nfft_psd_buttons:
            btn.config(bg='lightyellow' if val == self.n_fft_psd.get()
                       else 'SystemButtonFace',
                       relief=tk.SUNKEN if val == self.n_fft_psd.get()
                       else tk.RAISED)
        for btn, val in self.hop_psd_buttons:
            btn.config(bg='lightcoral' if val == self.hop_psd.get()
                       else 'SystemButtonFace',
                       relief=tk.SUNKEN if val == self.hop_psd.get()
                       else tk.RAISED)

    ##    <(''<)  <( ' ' )>  (>'')>
    # GUIDE AND BOUNDING BOX HELPERS
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
    # HARMONIC NUDGE HELPERS
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
    # CLEAR ACTIONS
    ##    <(''<)  <( ' ' )>  (>'')>

    def _clear_last_peak(self):
        """Remove the last annotated peak."""
        if self.peak_annotations:
            removed = self.peak_annotations.pop()
            self.changes_made = True
            self.update_display(recompute_spec=False)
            logger.info("Removed peak at %.1f Hz", removed['freq'])

    def _clear_all_peaks(self):
        """Clear all peak annotations after confirmation."""
        if self.peak_annotations:
            if messagebox.askyesno("Clear", "Remove all peaks?"):
                self.peak_annotations = []
                self.changes_made     = True
                self.update_display(recompute_spec=False)

    ##    <(''<)  <( ' ' )>  (>'')>
    # STATISTICS
    ##    <(''<)  <( ' ' )>  (>'')>

    def _update_stats(self):
        """Update peak statistics label."""
        if not self.peak_annotations:
            self.stats_label.config(text="No peaks annotated")
            return

        freqs       = [p['freq'] for p in self.peak_annotations]
        proms       = [p['prominence'] for p in self.peak_annotations]
        mean_freq   = np.mean(freqs)
        std_freq    = np.std(freqs)
        min_freq    = np.min(freqs)
        max_freq    = np.max(freqs)
        mean_prom   = np.mean(proms)

        self.stats_label.config(
            text=(f"μ={mean_freq:.1f} Hz σ={std_freq:.1f} Hz\n"
                  f"Range: {min_freq:.1f}–{max_freq:.1f} Hz\n"
                  f"Mean prominence: {mean_prom:.3f}"))

    def _update_peak_info(self):
        """Update peak count info label."""
        self.peak_info.config(
            text=(f"Peaks: {len(self.peak_annotations)} | "
                  f"Total: {self.total_peaks_across_files}"))

    def _count_total_peaks(self):
        """Count total peaks across all annotation files in the dataset."""
        self.total_peaks_across_files = 0
        self.total_files_annotated    = 0
        if self.annotation_dir is None:
            return

        for audio_file in self.audio_files:
            path = resolve_annotation_path(
                audio_file, self.base_audio_dir,
                self.annotation_dir, SUFFIX_PEAKS)
            if path.exists():
                data   = annotation_io.load_annotation_file(path)
                n_peaks = len(data.get('peaks', []))
                if n_peaks > 0:
                    self.total_peaks_across_files += n_peaks
                    self.total_files_annotated    += 1

    ##    <(''<)  <( ' ' )>  (>'')>
    # DEBUG
    ##    <(''<)  <( ' ' )>  (>'')>

    def _print_debug_info(self):
        """Print debug state to logger."""
        logger.info("=== PEAK ANNOTATOR DEBUG ===")
        logger.info("Audio loaded: %s", self.y is not None)
        if self.y is not None:
            logger.info("Audio length: %.2fs", len(self.y) / self.sr)
            logger.info("S_db shape: %s", self.S_db.shape)
            logger.info("PSD length: %d", len(self.pfreqs))
        logger.info("File: %d/%d", self.current_file_idx + 1,
                    len(self.audio_files))
        logger.info("Peaks: %d", len(self.peak_annotations))
        logger.info("Zoom stack: %d", len(self.zoom_stack))
        logger.info("Spec params: n_fft=%d hop=%d",
                    self.n_fft_spect.get(), self.hop_spect.get())
        logger.info("PSD params: n_fft=%d hop=%d",
                    self.n_fft_psd.get(), self.hop_psd.get())

    ##    <(''<)  <( ' ' )>  (>'')>
    # SAVE / LOAD
    ##    <(''<)  <( ' ' )>  (>'')>

    def save_custom_data(self):
        """Save peak annotations to _peaks.json via merge-write.

        spec_params uses vertical orientation to distinguish this file from
        horizontal-orientation files during param divergence checks.
        psd_params uses peak-annotator-specific n_fft_psd/hop_psd.
        contour_source points to the _changepoints.json canonical geometry file.
        Peak stats are computed at save time from current annotations.
        """
        if not self.audio_files or self.annotation_dir is None:
            return

        path = resolve_annotation_path(
            self.audio_files[self.current_file_idx],
            self.base_audio_dir,
            self.annotation_dir,
            SUFFIX_PEAKS
        )

        # contour_source points to canonical geometry file for cross-tab reference
        changepoints_path = resolve_annotation_path(
            self.audio_files[self.current_file_idx],
            self.base_audio_dir,
            self.annotation_dir,
            SUFFIX_CHANGEPOINTS
        )

        # Compute peak stats at save time
        if self.peak_annotations:
            freqs      = [p['freq'] for p in self.peak_annotations]
            peak_stats = {
                'num_peaks':  len(self.peak_annotations),
                'mean_freq':  float(np.mean(freqs)),
                'std_freq':   float(np.std(freqs)),
                'min_freq':   float(np.min(freqs)),
                'max_freq':   float(np.max(freqs)),
                'freq_range': float(np.max(freqs) - np.min(freqs)),
            }
        else:
            peak_stats = {}

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # spec_params: orientation='vertical' distinguishes _peaks.json from
        # horizontal-orientation files on param divergence check.
        # n_fft and hop_length reflect n_fft_spect and hop_spect.
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        spec_params = {
            "n_fft":        self.n_fft_spect.get(),
            "hop_length":   self.hop_spect.get(),
            "fmin_calc":    self.fmin_calc.get(),
            "fmax_calc":    self.fmax_calc.get(),
            "fmin_display": self.fmin_display.get(),
            "fmax_display": self.fmax_display.get(),
            "orientation":  "vertical",
        }

        psd_params = {
            "n_fft":      self.n_fft_psd.get(),
            "hop_length": self.hop_psd.get(),
            "fmin":       self.fmin_calc.get(),
            "fmax":       self.fmax_calc.get(),
        }

        tab_data = {
            "audio_file":      str(self.audio_files[self.current_file_idx]),
            "contour_source":  changepoints_path.name,
            "peaks":           self.peak_annotations,
            "peak_stats":      peak_stats,
            "spec_params":     spec_params,
            "psd_params":      psd_params,
            "skip":            False,
            "skip_reason":     "",
        }

        merge_and_save(path, tab_data)
        self.changes_made = True
        self._count_total_peaks()
        self.update_display(recompute_spec=False)
        self._update_peak_info()
        logger.info("Saved %d peaks to %s",
                    len(self.peak_annotations), path.name)

    def load_custom_data(self):
        """Load peak annotations from _peaks.json with param divergence check."""
        if not self.audio_files or self.annotation_dir is None:
            return

        path = resolve_annotation_path(
            self.audio_files[self.current_file_idx],
            self.base_audio_dir,
            self.annotation_dir,
            SUFFIX_PEAKS
        )

        self.peak_annotations    = []
        self.auto_detected_peaks = None

        data = load_and_check_params(
            path, self, SUFFIX_PEAKS, self.annotation_dir)

        if not data:
            return

        self.peak_annotations = data.get('peaks', [])
        logger.info("Loaded %d peaks from %s",
                    len(self.peak_annotations), path.name)


##    <(''<)  <( ' ' )>  (>'')>

def main():
    """Launch PeakAnnotator as a standalone tab."""
    root = tk.Tk()
    app  = PeakAnnotator(root)
    root.geometry("1400x800")
    root.mainloop()


if __name__ == "__main__":
    main()

# U S A G I
# from yaaat.tabs.peak_annotator import PeakAnnotator
# root = tk.Tk(); app = PeakAnnotator(root); root.geometry("1400x800"); root.mainloop()