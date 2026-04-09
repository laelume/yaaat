"""
tabs/harmonic_annotator.py

Harmonic annotator tab for YAAAT.
Inherits from BaseLayer — single-file spectrogram view.

Responsibilities:
    - Auto-detect F0 from mean spectrum using peak prominence
    - Build harmonic series from detected F0
    - Track harmonic ridges across time using selectable methods:
        max, peaks, centroid, parabolic, peak_ratio
    - Compute inter-harmonic valley boundaries using minimum energy search
      (basis for wavelet/harmonic band analysis — not cosmetic)
    - Render shaded harmonic bands: center ridge + valley-bounded extent
    - Allow dragging of harmonic lines to correct detections
    - Expose contour extraction over ridges (raw, smooth, poly, spline)
    - Save/load to _harmonics.json via annotation_io merge-write

Future integration (deferred):
    - FuzzyValley detection as an alternative valley method for comparison
      against min_energy. FuzzyValley is computationally heavy and requires
      its own evaluation pipeline. It will be loaded as an optional plugin
      and compared against the current min_energy baseline.

Valley boundary schema (parallel arrays, self-contained per boundary):
    "dynamic_lower" — min energy below F0 down to fmin_display
    "h{n}_h{n+1}"  — min energy between adjacent harmonic ridges
    "dynamic_upper" — min energy above highest harmonic up to fmax_display

    Convention follows HarmonicAnnotator.compute_boundary_data() from the
    original layers/harmonic_annotator.py for lower/upper naming.
    h1_h2 naming is defined here as the canonical inter-harmonic convention.

Ridge schema (parallel arrays, matching valley boundary format):
    {"times": [...], "freqs": [...]}

Annotation file: {prefix}_{stem}_harmonics.json
    Written via annotation_io.merge_and_save() — preserves other tab data.
"""

import json
import logging
import traceback

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from yaaat.core.base_layer import BaseLayer
from yaaat.core import audio_utils
from yaaat.core import annotation_io

# (つ -' _ '- )つ    (つ -' _ '- )つ
# Peak ratio harmonics — optional import, used for peak_ratio ridge method
# If algs package is not available, peak_ratio method is silently disabled
# (つ -' _ '- )つ    (つ -' _ '- )つ
try:
    from yaaat.algs.peak_ratio_harmonics import (
        analyze_harmonics,
        compute_harmonic_ratio_matrix,
    )
    _PEAK_RATIO_AVAILABLE = True
except ImportError:
    _PEAK_RATIO_AVAILABLE = False

logger = logging.getLogger(__name__)


# (つ -' _ '- )つ    (つ -' _ '- )つ
# HARMONIC COLOR PALETTE
# Consistent color assignment by harmonic number across all rendering methods.
# Index 0 = H1, index 1 = H2, etc. Wraps for harmonics beyond palette length.
# (つ -' _ '- )つ    (つ -' _ '- )つ

_HARMONIC_COLORS = [
    'red', 'orange', 'yellow', 'green',
    'blue', 'purple', 'pink', 'brown'
]

# Alpha values for shaded band regions — kept low to preserve spectrogram legibility
_BAND_FILL_ALPHA   = 0.15
_RIDGE_LINE_ALPHA  = 0.7
_VALLEY_LINE_ALPHA = 0.4


##    <(''<)  <( ' ' )>  (>'')>

class HarmonicAnnotator(BaseLayer):
    """Harmonic annotation tab — auto-detection with draggable correction lines.

    Detects F0 from the mean spectrum, builds a harmonic series, tracks
    ridges across time, and computes inter-harmonic valley boundaries for
    spectral band visualization and downstream metric computation.
    """

    def __init__(self, root):
        """Initialize harmonic state before calling BaseLayer.__init__."""

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # HARMONIC STATE
        # harmonic_lines: list of dicts — one per active harmonic
        #   {'freq': float, 'num': int, 'line': matplotlib Line2D or None}
        # harmonic_ridges: dict keyed by harmonic number (int)
        #   {1: {'times': [...], 'freqs': [...]}, ...}
        # valley_boundaries: dict keyed by boundary name string
        #   {'dynamic_lower': {'times': [...], 'freqs': [...]},
        #    'h1_h2': {...}, 'dynamic_upper': {...}}
        # harmonic_contours: dict keyed by harmonic number
        #   {1: {'times': [...], 'freqs': [...]}, ...}
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.harmonic_lines     = []
        self.harmonic_ridges    = {}
        self.valley_boundaries  = {}
        self.harmonic_contours  = {}
        self.detected_f0        = None
        self.mean_spectrum      = None

        # Peak ratio analysis result — stored for saving when method is active
        self.peak_ratio_result  = None

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # DRAG STATE
        # selected_line: reference to the harmonic_lines entry being dragged
        # drag_start_y: frequency at drag start for computing delta on release
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.selected_line  = None
        self.drag_start_y   = None

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # DETECTION PARAMETERS — tk vars bound to UI controls
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.prominence         = tk.DoubleVar(value=5.0)
        self.freq_min           = tk.IntVar(value=500)
        self.freq_max           = tk.IntVar(value=8000)
        self.peak_tolerance     = tk.DoubleVar(value=0.1)
        self.ridge_method       = tk.StringVar(value='max')

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # VALLEY METHOD
        # 'min_energy' — minimum energy search between adjacent ridges.
        # This is the analytically correct method for harmonic band definition
        # and the basis for wavelet band analysis. FuzzyValley will be added
        # as a second option in a future comparative pass.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.valley_method      = tk.StringVar(value='min_energy')

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # CONTOUR EXTRACTION PARAMETERS
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.show_contour           = tk.BooleanVar(value=False)
        self.contour_method         = tk.StringVar(value='raw')
        self.contour_smoothness     = tk.DoubleVar(value=5.0)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # DISPLAY OPTIONS
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.show_ridges        = tk.BooleanVar(value=True)
        self.show_valleys       = tk.BooleanVar(value=False)
        self.show_bands         = tk.BooleanVar(value=True)

        super().__init__(root)

        if isinstance(root, tk.Tk):
            self.root.title("Harmonic Annotator - YAAAT")

    ##    <(''<)  <( ' ' )>  (>'')>
    # CUSTOM CONTROLS
    ##    <(''<)  <( ' ' )>  (>'')>

    def setup_custom_controls(self):
        """Build harmonic-specific controls in the scrollable control panel."""

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # F0 DETECTION
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="F0 Detection:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        freq_frame = ttk.Frame(self.control_panel)
        freq_frame.pack(fill=tk.X, pady=2)
        ttk.Label(freq_frame, text="Range (Hz):", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.freq_min, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(freq_frame, text="-", font=('', 8)).pack(side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.freq_max, width=5).pack(side=tk.LEFT, padx=2)

        prom_frame = ttk.Frame(self.control_panel)
        prom_frame.pack(fill=tk.X, pady=2)
        ttk.Label(prom_frame, text="Prominence:", font=('', 8)).pack(side=tk.LEFT)
        ttk.Scale(prom_frame, from_=0.1, to=20, variable=self.prominence,
                  orient=tk.HORIZONTAL,
                  command=self.on_prominence_change).pack(
                      side=tk.LEFT, fill=tk.X, expand=True)
        self.prom_label = ttk.Label(prom_frame,
                                    text=f"{self.prominence.get():.1f}",
                                    font=('', 8), width=5)
        self.prom_label.pack(side=tk.LEFT)

        ttk.Button(self.control_panel, text="Detect F0",
                   command=self.detect_f0).pack(fill=tk.X, pady=2)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # HARMONIC TOGGLE BUTTONS H1-H5
        # Toggle buttons add/remove individual harmonics from harmonic_lines.
        # H1 = F0 and cannot be removed once detected.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Add/Remove Harmonics:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        harm_toggle_frame = ttk.Frame(self.control_panel)
        harm_toggle_frame.pack(fill=tk.X, pady=2)
        ttk.Label(harm_toggle_frame, text="Toggle:", font=('', 8)).pack(side=tk.LEFT)

        self.harmonic_buttons = {}
        for h in [1, 2, 3, 4, 5]:
            btn = tk.Button(harm_toggle_frame, text=f"H{h}", width=4,
                            relief=tk.RAISED, bg='lightgray',
                            command=lambda n=h: self.toggle_harmonic(n))
            btn.pack(side=tk.LEFT, padx=2)
            self.harmonic_buttons[h] = btn

        ttk.Button(self.control_panel, text="Clear All (except F0)",
                   command=self.clear_all_harmonics).pack(fill=tk.X, pady=2)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # RIDGE DETECTION METHOD
        # max       — simple max search within tolerance window
        # peaks     — prominence-based peak detection within window
        # centroid  — spectral centroid within window
        # parabolic — parabolic interpolation around peak
        # peak_ratio — peak ratio harmonic analysis (requires yaaat.algs)
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Ridge Detection Method:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        ridge_frame = ttk.Frame(self.control_panel)
        ridge_frame.pack(fill=tk.X, pady=2)

        for method in ['max', 'peaks', 'centroid', 'parabolic']:
            ttk.Radiobutton(ridge_frame, text=method.capitalize(),
                            variable=self.ridge_method, value=method,
                            command=self.on_ridge_method_change).pack(
                                side=tk.LEFT, padx=5)

        # Peak ratio only shown if algs package is available
        if _PEAK_RATIO_AVAILABLE:
            ttk.Radiobutton(self.control_panel, text="Peak Ratio",
                            variable=self.ridge_method, value='peak_ratio',
                            command=self.on_ridge_method_change).pack(
                                anchor=tk.W, padx=5)

        tol_frame = ttk.Frame(self.control_panel)
        tol_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tol_frame, text="Peak Tolerance:", font=('', 8)).pack(side=tk.LEFT)
        self.tol_scale = ttk.Scale(tol_frame, from_=0.01, to=0.5,
                                   variable=self.peak_tolerance,
                                   orient=tk.HORIZONTAL,
                                   command=self.on_tolerance_change)
        self.tol_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.tol_label = ttk.Label(tol_frame, text="0.10", font=('', 8), width=5)
        self.tol_label.pack(side=tk.LEFT)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # VALLEY METHOD
        # min_energy — minimum energy search between adjacent ridges.
        # This defines the spectral band extent for each harmonic, which is
        # the basis for wavelet decomposition band assignment and HNR computation.
        # FuzzyValley will be added as a second radio option when integrated.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Valley Method:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        ttk.Radiobutton(self.control_panel, text="Min Energy",
                        variable=self.valley_method, value='min_energy',
                        command=self.on_valley_method_change).pack(
                            anchor=tk.W, padx=5)

        # TODO: Add FuzzyValley radio button here when integrated.
        # FuzzyValley requires its own evaluation pass and will be compared
        # against min_energy as a baseline. Placeholder:
        # ttk.Radiobutton(self.control_panel, text="FuzzyValley (future)",
        #                 variable=self.valley_method, value='fuzzy_valley',
        #                 command=self.on_valley_method_change, state='disabled')

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # DISPLAY OPTIONS
        # show_ridges — draw time-varying ridge lines per harmonic
        # show_valleys — draw valley boundary lines
        # show_bands  — shade the region between valley boundaries per harmonic
        #               This is the primary visualization for harmonic band analysis.
        #               Band extent = [valley below, valley above] for each harmonic.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Display:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))

        ttk.Checkbutton(self.control_panel, text="Show Ridges",
                        variable=self.show_ridges,
                        command=lambda: self.update_display()).pack(anchor=tk.W)

        ttk.Checkbutton(self.control_panel, text="Show Valley Lines",
                        variable=self.show_valleys,
                        command=lambda: self.update_display()).pack(anchor=tk.W)

        ttk.Checkbutton(self.control_panel, text="Show Harmonic Bands",
                        variable=self.show_bands,
                        command=lambda: self.update_display()).pack(anchor=tk.W)

        ttk.Checkbutton(self.control_panel, text="Show Contour",
                        variable=self.show_contour,
                        command=self.on_show_contour_toggle).pack(anchor=tk.W)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # CONTOUR EXTRACTION
        # Extracts a smoothed frequency contour from the raw ridge for each harmonic.
        # Methods: raw (no smoothing), smooth (moving average), poly (polyfit), spline
        # Smoothness slider maps to window size (smooth), polynomial order (poly),
        # or spline smoothing factor (spline).
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(self.control_panel, text="Contour Extraction:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(4, 2))

        contour_frame = ttk.Frame(self.control_panel)
        contour_frame.pack(fill=tk.X, pady=2)

        for method in ['raw', 'smooth', 'poly', 'spline']:
            ttk.Radiobutton(contour_frame, text=method.capitalize(),
                            variable=self.contour_method, value=method,
                            command=self.on_contour_method_change).pack(
                                side=tk.LEFT, padx=5)

        contour_smooth_frame = ttk.Frame(self.control_panel)
        contour_smooth_frame.pack(fill=tk.X, pady=2)
        ttk.Label(contour_smooth_frame, text="Smoothness:",
                  font=('', 8)).pack(side=tk.LEFT)
        ttk.Scale(contour_smooth_frame, from_=1.0, to=20.0,
                  variable=self.contour_smoothness, orient=tk.HORIZONTAL,
                  command=self.on_contour_smoothness_change).pack(
                      side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # Active harmonics list — updated on detect/toggle
        ttk.Label(self.control_panel, text="Active Harmonics:",
                  font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.harmonics_listbox = tk.Listbox(self.control_panel, height=5,
                                            font=('', 8))
        self.harmonics_listbox.pack(fill=tk.X, pady=2)

        self.info_label = ttk.Label(self.control_panel,
                                    text="No F0 detected",
                                    wraplength=300, font=('', 8))
        self.info_label.pack(fill=tk.X, pady=2)

    ##    <(''<)  <( ' ' )>  (>'')>
    # AUDIO PROCESSING HOOK
    ##    <(''<)  <( ' ' )>  (>'')>

    def process_audio(self):
        """Auto-detect F0 when a new file is loaded."""
        if self.y is not None:
            self.detect_f0()

    ##    <(''<)  <( ' ' )>  (>'')>
    # F0 AND HARMONIC DETECTION
    ##    <(''<)  <( ' ' )>  (>'')>

    def detect_f0(self):
        """Detect F0 from the mean spectrum using prominence-based peak detection.

        Uses the time-averaged spectrogram as the detection signal.
        Strongest prominent peak within freq_min/freq_max is taken as F0.
        Clears existing harmonic lines and re-adds H1 at detected F0.
        Triggers ridge and valley detection if ridges are enabled.
        """
        if self.S_db is None:
            return

        self.mean_spectrum = np.mean(self.S_db, axis=1)

        # Mask to detection frequency range
        freq_mask = (
            (self.freqs >= self.freq_min.get()) &
            (self.freqs <= self.freq_max.get())
        )
        masked_spectrum = self.mean_spectrum.copy()
        masked_spectrum[~freq_mask] = -np.inf

        peaks, properties = find_peaks(
            masked_spectrum,
            prominence=self.prominence.get(),
            distance=max(5, int(50 / (self.freqs[1] - self.freqs[0])))
        )

        if len(peaks) == 0:
            self.detected_f0 = None
            self.info_label.config(text="No peaks found")
            return

        # Use strongest peak as F0
        strongest_idx = peaks[np.argmax(masked_spectrum[peaks])]
        self.detected_f0 = float(self.freqs[strongest_idx])

        # Reset harmonic lines to H1 only
        self.harmonic_lines = [{
            'freq': self.detected_f0,
            'num':  1,
            'line': None
        }]

        if self.show_ridges.get():
            self.detect_harmonic_ridges()

        # Valley boundaries require ridges to be computed first
        self._compute_valley_boundaries()

        if self.show_contour.get():
            self.compute_contours()

        self.update_display()
        self.update_info()
        self.update_button_states()

        logger.info("Detected F0: %.1f Hz", self.detected_f0)

    ##    <(''<)  <( ' ' )>  (>'')>
    # RIDGE DETECTION
    ##    <(''<)  <( ' ' )>  (>'')>

    def detect_harmonic_ridges(self):
        """Dispatch ridge detection to the currently selected method.

        Updates self.harmonic_ridges with parallel array dicts per harmonic.
        Recomputes contours if contour display is active.
        """
        if self.detected_f0 is None or self.S_db is None:
            return

        method = self.ridge_method.get()

        dispatch = {
            'max':        self._detect_ridges_max,
            'peaks':      self._detect_ridges_peaks,
            'centroid':   self._detect_ridges_centroid,
            'parabolic':  self._detect_ridges_parabolic,
            'peak_ratio': self._detect_ridges_peak_ratio,
        }

        fn = dispatch.get(method)
        if fn is None:
            logger.warning("Unknown ridge method: %s", method)
            return

        raw_ridges = fn()

        # Convert list-of-tuples to parallel array dicts for consistent storage
        # [(time, freq), ...] -> {'times': [...], 'freqs': [...]}
        self.harmonic_ridges = {}
        for harm_num, ridge in raw_ridges.items():
            if ridge:
                times, freqs = zip(*ridge)
                self.harmonic_ridges[harm_num] = {
                    'times': list(times),
                    'freqs': list(freqs),
                }

        if self.show_contour.get():
            self.compute_contours()

    def _detect_ridges_max(self):
        """Ridge detection via maximum search within a 10% tolerance window."""
        ridges = {}
        for h in self.harmonic_lines:
            harm_num      = h['num']
            expected_freq = h['freq']
            tolerance     = expected_freq * 0.1
            ridge         = []

            for t_idx in range(self.S_db.shape[1]):
                spectrum  = self.S_db[:, t_idx]
                freq_mask = (
                    (self.freqs >= expected_freq - tolerance) &
                    (self.freqs <= expected_freq + tolerance)
                )
                if np.any(freq_mask):
                    search_idx   = np.where(freq_mask)[0]
                    local_max    = np.argmax(spectrum[freq_mask])
                    freq_idx     = search_idx[local_max]
                    ridge.append((self.times[t_idx], self.freqs[freq_idx]))

            if ridge:
                ridges[harm_num] = ridge

        return ridges

    def _detect_ridges_peaks(self):
        """Ridge detection via prominence-based peak search within tolerance window."""
        ridges = {}
        for h in self.harmonic_lines:
            harm_num      = h['num']
            expected_freq = h['freq']
            tolerance     = expected_freq * self.peak_tolerance.get()
            ridge         = []

            for t_idx in range(self.S_db.shape[1]):
                spectrum  = self.S_db[:, t_idx]
                freq_mask = (
                    (self.freqs >= expected_freq - tolerance) &
                    (self.freqs <= expected_freq + tolerance)
                )
                if np.any(freq_mask):
                    search_idx     = np.where(freq_mask)[0]
                    local_spectrum = spectrum[freq_mask]
                    peaks, _       = find_peaks(local_spectrum,
                                               prominence=self.prominence.get())
                    if len(peaks) > 0:
                        peak_freqs  = self.freqs[search_idx[peaks]]
                        closest     = np.argmin(np.abs(peak_freqs - expected_freq))
                        freq_idx    = search_idx[peaks[closest]]
                        ridge.append((self.times[t_idx], self.freqs[freq_idx]))

            if ridge:
                ridges[harm_num] = ridge

        return ridges

    def _detect_ridges_centroid(self):
        """Ridge detection via spectral centroid within a 15% tolerance window."""
        ridges = {}
        for h in self.harmonic_lines:
            harm_num      = h['num']
            expected_freq = h['freq']
            tolerance     = expected_freq * 0.15
            ridge         = []

            for t_idx in range(self.S_db.shape[1]):
                spectrum  = self.S_db[:, t_idx]
                freq_mask = (
                    (self.freqs >= expected_freq - tolerance) &
                    (self.freqs <= expected_freq + tolerance)
                )
                if np.any(freq_mask):
                    search_idx     = np.where(freq_mask)[0]
                    local_spectrum = spectrum[freq_mask]
                    freq_values    = self.freqs[search_idx]

                    # Convert dB to linear for centroid computation
                    linear         = 10 ** (local_spectrum / 20)
                    total_energy   = np.sum(linear)
                    if total_energy > 0:
                        centroid = np.sum(freq_values * linear) / total_energy
                        ridge.append((self.times[t_idx], centroid))

            if ridge:
                ridges[harm_num] = ridge

        return ridges

    def _detect_ridges_parabolic(self):
        """Ridge detection via parabolic interpolation around spectral peak."""
        ridges = {}
        for h in self.harmonic_lines:
            harm_num      = h['num']
            expected_freq = h['freq']
            tolerance     = expected_freq * 0.1
            ridge         = []

            for t_idx in range(self.S_db.shape[1]):
                spectrum  = self.S_db[:, t_idx]
                freq_mask = (
                    (self.freqs >= expected_freq - tolerance) &
                    (self.freqs <= expected_freq + tolerance)
                )
                if np.any(freq_mask):
                    search_idx     = np.where(freq_mask)[0]
                    local_spectrum = spectrum[freq_mask]
                    peak_idx       = np.argmax(local_spectrum)

                    if 0 < peak_idx < len(local_spectrum) - 1:
                        y1 = local_spectrum[peak_idx - 1]
                        y2 = local_spectrum[peak_idx]
                        y3 = local_spectrum[peak_idx + 1]

                        # Parabolic peak offset formula
                        p        = (y3 - y1) / (2 * (2 * y2 - y1 - y3))
                        freq_idx = search_idx[peak_idx]
                        freq_step = self.freqs[1] - self.freqs[0]
                        interp_freq = self.freqs[freq_idx] + p * freq_step
                        ridge.append((self.times[t_idx], interp_freq))

            if ridge:
                ridges[harm_num] = ridge

        return ridges

    def _detect_ridges_peak_ratio(self):
        """Ridge detection via peak ratio harmonic analysis.

        Requires yaaat.algs.peak_ratio_harmonics. Falls back to max method
        if the package is not available.
        """
        if not _PEAK_RATIO_AVAILABLE:
            logger.warning("peak_ratio_harmonics not available, falling back to max")
            return self._detect_ridges_max()

        self.peak_ratio_result = analyze_harmonics(
            self.S_db, self.freqs,
            f_min=self.freq_min.get(),
            f_max=self.freq_max.get(),
            prominence_base=self.prominence.get(),
            prominence_scale='log',
            max_harmonic=50,
            tolerance=0.05
        )

        ridges = {}
        for h in self.harmonic_lines:
            harm_num      = h['num']
            expected_freq = h['freq']

            if len(self.peak_ratio_result['peak_freqs']) > 0:
                peak_freqs  = self.peak_ratio_result['peak_freqs']
                closest_idx = np.argmin(np.abs(peak_freqs - expected_freq))
                matched     = peak_freqs[closest_idx]

                if np.abs(matched - expected_freq) / expected_freq < 0.1:
                    # Peak ratio gives a static frequency — horizontal ridge
                    ridge = [(t, matched) for t in self.times]
                    ridges[harm_num] = ridge

        return ridges

    ##    <(''<)  <( ' ' )>  (>'')>
    # VALLEY BOUNDARY COMPUTATION
    # Valley boundaries define the spectral extent of each harmonic band.
    # This is the basis for wavelet band assignment, HNR computation,
    # and inter-harmonic interference detection.
    #
    # For each time frame, the boundary between H_n and H_(n+1) is the
    # frequency of minimum energy in the spectrogram between the two ridges.
    # This is stored as a parallel array dict {'times': [...], 'freqs': [...]}.
    #
    # dynamic_lower: minimum energy between fmin_display and H1 ridge
    # h{n}_h{n+1}:  minimum energy between H_n ridge and H_(n+1) ridge
    # dynamic_upper: minimum energy between highest harmonic ridge and fmax_display
    #
    # Convention: 'dynamic_lower' and 'dynamic_upper' naming follows
    # HarmonicAnnotator.compute_boundary_data() from the original codebase.
    # 'h{n}_h{n+1}' naming is defined here as the canonical inter-harmonic
    # convention — not present in the original code.
    ##    <(''<)  <( ' ' )>  (>'')>

    def _compute_valley_boundaries(self):
        """Compute valley boundaries for all harmonic pairs and dynamic edges.

        Dispatches to the currently selected valley method.
        Currently only 'min_energy' is implemented. FuzzyValley is deferred.
        Populates self.valley_boundaries with parallel array dicts.
        """
        method = self.valley_method.get()

        if method == 'min_energy':
            self._compute_valleys_min_energy()
        else:
            # TODO: Dispatch to FuzzyValley when integrated
            logger.warning("Valley method '%s' not yet implemented, using min_energy", method)
            self._compute_valleys_min_energy()

    def _compute_valleys_min_energy(self):
        """Compute valley boundaries using minimum energy search per time frame.

        For each adjacent harmonic pair, at each time frame, finds the
        frequency bin with minimum spectral energy between the two ridge
        frequencies. This defines the band edge.

        Dynamic lower boundary: min energy between fmin_display and H1 ridge.
        Dynamic upper boundary: min energy between highest ridge and fmax_display.
        """
        if not self.harmonic_ridges:
            self.valley_boundaries = {}
            return

        self.valley_boundaries = {}

        # Sort harmonics by number for consistent adjacent pair iteration
        sorted_harm_nums = sorted(self.harmonic_ridges.keys())

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # DYNAMIC LOWER BOUNDARY
        # Search between fmin_display and H1 ridge at each time frame.
        # Represents the lower edge of the H1 spectral band.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        h1_num    = sorted_harm_nums[0]
        h1_ridge  = self.harmonic_ridges[h1_num]
        h1_times  = np.array(h1_ridge['times'])
        h1_freqs  = np.array(h1_ridge['freqs'])

        lower_times  = []
        lower_freqs  = []
        fmin_display = self.fmin_display.get()

        for t_idx in range(self.S_db.shape[1]):
            t          = self.times[t_idx]
            h1_freq_at_t = float(np.interp(t, h1_times, h1_freqs))

            # Search from fmin_display up to H1 ridge
            mask = (self.freqs >= fmin_display) & (self.freqs <= h1_freq_at_t)
            if np.any(mask):
                search_spectrum = self.S_db[:, t_idx][mask]
                min_idx         = np.argmin(search_spectrum)
                valley_freq     = self.freqs[np.where(mask)[0][min_idx]]
            else:
                valley_freq = fmin_display

            lower_times.append(t)
            lower_freqs.append(float(valley_freq))

        self.valley_boundaries['dynamic_lower'] = {
            'times': lower_times,
            'freqs': lower_freqs,
        }

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # INTER-HARMONIC VALLEY BOUNDARIES
        # For each adjacent pair (H_n, H_{n+1}), find minimum energy between
        # the two ridges at each time frame.
        # Key format: 'h{n}_h{n+1}' — canonical inter-harmonic naming convention.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        for i in range(len(sorted_harm_nums) - 1):
            n     = sorted_harm_nums[i]
            n1    = sorted_harm_nums[i + 1]
            key   = f"h{n}_h{n1}"

            ridge_n  = self.harmonic_ridges[n]
            ridge_n1 = self.harmonic_ridges[n1]

            times_n  = np.array(ridge_n['times'])
            freqs_n  = np.array(ridge_n['freqs'])
            times_n1 = np.array(ridge_n1['times'])
            freqs_n1 = np.array(ridge_n1['freqs'])

            valley_times = []
            valley_freqs = []

            for t_idx in range(self.S_db.shape[1]):
                t           = self.times[t_idx]
                freq_lo     = float(np.interp(t, times_n,  freqs_n))
                freq_hi     = float(np.interp(t, times_n1, freqs_n1))

                # Ensure lo < hi regardless of ridge ordering
                freq_lo, freq_hi = sorted([freq_lo, freq_hi])

                mask = (self.freqs >= freq_lo) & (self.freqs <= freq_hi)
                if np.any(mask):
                    search_spectrum = self.S_db[:, t_idx][mask]
                    min_idx         = np.argmin(search_spectrum)
                    valley_freq     = self.freqs[np.where(mask)[0][min_idx]]
                else:
                    # No bins between ridges — ridges overlap, use midpoint
                    valley_freq = (freq_lo + freq_hi) / 2.0

                valley_times.append(t)
                valley_freqs.append(float(valley_freq))

            self.valley_boundaries[key] = {
                'times': valley_times,
                'freqs': valley_freqs,
            }

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # DYNAMIC UPPER BOUNDARY
        # Search between highest harmonic ridge and fmax_display.
        # Represents the upper edge of the highest harmonic's spectral band.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        h_max_num   = sorted_harm_nums[-1]
        h_max_ridge = self.harmonic_ridges[h_max_num]
        h_max_times = np.array(h_max_ridge['times'])
        h_max_freqs = np.array(h_max_ridge['freqs'])

        upper_times  = []
        upper_freqs  = []
        fmax_display = self.fmax_display.get()

        for t_idx in range(self.S_db.shape[1]):
            t               = self.times[t_idx]
            h_max_freq_at_t = float(np.interp(t, h_max_times, h_max_freqs))

            mask = (self.freqs >= h_max_freq_at_t) & (self.freqs <= fmax_display)
            if np.any(mask):
                search_spectrum = self.S_db[:, t_idx][mask]
                min_idx         = np.argmin(search_spectrum)
                valley_freq     = self.freqs[np.where(mask)[0][min_idx]]
            else:
                valley_freq = fmax_display

            upper_times.append(t)
            upper_freqs.append(float(valley_freq))

        self.valley_boundaries['dynamic_upper'] = {
            'times': upper_times,
            'freqs': upper_freqs,
        }

    ##    <(''<)  <( ' ' )>  (>'')>
    # CONTOUR EXTRACTION
    # Extracts a smoothed frequency contour from each harmonic ridge.
    # Stored separately from ridges — ridges are detection output,
    # contours are the annotation-ready smoothed representation.
    ##    <(''<)  <( ' ' )>  (>'')>

    def compute_contours(self):
        """Compute smoothed contours from current harmonic ridges."""
        self.harmonic_contours = {}
        if not self.harmonic_ridges:
            return

        method = self.contour_method.get()

        for harm_num, ridge in self.harmonic_ridges.items():
            times = np.array(ridge['times'])
            freqs = np.array(ridge['freqs'])

            if method == 'raw':
                contour_freqs = freqs
            elif method == 'smooth':
                contour_freqs = self._smooth_contour(
                    freqs, int(self.contour_smoothness.get()))
            elif method == 'poly':
                contour_freqs = self._polyfit_contour(
                    times, freqs, int(self.contour_smoothness.get()))
            elif method == 'spline':
                contour_freqs = self._spline_contour(
                    times, freqs, int(self.contour_smoothness.get()))
            else:
                contour_freqs = freqs

            self.harmonic_contours[harm_num] = {
                'times': times.tolist(),
                'freqs': contour_freqs.tolist(),
            }

    def _smooth_contour(self, freqs, window):
        """Moving-average smoothing over frequency contour."""
        window = max(1, int(window))
        if window == 1 or len(freqs) < 3:
            return freqs
        kernel = np.ones(window) / window
        padded = np.pad(freqs, (window // 2, window - 1 - window // 2), mode='edge')
        return np.convolve(padded, kernel, mode='valid')

    def _polyfit_contour(self, times, freqs, order_hint):
        """Polynomial fit of frequency vs time. order_hint maps to degree 1-3."""
        if len(times) < 3:
            return freqs
        order = 1 if order_hint < 5 else (2 if order_hint < 10 else 3)
        t0 = times.mean()
        ts = times - t0
        try:
            coeffs = np.polyfit(ts, freqs, order)
            return np.poly1d(coeffs)(ts)
        except np.linalg.LinAlgError:
            return freqs

    def _spline_contour(self, times, freqs, smooth_hint):
        """Spline smoothing of frequency vs time."""
        if len(times) < 3:
            return freqs
        t0 = times.mean()
        ts = times - t0
        s  = max(1, int(smooth_hint)) * np.var(freqs) * 0.1
        try:
            return UnivariateSpline(ts, freqs, s=s)(ts)
        except Exception:
            return freqs

    ##    <(''<)  <( ' ' )>  (>'')>
    # OVERLAY RENDERING
    ##    <(''<)  <( ' ' )>  (>'')>

    def draw_custom_overlays(self):
        """Draw harmonic lines, ridges, valley boundaries, and shaded bands.

        Rendering order:
            1. Shaded harmonic bands (fill_between valley boundaries)
               Each band is shaded between its lower and upper valley boundary.
               This visually encodes the spectral extent of each harmonic.
            2. Ridge lines (time-varying center of each harmonic)
            3. Valley boundary lines (if show_valleys enabled)
            4. Static horizontal harmonic lines (for dragging)
            5. Contour lines (smoothed ridges, if show_contour enabled)
            6. Shared point annotations
        """
        sorted_harm_nums = sorted(
            h['num'] for h in self.harmonic_lines)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # 1. SHADED HARMONIC BANDS
        # For each harmonic, shade the region between its lower and upper
        # valley boundary. Lower boundary for H_n is:
        #   H1: dynamic_lower
        #   H_n (n>1): h{n-1}_h{n}
        # Upper boundary for H_n is:
        #   H_max: dynamic_upper
        #   H_n (n<max): h{n}_h{n+1}
        # Alpha is kept low to preserve spectrogram legibility.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        if self.show_bands.get() and self.valley_boundaries:
            for i, harm_num in enumerate(sorted_harm_nums):
                color = _HARMONIC_COLORS[(harm_num - 1) % len(_HARMONIC_COLORS)]

                # Determine lower boundary key for this harmonic
                if i == 0:
                    lower_key = 'dynamic_lower'
                else:
                    prev_num  = sorted_harm_nums[i - 1]
                    lower_key = f"h{prev_num}_h{harm_num}"

                # Determine upper boundary key for this harmonic
                if i == len(sorted_harm_nums) - 1:
                    upper_key = 'dynamic_upper'
                else:
                    next_num  = sorted_harm_nums[i + 1]
                    upper_key = f"h{harm_num}_h{next_num}"

                lower_data = self.valley_boundaries.get(lower_key)
                upper_data = self.valley_boundaries.get(upper_key)

                if lower_data and upper_data:
                    lower_times = np.array(lower_data['times'])
                    lower_freqs = np.array(lower_data['freqs'])
                    upper_times = np.array(upper_data['times'])
                    upper_freqs = np.array(upper_data['freqs'])

                    # Interpolate both to a common time grid for fill_between
                    common_times = self.times
                    lower_interp = np.interp(common_times, lower_times, lower_freqs)
                    upper_interp = np.interp(common_times, upper_times, upper_freqs)

                    self.ax.fill_between(
                        common_times,
                        lower_interp,
                        upper_interp,
                        alpha=_BAND_FILL_ALPHA,
                        color=color,
                        label=f"H{harm_num} band"
                    )

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # 2. RIDGE LINES
        # Time-varying center frequency of each harmonic — the detected ridge.
        # Drawn as dashed lines to distinguish from the static harmonic line.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        if self.show_ridges.get() and self.harmonic_ridges:
            for harm_num, ridge in self.harmonic_ridges.items():
                color = _HARMONIC_COLORS[(harm_num - 1) % len(_HARMONIC_COLORS)]
                self.ax.plot(
                    ridge['times'], ridge['freqs'],
                    color=color, linewidth=1,
                    alpha=_RIDGE_LINE_ALPHA, linestyle='--'
                )

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # 3. VALLEY BOUNDARY LINES
        # Explicit valley boundary positions — drawn only when show_valleys
        # is enabled, as they can clutter the display.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        if self.show_valleys.get() and self.valley_boundaries:
            valley_line_colors = ['cyan', 'magenta', 'lime', 'white', 'gold']
            for idx, (key, data) in enumerate(self.valley_boundaries.items()):
                color = valley_line_colors[idx % len(valley_line_colors)]
                self.ax.plot(
                    data['times'], data['freqs'],
                    color=color, linewidth=1,
                    alpha=_VALLEY_LINE_ALPHA, linestyle=':'
                )

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # 4. STATIC HARMONIC LINES
        # Horizontal lines at each harmonic's current frequency.
        # These are the draggable handles for manual correction.
        # Frequency label shown at right edge of plot.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        for h in self.harmonic_lines:
            color  = _HARMONIC_COLORS[(h['num'] - 1) % len(_HARMONIC_COLORS)]
            h['line'] = self.ax.axhline(
                h['freq'], color=color, linewidth=1,
                alpha=0.5, label=f"H{h['num']}"
            )
            self.ax.text(
                self.ax.get_xlim()[1],
                h['freq'],
                f"{h['freq']:.1f} Hz",
                color=color, fontsize=7,
                va='bottom', ha='right', alpha=0.8
            )

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # 5. CONTOUR LINES
        # Smoothed ridge representation — drawn on top of ridge lines.
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        if self.show_contour.get() and self.harmonic_contours:
            for harm_num, contour in self.harmonic_contours.items():
                color = _HARMONIC_COLORS[(harm_num - 1) % len(_HARMONIC_COLORS)]
                self.ax.plot(
                    contour['times'], contour['freqs'],
                    color=color, alpha=0.5, linewidth=1
                )

        # 6. Shared point annotations from API
        from yaaat.core.visualization import draw_shared_point_annotations
        draw_shared_point_annotations(self)

    ##    <(''<)  <( ' ' )>  (>'')>
    # MOUSE INTERACTION HOOKS
    ##    <(''<)  <( ' ' )>  (>'')>

    def on_custom_press(self, event):
        """Start dragging the nearest harmonic line on left-click.

        Searches harmonic_lines for the closest line within 100 Hz.
        If found, stores reference and drag start frequency.
        """
        if event.button != 1 or event.ydata is None:
            return False

        min_dist     = 100.0
        clicked_line = None

        for h in self.harmonic_lines:
            dist = abs(event.ydata - h['freq'])
            if dist < min_dist:
                min_dist     = dist
                clicked_line = h

        if clicked_line:
            self.selected_line = clicked_line
            self.drag_start_y  = clicked_line['freq']
            return True

        return False

    def on_custom_motion(self, event):
        """Update harmonic line frequency during drag."""
        if self.selected_line is None:
            return False
        if event.ydata is not None:
            self.selected_line['freq'] = event.ydata
            self.update_display()
        return True

    def on_custom_release(self, event):
        """Finalize drag — rescale harmonics if H1 moved, redetect ridges and valleys."""
        if self.selected_line is None:
            return False

        old_freq = self.drag_start_y
        new_freq = self.selected_line['freq']

        if abs(new_freq - old_freq) > 1.0:
            self.changes_made = True

            if self.selected_line['num'] == 1:
                # (つ -' _ '- )つ    (つ -' _ '- )つ
                # H1 drag rescales all other harmonics proportionally.
                # This preserves harmonic ratios when F0 is corrected manually.
                # detected_f0 is updated to the new H1 frequency.
                # (つ -' _ '- )つ    (つ -' _ '- )つ
                ratio            = new_freq / old_freq if old_freq > 0 else 1.0
                self.detected_f0 = new_freq
                for h in self.harmonic_lines:
                    if h is not self.selected_line:
                        h['freq'] *= ratio

            # Redetect ridges and valley boundaries at new harmonic positions
            if self.show_ridges.get():
                self.detect_harmonic_ridges()
            self._compute_valley_boundaries()

            if self.show_contour.get():
                self.compute_contours()

            self.update_display()
            logger.info("Moved H%d from %.1f to %.1f Hz",
                        self.selected_line['num'], old_freq, new_freq)

        self.selected_line = None
        self.drag_start_y  = None
        self.update_info()
        return True

    ##    <(''<)  <( ' ' )>  (>'')>
    # HARMONIC MANAGEMENT
    ##    <(''<)  <( ' ' )>  (>'')>

    def toggle_harmonic(self, harm_num):
        """Add or remove a harmonic by number. H1 cannot be removed."""
        if self.detected_f0 is None:
            messagebox.showinfo("No F0", "Detect F0 first")
            return

        exists = any(h['num'] == harm_num for h in self.harmonic_lines)

        if exists:
            if harm_num == 1:
                messagebox.showinfo("Cannot Remove", "Cannot remove H1 (fundamental)")
                return
            self.harmonic_lines = [h for h in self.harmonic_lines
                                   if h['num'] != harm_num]
            if harm_num in self.harmonic_ridges:
                del self.harmonic_ridges[harm_num]
        else:
            new_freq = self.detected_f0 * harm_num
            self.harmonic_lines.append({
                'freq': new_freq,
                'num':  harm_num,
                'line': None
            })
            self.harmonic_lines.sort(key=lambda h: h['num'])

            if self.show_ridges.get():
                self.detect_harmonic_ridges()

        # Recompute valley boundaries after harmonic set changes
        self._compute_valley_boundaries()

        self.update_button_states()
        self.update_display()
        self.update_info()
        self.changes_made = True

    def clear_all_harmonics(self):
        """Remove all harmonics except H1."""
        if len(self.harmonic_lines) <= 1:
            return
        if messagebox.askyesno("Clear", f"Remove {len(self.harmonic_lines)-1} harmonics?"):
            self.harmonic_lines  = [h for h in self.harmonic_lines if h['num'] == 1]
            self.harmonic_ridges = {1: self.harmonic_ridges.get(1, {})}
            self._compute_valley_boundaries()
            self.update_display()
            self.update_info()
            self.changes_made = True

    def update_button_states(self):
        """Sync toggle button appearance to current harmonic_lines state."""
        for harm_num, btn in self.harmonic_buttons.items():
            exists = any(h['num'] == harm_num for h in self.harmonic_lines)
            if exists:
                btn.config(relief=tk.SUNKEN, bg='lightgreen')
            else:
                btn.config(relief=tk.RAISED, bg='lightgray')

    ##    <(''<)  <( ' ' )>  (>'')>
    # PARAMETER CHANGE CALLBACKS
    ##    <(''<)  <( ' ' )>  (>'')>

    def on_prominence_change(self, value):
        """Update prominence label."""
        self.prom_label.config(text=f"{self.prominence.get():.1f}")

    def on_ridge_method_change(self):
        """Re-detect ridges with new method. Disable tolerance slider for non-peaks methods."""
        method = self.ridge_method.get()
        if hasattr(self, 'tol_scale'):
            state = 'normal' if method == 'peaks' else 'disabled'
            self.tol_scale.configure(state=state)

        if self.harmonic_lines:
            self.detect_harmonic_ridges()
            self._compute_valley_boundaries()
            if self.show_contour.get():
                self.compute_contours()
            self.update_display()

    def on_tolerance_change(self, value):
        """Re-detect ridges when tolerance slider changes (peaks method only)."""
        if hasattr(self, 'tol_label'):
            self.tol_label.config(text=f"{self.peak_tolerance.get():.2f}")
        if self.ridge_method.get() == 'peaks' and self.harmonic_lines:
            self.detect_harmonic_ridges()
            self._compute_valley_boundaries()
            if self.show_contour.get():
                self.compute_contours()
            self.update_display()

    def on_valley_method_change(self):
        """Recompute valley boundaries with new method."""
        self._compute_valley_boundaries()
        self.update_display()

    def on_show_contour_toggle(self):
        """Compute contours on first enable, then redraw."""
        if self.show_contour.get():
            self.compute_contours()
        self.update_display()

    def on_contour_method_change(self):
        """Recompute contours with new method."""
        if self.show_contour.get():
            self.compute_contours()
            self.update_display()

    def on_contour_smoothness_change(self, value):
        """Recompute contours when smoothness slider changes."""
        if self.show_contour.get():
            self.compute_contours()
            self.update_display()

    ##    <(''<)  <( ' ' )>  (>'')>
    # INFO AND LIST UPDATE
    ##    <(''<)  <( ' ' )>  (>'')>

    def update_info(self):
        """Update info label and harmonics listbox with current state."""
        if self.detected_f0 is None:
            self.info_label.config(text="No F0 detected")
        else:
            self.info_label.config(
                text=f"F0: {self.detected_f0:.1f} Hz\n"
                     f"Harmonics: {len(self.harmonic_lines)}\n"
                     f"Drag line to correct"
            )
        self._update_harmonics_listbox()

    def _update_harmonics_listbox(self):
        """Refresh the active harmonics listbox."""
        if hasattr(self, 'harmonics_listbox'):
            self.harmonics_listbox.delete(0, tk.END)
            for h in sorted(self.harmonic_lines, key=lambda x: x['num']):
                self.harmonics_listbox.insert(
                    tk.END, f"H{h['num']}: {h['freq']:.1f} Hz")

    ##    <(''<)  <( ' ' )>  (>'')>
    # SAVE / LOAD
    ##    <(''<)  <( ' ' )>  (>'')>

    def save_custom_data(self):
        """Save harmonic annotations to _harmonics.json via merge-write.

        Preserves any data written by other tabs to the same file.
        valley_boundaries, ridges, and contours all use parallel array format.
        valley_method is recorded so downstream consumers know how boundaries
        were computed — critical for comparative analysis against FuzzyValley.
        """
        if not self.audio_files or self.annotation_dir is None:
            return

        path = annotation_io.resolve_annotation_path(
            self.audio_files[self.current_file_idx],
            self.base_audio_dir,
            self.annotation_dir,
            annotation_io.SUFFIX_HARMONICS
        )

        # Serialize harmonic lines
        harmonic_data = [
            {
                'harmonic_num': h['num'],
                'frequency':    float(h['freq']),
            }
            for h in self.harmonic_lines
        ]

        # Serialize peak ratio result if available
        peak_ratio_data = None
        if self.peak_ratio_result is not None and _PEAK_RATIO_AVAILABLE:
            peak_ratio_data = {
                'peak_freqs':    self.peak_ratio_result['peak_freqs'].tolist(),
                'peak_amps':     self.peak_ratio_result['peak_amps'].tolist(),
                'relationships': self.peak_ratio_result['relationships'],
            }

        tab_data = {
            "detected_f0":       float(self.detected_f0) if self.detected_f0 else None,
            "harmonics":         harmonic_data,
            "ridges":            {
                str(k): v for k, v in self.harmonic_ridges.items()
            },
            "valley_boundaries": self.valley_boundaries,

            # valley_method recorded for downstream comparative analysis.
            # When FuzzyValley is integrated, this field distinguishes which
            # method produced the stored boundaries.
            "valley_method":     self.valley_method.get(),

            "contours":          {
                str(k): v for k, v in self.harmonic_contours.items()
            },
            "contour_method":    self.contour_method.get(),
            "peak_ratio_result": peak_ratio_data,
            "spec_params":       annotation_io.build_spec_params(
                self, orientation="horizontal"),
            "psd_params":        annotation_io.build_psd_params(self),
            "skip":              False,
            "skip_reason":       "",
        }

        annotation_io.merge_and_save(path, tab_data)
        self.changes_made = False
        logger.info("Saved harmonic annotations: %s", path.name)

    def load_custom_data(self):
        """Load harmonic annotations from _harmonics.json with param divergence check."""
        if not self.audio_files or self.annotation_dir is None:
            return

        path = annotation_io.resolve_annotation_path(
            self.audio_files[self.current_file_idx],
            self.base_audio_dir,
            self.annotation_dir,
            annotation_io.SUFFIX_HARMONICS
        )

        data = annotation_io.load_and_check_params(
            path, self, annotation_io.SUFFIX_HARMONICS, self.annotation_dir)

        if not data:
            return

        # Restore harmonic lines
        self.harmonic_lines = []
        for h in data.get('harmonics', []):
            self.harmonic_lines.append({
                'freq': h['frequency'],
                'num':  h['harmonic_num'],
                'line': None,
            })

        self.detected_f0 = data.get('detected_f0')

        # Restore ridges — keys stored as strings, convert back to int
        self.harmonic_ridges = {
            int(k): v for k, v in data.get('ridges', {}).items()
        }

        # Restore valley boundaries
        self.valley_boundaries = data.get('valley_boundaries', {})

        # Restore valley method — used to display which method was used
        saved_method = data.get('valley_method')
        if saved_method:
            self.valley_method.set(saved_method)

        # Restore contours
        self.harmonic_contours = {
            int(k): v for k, v in data.get('contours', {}).items()
        }

        # Restore contour method
        saved_contour_method = data.get('contour_method')
        if saved_contour_method:
            self.contour_method.set(saved_contour_method)

        # Restore display state from saved params
        saved_spec = data.get('spec_params', {})
        if saved_spec.get('show_ridges') is not None:
            self.show_ridges.set(bool(saved_spec['show_ridges']))
        if saved_spec.get('show_contour') is not None:
            self.show_contour.set(bool(saved_spec['show_contour']))

        # Recompute contours if enabled after load
        if self.show_contour.get() and self.harmonic_ridges:
            self.compute_contours()

        self.update_info()
        self.update_button_states()
        logger.info("Loaded %d harmonics from %s",
                    len(self.harmonic_lines), path.name)


##    <(''<)  <( ' ' )>  (>'')>

def main():
    """Launch HarmonicAnnotator as a standalone tab."""
    root = tk.Tk()
    app  = HarmonicAnnotator(root)
    root.geometry("1400x800")
    root.mainloop()


if __name__ == "__main__":
    main()

# U S A G I
# from yaaat.tabs.harmonic_annotator import HarmonicAnnotator
# root = tk.Tk(); app = HarmonicAnnotator(root); root.geometry("1400x800"); root.mainloop()