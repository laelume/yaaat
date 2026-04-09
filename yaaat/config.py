"""
config.py

Global configuration and tunable defaults for YAAAT.

All hardcoded default values that appear in BaseLayer, annotator tabs,
and audio_utils are defined here as a single CONFIG dict.

Tabs and modules import from config.py rather than hardcoding values.
Override at runtime by modifying CONFIG before instantiating any tab.

TODO: Add YAML/JSON config file loading so CONFIG can be overridden
per-project without modifying source. Defer until distribution targets
are confirmed.
"""

# (つ -' _ '- )つ    (つ -' _ '- )つ
# from jellyfish.utils.jelly_funcs import make_daily_directory
# daily_dir = make_daily_directory()
# (つ -' _ '- )つ    (つ -' _ '- )つ

CONFIG = {

    ##    <(''<)  <( ' ' )>  (>'')>
    # SPECTROGRAM DEFAULTS
    ##    <(''<)  <( ' ' )>  (>'')>

    # Default FFT size — balances frequency and temporal resolution
    "n_fft":        256,

    # Default hop length — 25% of n_fft for reasonable overlap
    "hop_length":   64,

    # Default frequency computation range in Hz
    "fmin_calc":    500,
    "fmax_calc":    16000,

    # Default frequency display range in Hz
    "fmin_display": 500,
    "fmax_display": 8000,

    # Default scale — 'linear' or 'mel'
    "y_scale":      "linear",

    # Default number of mel bands when scale='mel'
    "n_mels":       256,

    ##    <(''<)  <( ' ' )>  (>'')>
    # PSD DEFAULTS
    ##    <(''<)  <( ' ' )>  (>'')>

    "n_fft_psd":    1024,
    "hop_psd":      512,

    ##    <(''<)  <( ' ' )>  (>'')>
    # PEAK ANNOTATOR SPECTROGRAM DEFAULTS
    # Separate from base spectrogram defaults — peak annotator uses
    # higher temporal resolution (smaller hop) for vertical display.
    ##    <(''<)  <( ' ' )>  (>'')>

    "n_fft_spect":  512,
    "hop_spect":    32,

    ##    <(''<)  <( ' ' )>  (>'')>
    # PLAYBACK DEFAULTS
    ##    <(''<)  <( ' ' )>  (>'')>

    "playback_gain": 1.0,

    ##    <(''<)  <( ' ' )>  (>'')>
    # WAVEFORM OVERLAY DEFAULTS
    ##    <(''<)  <( ' ' )>  (>'')>

    "waveform_alpha": 0.2,

    ##    <(''<)  <( ' ' )>  (>'')>
    # GRID ANNOTATOR DEFAULTS
    ##    <(''<)  <( ' ' )>  (>'')>

    # Default number of files per grid page
    "grid_size":           25,

    # Highpass filter cutoff for grid spectrogram computation
    "grid_highpass_hz":    800,

    # Number of mel bands for grid spectrograms
    "grid_n_mels":         64,

    ##    <(''<)  <( ' ' )>  (>'')>
    # HARMONIC ANNOTATOR DEFAULTS
    ##    <(''<)  <( ' ' )>  (>'')>

    "harmonic_prominence":    5.0,
    "harmonic_freq_min":      500,
    "harmonic_freq_max":      8000,
    "harmonic_peak_tolerance": 0.1,
    "harmonic_ridge_method":  "max",
    "harmonic_valley_method": "min_energy",
    "harmonic_contour_method": "raw",
    "harmonic_contour_smoothness": 5.0,

    ##    <(''<)  <( ' ' )>  (>'')>
    # CHANGEPOINT ANNOTATOR DEFAULTS
    ##    <(''<)  <( ' ' )>  (>'')>

    # Minimum points required to finish a contour
    "changepoint_min_points": 2,

    # Click-to-remove threshold in seconds
    "changepoint_time_thresh_s":  0.05,

    # Click-to-remove threshold in Hz
    "changepoint_freq_thresh_hz": 100.0,

    ##    <(''<)  <( '' )>  (>'')>
    # PEAK ANNOTATOR DEFAULTS
    ##    <(''<)  <( ' ' )>  (>'')>

    "peak_prominence":         0.1,
    "peak_click_threshold_hz": 100.0,

    ##    <(''<)  <( ' ' )>  (>'')>
    # ANNOTATION PATHS
    ##    <(''<)  <( ' ' )>  (>'')>

    # Default annotation root under home directory
    "annotation_root": "yaaat_annotations",

    # Config file name stored in home directory
    "config_filename": ".yaaat_config.json",

    ##    <(''<)  <( ' ' )>  (>'')>
    # DISPLAY
    ##    <(''<)  <( ' ' )>  (>'')>

    "spectrogram_cmap":          "magma",
    "spectrogram_interpolation": "bilinear",
    "window_width":              1400,
    "window_height":             800,
    "window_min_width":          900,
    "window_min_height":         600,
}