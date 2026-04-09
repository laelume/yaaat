# yaaat/__init__.py

"""
YAAAT — Yet Another Audio Annotation Tool

Pip-installable audio annotation toolkit for bioacoustic signal processing.
Provides multi-tab Tkinter GUI for spectrogram-based annotation of animal
vocalizations.

Core tabs:
    BaseAnnotator        — spectrogram viewer
    ChangepointAnnotator — contour/changepoint annotation
    PeakAnnotator        — dual-resolution peak annotation
    HarmonicAnnotator    — harmonic detection and correction
    BinaryAnnotator      — grid-based binary dataset labeling

Companion library: pysoniq (audio I/O)
"""

__version__ = "0.2.0"
__author__  = "laelume"
__license__ = "MIT"

# (つ -' _ '- )つ    (つ -' _ '- )つ
# Minimal top-level exports — import from submodules for full API access.
# Keeping __init__.py thin avoids circular import issues and reduces
# startup overhead when only specific submodules are needed.
# (つ -' _ '- )つ    (つ -' _ '- )つ

from yaaat.main import YAAATApp, main

__all__ = [
    "YAAATApp",
    "main",
    "__version__",
    "__author__",
    "__license__",
]