"""
core/__init__.py

Public interface for the yaaat.core package.
Exposes the primary classes and the most commonly imported utilities.

Import map:
    BaseLayer       — root parent class for all annotator tabs
    GridLayer       — grid layout parent class for multi-file tabs
    audio_utils     — signal processing functions (spectrogram, PSD, mel)
    file_nav        — filesystem and navigation functions
    annotation_io   — annotation persistence and path resolution
    visualization   — rendering functions
    interaction     — mouse and scroll event handlers
"""

from yaaat.core.base_layer   import BaseLayer
from yaaat.core.grid_layer   import GridLayer

from yaaat.core import audio_utils
from yaaat.core import file_nav
from yaaat.core import annotation_io
from yaaat.core import visualization
from yaaat.core import interaction

__all__ = [
    # Classes
    "BaseLayer",
    "GridLayer",
    # Modules — imported as namespaces, not flattened
    "audio_utils",
    "file_nav",
    "annotation_io",
    "visualization",
    "interaction",
]