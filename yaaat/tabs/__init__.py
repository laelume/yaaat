# yaaat/tabs/__init__.py

"""
tabs/__init__.py

Public interface for the yaaat.tabs package.
Exports all core annotator tab classes.

Import from here for tab registration in main.py or
for external code that needs to subclass a tab.
"""

from yaaat.tabs.base_annotator        import BaseAnnotator
from yaaat.tabs.changepoint_annotator import ChangepointAnnotator
from yaaat.tabs.peak_annotator        import PeakAnnotator
from yaaat.tabs.harmonic_annotator    import HarmonicAnnotator
from yaaat.tabs.batch_annotator      import BatchAnnotator

__all__ = [
    "BaseAnnotator",
    "ChangepointAnnotator",
    "PeakAnnotator",
    "HarmonicAnnotator",
    "BatchAnnotator",
]