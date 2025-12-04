# Copyright (c) 2025 laelume | Ashlae Blum'e 
# Licensed under the MIT License

"""YAAAT - Yet Another Audio Annotation Tool"""

from .changepoint_annotator import ChangepointAnnotator, main
from .peak_annotator import PeakAnnotator
from .harmonic_annotator import HarmonicAnnotator
from .sequence_annotator import SequenceAnnotator
from .main import YAAATApp, main

__version__ = "0.1.8"
__author__ = "laelume"
__license__ = "MIT"
__all__ = ["ChangepointAnnotator", "PeakAnnotator", "HarmonicAnnotator", "YAAATApp"]
