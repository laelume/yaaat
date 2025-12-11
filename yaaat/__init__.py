# Copyright (c) 2025 laelume | Ashlae Blum'e 
# Licensed under the MIT License

"""YAAAT - Yet Another Audio Annotation Tool"""

from .tabs.changepoint_annotator import ChangepointAnnotator, main
from .tabs.peak_annotator import PeakAnnotator
from .tabs.harmonic_annotator import HarmonicAnnotator
from .tabs.sequence_annotator import SequenceAnnotator

from .layers.base_layer import BaseLayer
from .layers.harmonic_layer import HarmonicLayer

from .main import YAAATApp, main

__version__ = "0.1.13"
__author__ = "laelume"
__license__ = "MIT"
__all__ = [
    "ChangepointAnnotator", "PeakAnnotator", "HarmonicAnnotator", "SequenceAnnotator",  
    "BaseLayer", "HarmonicLayer", "ChangepointLayer", 
    "ContourExtractor", 
    "YAAATApp"
    ]