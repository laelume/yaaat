<<<<<<< HEAD
# Copyright (c) 2025 laelume | Ashlae Blum'e 
# Licensed under the MIT License

"""YAAAT - Yet Another Audio Annotation Tool"""
=======
# Copyright (c) 2025-2026 laelume | Ashlae Blum'e 
# Licensed under the MIT License

"""YAAAT! Yet Another Audio Annotation Tool"""
>>>>>>> origin/dev

from .tabs.changepoint_annotator import ChangepointAnnotator, main
from .tabs.peak_annotator import PeakAnnotator
from .tabs.harmonic_annotator import HarmonicAnnotator
from .tabs.sequence_annotator import SequenceAnnotator

from .layers.base_layer import BaseLayer
<<<<<<< HEAD
from .layers.harmonic_layer import HarmonicLayer

from .main import YAAATApp, main

__version__ = "0.1.13"
=======
from .layers.changepoint_layer import ChangepointLayer
from .layers.harmonic_layer import HarmonicLayer

# Omitted until ready to ship (see dev-local)
# from .layers.contour_layer import ContourLayer
# from .layers.binary_annotation_layer import BinaryAnnotationLayer


from .algs.peak_ratio_harmonics import analyze_harmonics, find_spectral_peaks

from .contours.processing import process_audio_file
from .contours.optimize import optimize_parameters, batch_optimize, evaluate_contour_quality

from .main import YAAATApp, main

__version__ = "0.1.12"
>>>>>>> origin/dev
__author__ = "laelume"
__license__ = "MIT"
__all__ = [
    "ChangepointAnnotator", "PeakAnnotator", "HarmonicAnnotator", "SequenceAnnotator",  
    "BaseLayer", "HarmonicLayer", "ChangepointLayer", 
<<<<<<< HEAD
    "ContourExtractor", 
    "YAAATApp"
    ]
=======
    "ContourLayer", 
    "BinaryAnnotationLayer", 
    "YAAATApp", 
    "analyze_harmonics", "find_spectral_peaks",
    "process_audio_file", "optimize_parameters", "batch_optimize", "evaluate_contour_quality",
]
>>>>>>> origin/dev
