# Copyright (c) 2025 Ashlae Blum'e | laelume
# Licensed under the MIT License

"""YAAAT - Yet Another Audio Annotation Tool"""

from .changepoint_annotator import ChangepointAnnotator, main
from .peak_annotator import PeakAnnotator
from .main import YAATApp, main

__version__ = "0.1.0"
__author__ = "laelume"
__license__ = "MIT"
__all__ = ["ChangepointAnnotator", "PeakAnnotator", "YAATApp", "main"]
