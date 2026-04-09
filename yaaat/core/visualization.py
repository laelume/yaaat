"""
core/visualization.py

Standalone rendering functions for YAAAT annotator tabs.
All functions receive the layer instance explicitly — no inheritance required.

Responsibilities:
    - Full spectrogram redraw (imshow + axis labels + ylim)
    - Overlay-only redraw (removes patches, lines, collections, text)
    - Waveform overlay rendering on twin y-axis
    - Shared point annotation rendering
    - Frequency display range update
    - Zoom reset

All functions operate on:
    layer.ax            : matplotlib Axes
    layer.canvas        : FigureCanvasTkAgg
    layer.fig           : matplotlib Figure
    layer.S_db          : spectrogram array (freq x time)
    layer.freqs         : frequency array
    layer.times         : time array
    layer.y             : audio signal
    layer.sr            : sample rate
    layer.spec_image    : cached AxesImage or None
    layer.waveform_ax   : twin y-axis for waveform or None
    layer.zoom_stack    : list of (xlim, ylim) tuples
"""

import logging
import traceback

import numpy as np
import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt

import tkinter as tk

from yaaat.core import audio_utils
from yaaat.core import annotation_io

logger = logging.getLogger(__name__)


# (つ -' _ '- )つ    (つ -' _ '- )つ
# SPECTROGRAM COLORMAP — single definition point for consistency across all tabs
# (つ -' _ '- )つ    (つ -' _ '- )つ

_SPEC_CMAP = 'magma'
_SPEC_INTERPOLATION = 'bilinear'


##    <(''<)  <( ' ' )>  (>'')>
# PRIMARY DISPLAY UPDATE
##    <(''<)  <( ' ' )>  (>'')>

def update_display(layer, recompute_spec=False):
    """Redraw the spectrogram panel.

    Two paths:
        recompute_spec=True  — full redraw: clears axis, redraws spectrogram image,
                               resets axis labels and ylim. Expensive.
        recompute_spec=False — overlay-only redraw: removes scatter, patches, lines,
                               and text; preserves the cached spectrogram image.
                               Used for annotation updates without parameter changes.

    Always calls layer.draw_custom_overlays() and conditionally draws the waveform.
    Updates the plot title to reflect save state and current parameters.

    Args:
        layer:          BaseLayer instance
        recompute_spec: bool — True forces full spectrogram redraw
    """
    try:
        if layer.y is None:
            return

        if recompute_spec or layer.spec_image is None:
            _full_redraw(layer)
        else:
            _overlay_only_redraw(layer)

        # Tab-specific overlays — defined in each tab subclass
        layer.draw_custom_overlays()

        # Waveform overlay — toggled by checkbox in base UI
        if layer.show_waveform.get():
            draw_waveform(layer)
        elif layer.waveform_ax is not None:
            _remove_waveform_ax(layer)

        # Title reflects save state and active parameters
        _update_title(layer)

        layer.canvas.draw()

    except Exception as e:
        logger.error("ERROR in update_display: %s", e)
        logger.debug(traceback.format_exc())


##    <(''<)  <( ' ' )>  (>'')>
# FULL REDRAW
##    <(''<)  <( ' ' )>  (>'')>

def _full_redraw(layer):
    """Clear the axis and redraw the spectrogram image from scratch.

    Removes any existing waveform twin axis before clearing.
    Resets axis labels and display frequency limits.

    Args:
        layer: BaseLayer instance
    """
    # Remove waveform twin axis before clearing — avoids orphaned axes
    if layer.waveform_ax is not None:
        _remove_waveform_ax(layer)

    layer.ax.clear()

    extent = [
        layer.times[0],
        layer.times[-1],
        layer.freqs[0],
        layer.freqs[-1]
    ]

    layer.spec_image = layer.ax.imshow(
        layer.S_db,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap=_SPEC_CMAP,
        interpolation=_SPEC_INTERPOLATION
    )

    layer.ax.set_xlabel('Time (s)', fontsize=8)

    # Y-axis label reflects active scale
    if layer.y_scale.get() == 'mel':
        layer.ax.set_ylabel('Frequency (mel)', fontsize=8)
    else:
        layer.ax.set_ylabel('Frequency (Hz)', fontsize=8)

    # Apply display frequency limits
    ymin, ymax = layer._convert_ylim_to_scale(
        layer.fmin_display.get(), layer.fmax_display.get())
    layer.ax.set_ylim(ymin, ymax)


##    <(''<)  <( ' ' )>  (>'')>
# OVERLAY-ONLY REDRAW
##    <(''<)  <( ' ' )>  (>'')>

def _overlay_only_redraw(layer):
    """Remove all overlays from the axis without touching the spectrogram image.

    Removes: scatter collections, patches, lines, text artists.
    Clears waveform twin axis contents if present.
    Preserves layer.spec_image and axis limits.

    Args:
        layer: BaseLayer instance
    """
    # Remove scatter plot collections
    collections_to_remove = [
        c for c in layer.ax.collections
        if isinstance(c, matplotlib.collections.PathCollection)
    ]
    for c in collections_to_remove:
        c.remove()

    # Remove bounding boxes, rectangles, ellipses, polygons
    for patch in layer.ax.patches[:]:
        patch.remove()

    # Remove guide lines and contour lines
    for line in layer.ax.lines[:]:
        line.remove()

    # Remove text annotations (guide labels, stats overlays)
    for text in layer.ax.texts[:]:
        text.remove()

    # Clear waveform contents without removing the twin axis
    if layer.waveform_ax is not None:
        layer.waveform_ax.cla()


##    <(''<)  <( ' ' )>  (>'')>
# WAVEFORM OVERLAY
##    <(''<)  <( ' ' )>  (>'')>

def draw_waveform(layer):
    """Draw raw waveform and 5ms smoothed envelope on a twin y-axis.

    Creates layer.waveform_ax if it does not exist.
    Clears and redraws if it already exists.
    Raw waveform: cyan. Smoothed envelope: yellow.
    Smoothing window: 5ms (sr * 0.005 samples), box convolution.

    Args:
        layer: BaseLayer instance
    """
    if layer.y is None or layer.sr is None:
        return

    # Create twin axis on first call; reuse and clear on subsequent calls
    if layer.waveform_ax is None:
        layer.waveform_ax = layer.ax.twinx()
        # Offset right spine to avoid overlap with frequency axis ticks
        layer.waveform_ax.spines['right'].set_position(('outward', 40))
        layer.waveform_ax.yaxis.set_label_position('right')
        layer.waveform_ax.yaxis.set_ticks_position('right')
    else:
        layer.waveform_ax.cla()

    t = np.arange(len(layer.y)) / layer.sr

    alpha = float(np.clip(layer.waveform_alpha.get(), 0.0, 1.0))

    # Raw waveform
    layer.waveform_ax.plot(t, layer.y, color='cyan', alpha=alpha, linewidth=0.6)

    # 5ms smoothed envelope via box convolution
    window = max(1, int(layer.sr * 0.005))
    if window > 1 and len(layer.y) >= window:
        smooth = np.convolve(layer.y, np.ones(window) / window, mode='same')
        layer.waveform_ax.plot(
            t, smooth, color='yellow', linewidth=1,
            alpha=min(1.0, alpha + 0.3))

    layer.waveform_ax.set_ylabel("Amplitude", fontsize=8)
    layer.waveform_ax.tick_params(axis='y', labelsize=7)
    layer.waveform_ax.grid(False)


def _remove_waveform_ax(layer):
    """Remove the waveform twin axis from the figure.

    Sets layer.waveform_ax to None after removal.
    Suppresses errors if the axis has already been removed.

    Args:
        layer: BaseLayer instance
    """
    try:
        layer.waveform_ax.remove()
    except Exception:
        pass
    layer.waveform_ax = None


##    <(''<)  <( ' ' )>  (>'')>
# SHARED POINT ANNOTATION RENDERING
##    <(''<)  <( ' ' )>  (>'')>

def draw_shared_point_annotations(layer):
    """Draw global and class-scoped shared point annotations onto layer.ax.

    Global points: white hollow circles.
    Class-scoped points: cyan hollow circles.
    Labels rendered as small text above each point if non-empty.

    Called from tab subclasses via draw_custom_overlays() where needed.

    Args:
        layer: BaseLayer instance
    """
    if not layer.audio_files:
        return

    from yaaat.core.base_layer import BaseLayer

    global_points, class_points = annotation_io.get_shared_point_annotations_for_file(
        BaseLayer.global_point_annotations,
        layer.audio_files,
        layer.current_file_idx,
        layer.__class__.__name__
    )

    # Global scope — white hollow markers
    for ann in global_points:
        layer.ax.scatter(ann["t"], ann["f"], s=30,
                         edgecolors="white", facecolors="none")
        if ann["label"]:
            layer.ax.text(ann["t"], ann["f"], ann["label"],
                          fontsize=7, color="white", va="bottom", ha="left")

    # Class scope — cyan hollow markers
    for ann in class_points:
        layer.ax.scatter(ann["t"], ann["f"], s=30,
                         edgecolors="cyan", facecolors="none")
        if ann["label"]:
            layer.ax.text(ann["t"], ann["f"], ann["label"],
                          fontsize=7, color="cyan", va="bottom", ha="left")


##    <(''<)  <( ' ' )>  (>'')>
# DISPLAY RANGE AND ZOOM
##    <(''<)  <( ' ' )>  (>'')>

def update_display_range(layer):
    """Apply current fmin/fmax display vars to the y-axis without recomputing.

    Args:
        layer: BaseLayer instance
    """
    if layer.y is None:
        return

    ymin, ymax = layer._convert_ylim_to_scale(
        layer.fmin_display.get(), layer.fmax_display.get())
    layer.ax.set_ylim(ymin, ymax)
    layer.canvas.draw_idle()


def reset_zoom(layer):
    """Reset view to full audio extent and current display frequency limits.

    Clears the zoom stack.

    Args:
        layer: BaseLayer instance
    """
    if layer.y is None:
        return

    layer.zoom_stack = []

    full_xlim = (0, len(layer.y) / layer.sr)
    ymin, ymax = layer._convert_ylim_to_scale(
        layer.fmin_display.get(), layer.fmax_display.get())

    layer.ax.set_xlim(full_xlim)
    layer.ax.set_ylim(ymin, ymax)
    layer.canvas.draw_idle()


##    <(''<)  <( ' ' )>  (>'')>
# TITLE
##    <(''<)  <( ' ' )>  (>'')>

def _update_title(layer):
    """Set the plot title to show save state, file path context, and active parameters.

    Format: <save_marker> <grandparent> | <parent> | <filename> | n_fft=N hop=H

    Save marker:
        '✓ ' — no unsaved changes
        ''   — unsaved changes exist

    Args:
        layer: BaseLayer instance
    """
    if not layer.audio_files:
        return

    audio_file = layer.audio_files[layer.current_file_idx]
    filename      = audio_file.name
    parent_dir    = audio_file.parent.name
    grandparent   = audio_file.parent.parent.name

    save_marker = "" if layer.changes_made else "✓ "

    layer.ax.set_title(
        f"{save_marker}{grandparent} | {parent_dir} | {filename} | "
        f"n_fft={layer.n_fft.get()} hop={layer.hop_length.get()}",
        fontsize=9
    )


# U S A G I
# from yaaat.core.visualization import update_display, draw_waveform, draw_shared_point_annotations
# update_display(layer, recompute_spec=True)
# draw_waveform(layer)
# draw_shared_point_annotations(layer)