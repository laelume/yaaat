"""
core/interaction.py

Standalone mouse and scroll event handling for YAAAT annotator tabs.
All functions receive the layer instance explicitly — no inheritance required.

Responsibilities:
    - Mouse press: right-click zoom undo, subclass hook, left-click bounding box-select init
    - Mouse motion: bounding box preview, subclass hook
    - Mouse release: bounding box commit or click passthrough, subclass hook
    - Scroll: horizontal zoom, vertical zoom, horizontal pan, vertical pan
      Modifier key detection is Windows-native via ctypes; falls back to
      matplotlib event.key for other platforms.

# bounding box drawn as a dashed yellow Rectangle patch during drag.
# Zoom stack stores (xlim, ylim) tuples for sequential undo via right-click.

Plain left-drag selects a time-frequency bounding box, committed on release
via layer.on_bounding_box_selected(t_min, t_max, f_min, f_max). interaction.py
performs no file I/O — the tab owns persistence. Zoom is no longer bound to
drag; it lives exclusively in scroll + modifiers (see on_scroll).

Selection box drawn as a solid cyan Rectangle patch during drag.
Zoom stack stores (xlim, ylim) tuples for sequential undo via right-click,
fed by scroll-zoom only.

All functions operate on:
    layer.ax              : matplotlib Axes
    layer.canvas          : FigureCanvasTkAgg
    layer.drag_start      : (x, y) tuple or None
    layer.drag_rect       : matplotlib Rectangle patch or None
    layer.zoom_stack      : list of (xlim, ylim) tuples
    layer.zoom_info_label : ttk.Label displaying bbox dimensions
"""

import sys
import logging
import traceback

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# (つ -' _ '- )つ    (つ -' _ '- )つ
# ZOOM THRESHOLDS — prevent accidental micro-zooms
# (つ -' _ '- )つ    (つ -' _ '- )つ

_MIN_BBOX_TIME_S  = 0.01   # seconds — minimum x-range for a valid bbox region
_MIN_BBOX_FREQ_HZ = 10.0   # Hz      — minimum y-range for a valid bbox region
_CLICK_THRESHOLD  = 0.05   # normalized drag distance below which release = click

# Zoom in/out factors for scroll events
_ZOOM_IN_FACTOR  = 0.8
_ZOOM_OUT_FACTOR = 1.25

# Pan step as a fraction of the current axis range
_PAN_STEP = 0.1


##    <(''<)  <( ' ' )>  (>'')>
# MOUSE PRESS
##    <(''<)  <( ' ' )>  (>'')>

def on_press(layer, event):
    """Handle mouse button press on the spectrogram canvas.

    Dispatch order:
        1. Ignore clicks outside the spectrogram axes.
        2. Right-click: pop zoom stack and restore previous view.
        3. Subclass hook: layer.on_custom_press(event) — consumed if returns True.
        4. Left-click: record drag_start for bounding box.

    Args:
        layer: BaseLayer instance
        event: matplotlib MouseEvent
    """
    if event.inaxes != layer.ax or event.xdata is None or event.ydata is None:
        return

    # Right-click — undo one zoom level
    if event.button == 3:
        if layer.zoom_stack:
            xlim, ylim = layer.zoom_stack.pop()
            layer.ax.set_xlim(xlim)
            layer.ax.set_ylim(ylim)
            layer.canvas.draw_idle()
        return

    # Subclass hook — tab-specific press handling takes priority
    if layer.on_custom_press(event):
        return

    # Left-click — begin drag for bounding box
    if event.button == 1:
        layer.drag_start = (event.xdata, event.ydata)


##    <(''<)  <( ' ' )>  (>'')>
# MOUSE MOTION
##    <(''<)  <( ' ' )>  (>'')>

def on_motion(layer, event):
    """Handle mouse motion during drag to draw the bounding box preview.

    Dispatch order:
        1. Ignore motion outside axes.
        2. Subclass hook: layer.on_custom_motion(event) — consumed if returns True.
        3. Draw or update the bounding box line.
        4. Update zoom_info_label with current bbox dimensions.

    Args:
        layer: BaseLayer instance
        event: matplotlib MouseEvent
    """
    if event.inaxes != layer.ax or event.xdata is None or event.ydata is None:
        return

    # Subclass hook
    if layer.on_custom_motion(event):
        return

    if layer.drag_start is None:
        return

    # Remove previous rectangle patch before drawing updated one
    if layer.drag_rect is not None:
        layer.drag_rect.remove()
        layer.drag_rect = None

    x0, y0 = layer.drag_start
    width  = event.xdata - x0
    height = event.ydata - y0

    # layer.drag_rect = layer.ax.add_patch(
    #     plt.Rectangle(
    #         (x0, y0), width, height,
    #         fill=False, edgecolor='yellow', linewidth=2, linestyle='--'
    #     )
    # )
    # Adding bounding box region support
    # Solid cyan box = region selection (was dashed yellow = zoom)
    layer.drag_rect = layer.ax.add_patch(
        plt.Rectangle(
            (x0, y0), width, height,
            fill=False, edgecolor='cyan', linewidth=2, linestyle='-'
        )
    )

    # Update dimension readout below the plot
    layer.zoom_info_label.config(
        text=f"Time: {abs(width):.3f}s | Freq: {abs(height):.1f} Hz"
    )

    layer.canvas.draw_idle()


##    <(''<)  <( ' ' )>  (>'')>
# MOUSE RELEASE
##    <(''<)  <( ' ' )>  (>'')>

def on_release(layer, event):
    """Handle mouse button release — commit bounding box or pass through as click.

    Dispatch order:
        1. Subclass hook: layer.on_custom_release(event) — consumed if returns True.
           Cleans up drag state before returning.
        2. Guard: ignore if drag_start is None.
        3. Guard: if release is outside axes, clean up and return.
        4. If drag distance < _CLICK_THRESHOLD: treat as click, clear drag state.
        5. If bbox too small (< _MIN_BBOX thresholds): ignore.
        # deprecated 6. Otherwise: push current limits onto zoom_stack, apply new limits.

    Args:
        layer: BaseLayer instance
        event: matplotlib MouseEvent
    """
    try:
        # Subclass hook — consumes event if tab handles it
        if layer.on_custom_release(event):
            _clear_drag(layer)
            return

        if layer.drag_start is None:
            return

        # Release outside axes — clean up without zooming
        if event.inaxes != layer.ax or event.xdata is None or event.ydata is None:
            layer.zoom_info_label.config(text="")
            _clear_drag(layer)
            layer.canvas.draw_idle()
            return

        x0, y0 = layer.drag_start
        x1, y1 = event.xdata, event.ydata

        _clear_drag(layer)

        drag_dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Small drag — treat as click, no zoom
        if drag_dist < _CLICK_THRESHOLD:
            layer.zoom_info_label.config(text="")
            return

        # new_xlim = sorted([x0, x1])
        # new_ylim = sorted([y0, y1])

        # x_range = new_xlim[1] - new_xlim[0]
        # y_range = new_ylim[1] - new_ylim[0]

        # # Reject micro-zooms
        # if x_range < _MIN_BBOX_TIME_S or y_range < _MIN_BBOX_FREQ_HZ:
        #     layer.zoom_info_label.config(text="")
        #     return

        # # Push current view onto stack before applying new limits
        # layer.zoom_stack.append((layer.ax.get_xlim(), layer.ax.get_ylim()))

        # layer.ax.set_xlim(new_xlim)
        # layer.ax.set_ylim(new_ylim)
        # layer.canvas.draw_idle()

        # layer.zoom_info_label.config(text="")

        # Modified boundaries for bounding box time-frequency regions
        t_min, t_max = sorted([x0, x1])
        f_min, f_max = sorted([y0, y1])

        t_range = t_max - t_min
        f_range = f_max - f_min

        # Reject micro-boxes — same thresholds that guarded micro-zooms
        if t_range < _MIN_BBOX_TIME_S or f_range < _MIN_BBOX_FREQ_HZ:
            layer.zoom_info_label.config(text="")
            return

        # Commit bbox to the tab via hook.
        # interaction.py does no file I/O — tab owns persistence.
        # Coordinates pre-sorted: t_min < t_max, f_min < f_max.
        layer.on_bounding_box_selected(t_min, t_max, f_min, f_max)

        layer.zoom_info_label.config(text="")




    except Exception as e:
        logger.error("ERROR in on_release: %s", e)
        logger.debug(traceback.format_exc())
        _clear_drag(layer)


##    <(''<)  <( ' ' )>  (>'')>
# SCROLL
##    <(''<)  <( ' ' )>  (>'')>

def on_scroll(layer, event):
    """Handle mouse wheel zoom and pan with modifier key detection.

    Modifier combinations (Windows via ctypes; other platforms via event.key):
        No modifier       : vertical pan
        Ctrl              : horizontal zoom centered on cursor
        Shift             : horizontal pan
        Ctrl + Shift      : vertical zoom centered on cursor

    Zoom factor:
        Scroll up   → _ZOOM_IN_FACTOR  (0.8)  — zoom in
        Scroll down → _ZOOM_OUT_FACTOR (1.25) — zoom out

    Pan step: _PAN_STEP fraction of current axis range.

    Args:
        layer: BaseLayer instance
        event: matplotlib ScrollEvent
    """
    try:
        if event.inaxes != layer.ax or event.xdata is None or event.ydata is None:
            return

        is_ctrl, is_shift = _get_modifier_keys(event)
        is_ctrlshift = is_ctrl and is_shift

        xlim = layer.ax.get_xlim()
        ylim = layer.ax.get_ylim()

        zoom_factor = _ZOOM_IN_FACTOR if event.button == 'up' else _ZOOM_OUT_FACTOR

        if is_ctrlshift:
            # Vertical zoom centered on cursor y position
            ydata = event.ydata
            y_range = (ylim[1] - ylim[0]) * zoom_factor
            y_ratio = (ydata - ylim[0]) / (ylim[1] - ylim[0])
            new_ylim = (
                ydata - y_range * y_ratio,
                ydata + y_range * (1 - y_ratio)
            )
            layer.ax.set_ylim(new_ylim)

        elif is_ctrl:
            # Horizontal zoom centered on cursor x position
            xdata = event.xdata
            x_range = (xlim[1] - xlim[0]) * zoom_factor
            x_ratio = (xdata - xlim[0]) / (xlim[1] - xlim[0])
            new_xlim = (
                xdata - x_range * x_ratio,
                xdata + x_range * (1 - x_ratio)
            )
            layer.ax.set_xlim(new_xlim)

        elif is_shift:
            # Horizontal pan
            x_range = xlim[1] - xlim[0]
            pan = x_range * _PAN_STEP
            if event.button == 'up':
                layer.ax.set_xlim(xlim[0] + pan, xlim[1] + pan)
            else:
                layer.ax.set_xlim(xlim[0] - pan, xlim[1] - pan)

        else:
            # Vertical pan (default scroll behavior)
            y_range = ylim[1] - ylim[0]
            pan = y_range * _PAN_STEP
            if event.button == 'up':
                layer.ax.set_ylim(ylim[0] + pan, ylim[1] + pan)
            else:
                layer.ax.set_ylim(ylim[0] - pan, ylim[1] - pan)

        layer.canvas.draw_idle()

    except Exception as e:
        logger.error("ERROR in on_scroll: %s", e)
        logger.debug(traceback.format_exc())


##    <(''<)  <( ' ' )>  (>'')>
# INTERNAL HELPERS
##    <(''<)  <( ' ' )>  (>'')>

def _clear_drag(layer):
    """Remove the bounding box patch and reset drag state.

    Safe to call even if drag_rect is None or already removed.

    Args:
        layer: BaseLayer instance
    """
    layer.drag_start = None

    if layer.drag_rect is not None:
        try:
            layer.drag_rect.remove()
        except Exception:
            pass
        layer.drag_rect = None


def _get_modifier_keys(event):
    """Detect Ctrl and Shift modifier key state.

    On Windows: uses ctypes.windll.user32.GetKeyState for reliable
    modifier detection independent of matplotlib's key tracking.
    On other platforms: falls back to event.key string matching.

    Args:
        event: matplotlib ScrollEvent or MouseEvent

    Returns:
        tuple(bool, bool) — (is_ctrl, is_shift)
    """
    if sys.platform == 'win32':
        import ctypes
        is_ctrl  = bool(ctypes.windll.user32.GetKeyState(0x11) & 0x8000)
        is_shift = bool(ctypes.windll.user32.GetKeyState(0x10) & 0x8000)
    else:
        key      = getattr(event, 'key', None)
        is_ctrl  = key == 'control'
        is_shift = key == 'shift'

    return is_ctrl, is_shift


# U S A G I
# from yaaat.core.interaction import on_press, on_motion, on_release, on_scroll
# canvas.mpl_connect('button_press_event',   lambda e: on_press(layer, e))
# canvas.mpl_connect('button_release_event', lambda e: on_release(layer, e))
# canvas.mpl_connect('motion_notify_event',  lambda e: on_motion(layer, e))
# canvas.mpl_connect('scroll_event',         lambda e: on_scroll(layer, e))