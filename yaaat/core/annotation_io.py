"""
core/annotation_io.py

Standalone annotation I/O functions for YAAAT annotator tabs.
All functions operate on explicit arguments — no inheritance required.

File structure per audio file (one file per tab, shared annotation directory):
    {prefix}_{stem}_changepoints.json   ← canonical geometry: contours, contour_metrics
    {prefix}_{stem}_peaks.json          ← peak annotations and PSD peaks
    {prefix}_{stem}_harmonics.json      ← harmonic lines, ridges, contours
    {prefix}_{stem}_binary.json         ← binary grid selections

All four files carry:
    spec_params   ← spectrogram parameters at time of save
    psd_params    ← PSD parameters at time of save
    skip          ← file-level skip flag (Option A: any tab can mark skip)
    skip_reason   ← string reason for skip

_changepoints.json is the canonical geometry file. Other files carry a
'contour_source' key pointing to the _changepoints filename for cross-tab
reference resolution.

Param divergence:
    On load, each file compares stored spec_params and psd_params against
    current layer state. Divergence triggers a non-blocking dialog and is
    logged to _param_mismatches.log in the annotation directory.
    Relevance: all four files check both spec_params and psd_params.
    orientation mismatch in spec_params is treated as a tab mismatch warning,
    not a param divergence.

Global point annotation schema:
    {
        "<audio_path>": {
            "global": [{"t": float, "f": float, "label": str, "scope": str}, ...],
            "<ClassName>": [{"t": float, "f": float, "label": str, "scope": str}, ...]
        }
    }
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# (つ -' _ '- )つ    (つ -' _ '- )つ
# FILE SUFFIX CONSTANTS
# Single definition point — all tabs import from here
# (つ -' _ '- )つ    (つ -' _ '- )つ

SUFFIX_CHANGEPOINTS = "changepoints"
SUFFIX_PEAKS        = "peaks"
SUFFIX_HARMONICS    = "harmonics"
SUFFIX_BINARY       = "binary"

# Suffix to orientation mapping — used for tab mismatch detection
_SUFFIX_ORIENTATION = {
    SUFFIX_CHANGEPOINTS: "horizontal",
    SUFFIX_PEAKS:        "vertical",
    SUFFIX_HARMONICS:    "horizontal",
    SUFFIX_BINARY:       "horizontal",
}

# Param mismatch log filename
_MISMATCH_LOG_FILENAME = "_param_mismatches.log"

# Global point annotations filename
_GLOBAL_POINT_ANNOTATION_FILENAME = "_global_point_annotations.json"


##    <(''<)  <( ' ' )>  (>'')>
# ANNOTATION PATH RESOLUTION
##    <(''<)  <( ' ' )>  (>'')>

def resolve_annotation_path(audio_file, base_audio_dir, annotation_dir, suffix):
    """Resolve the annotation file path for a given audio file and tab suffix.

    Constructs a flat filename from the relative subdirectory path and stem,
    preserving dataset structure without nesting directories.

    Convention:
        <annotation_dir>/<subdir_prefix>_<stem>_<suffix>.json
        where subdir_prefix collapses relative path separators to underscores.
        If audio file is directly in base_audio_dir, no prefix is prepended.

    Args:
        audio_file:     Path — absolute path to the audio file
        base_audio_dir: Path — root directory of the loaded dataset
        annotation_dir: Path — directory where annotation files are written
        suffix:         str  — tab suffix constant e.g. SUFFIX_CHANGEPOINTS

    Returns:
        Path — resolved annotation file path
    """
    audio_file     = Path(audio_file)
    base_audio_dir = Path(base_audio_dir)
    annotation_dir = Path(annotation_dir)

    relative_path = audio_file.relative_to(base_audio_dir).parent
    prefix = str(relative_path).replace('/', '_').replace('\\', '_')

    if prefix and prefix != '.':
        filename = f"{prefix}_{audio_file.stem}_{suffix}.json"
    else:
        filename = f"{audio_file.stem}_{suffix}.json"

    return annotation_dir / filename


##    <(''<)  <( ' ' )>  (>'')>
# PARAM BLOCK CONSTRUCTION
# Each tab builds its own spec_params and psd_params blocks for writing.
# These helpers construct the canonical dict from a layer instance.
##    <(''<)  <( ' ' )>  (>'')>

def build_spec_params(layer, orientation="horizontal"):
    """Build the spec_params dict from a layer instance.

    Args:
        layer:       BaseLayer instance
        orientation: str — 'horizontal' or 'vertical'

    Returns:
        dict — spec_params block
    """
    return {
        "n_fft":        layer.n_fft.get(),
        "hop_length":   layer.hop_length.get(),
        "fmin_calc":    layer.fmin_calc.get(),
        "fmax_calc":    layer.fmax_calc.get(),
        "fmin_display": layer.fmin_display.get(),
        "fmax_display": layer.fmax_display.get(),
        "orientation":  orientation,
    }


def build_psd_params(layer):
    """Build the psd_params dict from a layer instance.

    Uses BaseLayer standard PSD vars if tab-specific vars are not present.

    Args:
        layer: BaseLayer instance

    Returns:
        dict — psd_params block
    """
    # Peak annotator carries its own psd vars; fall back to base defaults
    n_fft     = getattr(layer, 'n_fft_psd',  None)
    hop       = getattr(layer, 'hop_psd',     None)
    fmin      = getattr(layer, 'fmin_calc',   None)
    fmax      = getattr(layer, 'fmax_calc',   None)

    return {
        "n_fft":      n_fft.get()  if n_fft  else layer.n_fft.get(),
        "hop_length": hop.get()    if hop    else layer.hop_length.get(),
        "fmin":       fmin.get()   if fmin   else layer.fmin_calc.get(),
        "fmax":       fmax.get()   if fmax   else layer.fmax_calc.get(),
    }


##    <(''<)  <( ' ' )>  (>'')>
# PARAM DIVERGENCE CHECKING
##    <(''<)  <( ' ' )>  (>'')>

def check_param_divergence(stored_params, current_params, param_type,
                           annotation_path, annotation_dir):
    """Compare stored params against current layer params and log divergence.

    Orientation mismatch in spec_params is treated as a tab mismatch warning,
    not a param divergence — logged separately, no dialog triggered.

    Args:
        stored_params:   dict — params loaded from file
        current_params:  dict — params from current layer state
        param_type:      str  — 'spec_params' or 'psd_params'
        annotation_path: Path — path to the annotation file being checked
        annotation_dir:  Path — annotation directory for mismatch log

    Returns:
        dict — {'diverged': bool, 'tab_mismatch': bool, 'differences': dict}
    """
    result = {
        'diverged':     False,
        'tab_mismatch': False,
        'differences':  {}
    }

    if not stored_params:
        return result

    for key, current_val in current_params.items():
        stored_val = stored_params.get(key)
        if stored_val is None:
            continue

        if key == 'orientation':
            # Orientation mismatch = tab mismatch, not param divergence
            if stored_val != current_val:
                result['tab_mismatch'] = True
                _log_mismatch(
                    annotation_dir,
                    annotation_path,
                    param_type,
                    key,
                    stored_val,
                    current_val,
                    is_tab_mismatch=True
                )
            continue

        if stored_val != current_val:
            result['diverged'] = True
            result['differences'][key] = {
                'stored':  stored_val,
                'current': current_val,
            }

    if result['diverged']:
        _log_mismatch(
            annotation_dir,
            annotation_path,
            param_type,
            list(result['differences'].keys()),
            {k: v['stored']  for k, v in result['differences'].items()},
            {k: v['current'] for k, v in result['differences'].items()},
            is_tab_mismatch=False
        )
        logger.warning(
            "Param divergence in %s [%s]: %s",
            annotation_path.name, param_type, result['differences']
        )

    return result


def _log_mismatch(annotation_dir, annotation_path, param_type,
                  keys, stored_vals, current_vals, is_tab_mismatch=False):
    """Append a param mismatch entry to the annotation directory mismatch log.

    Args:
        annotation_dir:  Path
        annotation_path: Path
        param_type:      str
        keys:            str or list — param key(s) that diverged
        stored_vals:     value or dict — stored value(s)
        current_vals:    value or dict — current value(s)
        is_tab_mismatch: bool — True if orientation mismatch
    """
    log_path = Path(annotation_dir) / _MISMATCH_LOG_FILENAME
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mismatch_type = "TAB_MISMATCH" if is_tab_mismatch else "PARAM_DIVERGENCE"

    entry = (
        f"[{timestamp}] {mismatch_type} | "
        f"file={annotation_path.name} | "
        f"param_type={param_type} | "
        f"keys={keys} | "
        f"stored={stored_vals} | "
        f"current={current_vals}\n"
    )

    try:
        with open(log_path, 'a') as f:
            f.write(entry)
    except Exception as e:
        logger.error("Failed to write mismatch log: %s", e)


def handle_param_divergence_dialog(layer, differences, param_type,
                                   stored_params, annotation_path):
    """Present a non-blocking dialog when param divergence is detected on load.

    Three options:
        Keep file params   — restore layer vars to stored values
        Keep current params — recompute on next save, overwrite stored
        Cancel             — leave both, flag title bar

    Args:
        layer:         BaseLayer instance
        differences:   dict — {key: {stored, current}}
        param_type:    str  — 'spec_params' or 'psd_params'
        stored_params: dict — full stored params block
        annotation_path: Path
    """
    import tkinter as tk
    from tkinter import messagebox

    diff_lines = "\n".join(
        f"  {k}: stored={v['stored']} | current={v['current']}"
        for k, v in differences.items()
    )

    message = (
        f"Param divergence detected in {annotation_path.name}\n"
        f"[{param_type}]\n\n"
        f"{diff_lines}\n\n"
        f"Keep file params → restores UI to stored values\n"
        f"Keep current params → overwrites on next save\n"
        f"Cancel → flag mismatch in title bar, take no action"
    )

    # Three-way dialog via askyesnocancel
    # Yes = keep file, No = keep current, Cancel = flag
    response = messagebox.askyesnocancel(
        "Param Divergence",
        message
    )

    if response is True:
        # Restore layer vars to stored values
        _restore_params_to_layer(layer, stored_params, param_type)
        logger.info("Restored %s from file: %s", param_type, annotation_path.name)

    elif response is False:
        # Keep current — will overwrite on next save
        logger.info(
            "Keeping current %s, will overwrite on next save: %s",
            param_type, annotation_path.name
        )

    else:
        # Flag in title bar
        if hasattr(layer, 'root') and hasattr(layer.root, 'title'):
            current_title = layer.root.title()
            if '⚠' not in current_title:
                layer.root.title(f"⚠ PARAM MISMATCH | {current_title}")
        logger.warning(
            "Param mismatch flagged, no action taken: %s",
            annotation_path.name
        )


def _restore_params_to_layer(layer, stored_params, param_type):
    """Restore layer tk vars to stored param values.

    Args:
        layer:        BaseLayer instance
        stored_params: dict — stored params block
        param_type:   str  — 'spec_params' or 'psd_params'
    """
    if param_type == 'spec_params':
        _set_if_present(layer, 'n_fft',        stored_params, 'n_fft',        int)
        _set_if_present(layer, 'hop_length',   stored_params, 'hop_length',   int)
        _set_if_present(layer, 'fmin_calc',    stored_params, 'fmin_calc',    int)
        _set_if_present(layer, 'fmax_calc',    stored_params, 'fmax_calc',    int)
        _set_if_present(layer, 'fmin_display', stored_params, 'fmin_display', int)
        _set_if_present(layer, 'fmax_display', stored_params, 'fmax_display', int)

    elif param_type == 'psd_params':
        # Peak annotator uses its own psd vars if present
        if hasattr(layer, 'n_fft_psd'):
            _set_if_present(layer, 'n_fft_psd', stored_params, 'n_fft', int)
        if hasattr(layer, 'hop_psd'):
            _set_if_present(layer, 'hop_psd', stored_params, 'hop_length', int)


def _set_if_present(layer, attr, params_dict, key, cast):
    """Set a tk var on layer if the key exists in params_dict.

    Args:
        layer:      BaseLayer instance
        attr:       str — attribute name on layer
        params_dict: dict
        key:        str — key in params_dict
        cast:       type — int or float
    """
    if key in params_dict and hasattr(layer, attr):
        try:
            getattr(layer, attr).set(cast(params_dict[key]))
        except Exception as e:
            logger.error("Failed to restore param %s: %s", attr, e)


##    <(''<)  <( ' ' )>  (>'')>
# GENERIC ANNOTATION FILE READ / WRITE
# All tabs use these for safe merge-write and checked load
##    <(''<)  <( ' ' )>  (>'')>

def load_annotation_file(path):
    """Load an annotation JSON file, returning an empty dict if not found.

    Args:
        path: Path — annotation file path

    Returns:
        dict — loaded annotation data, or empty dict
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load annotation file %s: %s", path, e)
        return {}


def save_annotation_file(path, data):
    """Write annotation data to JSON, creating parent directories if needed.

    Args:
        path: Path — annotation file path
        data: dict — annotation data to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug("Saved annotation file: %s", path)
    except Exception as e:
        logger.error("Failed to save annotation file %s: %s", path, e)


def merge_and_save(path, tab_data):
    """Merge tab_data into existing annotation file and write back.

    Reads existing content, updates only the keys present in tab_data,
    and writes the merged result. Preserves all keys from other tabs.

    Args:
        path:     Path — annotation file path
        tab_data: dict — keys and values owned by the calling tab
    """
    existing = load_annotation_file(path)
    existing.update(tab_data)
    save_annotation_file(path, existing)


def load_and_check_params(path, layer, suffix, annotation_dir):
    """Load annotation file and check for spec/psd param divergence.

    Triggers handle_param_divergence_dialog() if divergence is detected.
    Orientation mismatch is logged but does not trigger a dialog.

    Args:
        path:           Path — annotation file path
        layer:          BaseLayer instance
        suffix:         str  — tab suffix constant
        annotation_dir: Path — for mismatch log

    Returns:
        dict — loaded annotation data (may be empty)
    """
    data = load_annotation_file(path)
    if not data:
        return data

    current_spec = build_spec_params(
        layer, orientation=_SUFFIX_ORIENTATION.get(suffix, "horizontal"))
    current_psd  = build_psd_params(layer)

    # Check spec_params divergence
    stored_spec = data.get('spec_params', {})
    if stored_spec:
        spec_result = check_param_divergence(
            stored_spec, current_spec, 'spec_params', path, annotation_dir)

        if spec_result['diverged']:
            handle_param_divergence_dialog(
                layer, spec_result['differences'],
                'spec_params', stored_spec, path)

    # Check psd_params divergence
    stored_psd = data.get('psd_params', {})
    if stored_psd:
        psd_result = check_param_divergence(
            stored_psd, current_psd, 'psd_params', path, annotation_dir)

        if psd_result['diverged']:
            handle_param_divergence_dialog(
                layer, psd_result['differences'],
                'psd_params', stored_psd, path)

    return data


##    <(''<)  <( ' ' )>  (>'')>
# SKIP MANAGEMENT
# Skip is a file-level concern owned by the shared annotation system.
# Any tab can mark a file as skipped. Skip state lives in _changepoints.json.
##    <(''<)  <( ' ' )>  (>'')>

def mark_skip(changepoints_path, reason, layer):
    """Mark the current file as skipped in the _changepoints.json file.

    Merges skip state into existing changepoints file, preserving all
    other annotation data. Records spec_params and psd_params at skip time.

    Args:
        changepoints_path: Path — path to _changepoints.json
        reason:            str  — skip reason string
        layer:             BaseLayer instance
    """
    tab_data = {
        "skip":        True,
        "skip_reason": str(reason),
        "spec_params": build_spec_params(layer, orientation="horizontal"),
        "psd_params":  build_psd_params(layer),
    }
    merge_and_save(changepoints_path, tab_data)
    logger.info("Marked skip: %s | reason: %s", changepoints_path.name, reason)


def clear_skip(changepoints_path):
    """Clear skip state from the _changepoints.json file.

    Args:
        changepoints_path: Path — path to _changepoints.json
    """
    tab_data = {
        "skip":        False,
        "skip_reason": "",
    }
    merge_and_save(changepoints_path, tab_data)
    logger.info("Cleared skip: %s", changepoints_path.name)


def is_skipped(changepoints_path):
    """Check whether a file is marked as skipped.

    Reads only the skip key from _changepoints.json.
    Returns False if file does not exist.

    Args:
        changepoints_path: Path — path to _changepoints.json

    Returns:
        bool
    """
    data = load_annotation_file(changepoints_path)
    return bool(data.get('skip', False))


##    <(''<)  <( ' ' )>  (>'')>
# GLOBAL POINT ANNOTATION PERSISTENCE
##    <(''<)  <( ' ' )>  (>'')>

def save_global_point_annotations(global_point_annotations, annotation_dir):
    """Persist the global point annotation dict to disk.

    Writes to <annotation_dir>/_global_point_annotations.json.
    No-ops silently if annotation_dir is None.

    Args:
        global_point_annotations: dict — class-level shared annotation store
        annotation_dir:           Path or None
    """
    if not annotation_dir:
        return

    out = Path(annotation_dir) / _GLOBAL_POINT_ANNOTATION_FILENAME
    try:
        with open(out, 'w') as f:
            json.dump(global_point_annotations, f, indent=2)
        logger.debug("Saved global point annotations to %s", out)
    except Exception as e:
        logger.error("Failed to save global point annotations: %s", e)


def load_global_point_annotations(annotation_dir):
    """Load global point annotations from disk.

    Returns empty dict if file does not exist or cannot be parsed.

    Args:
        annotation_dir: Path or None

    Returns:
        dict — global point annotation store keyed by audio path
    """
    if not annotation_dir:
        return {}

    inp = Path(annotation_dir) / _GLOBAL_POINT_ANNOTATION_FILENAME
    if not inp.exists():
        return {}

    try:
        with open(inp, 'r') as f:
            data = json.load(f)
        logger.debug("Loaded global point annotations from %s", inp)
        return data
    except Exception as e:
        logger.error("Failed to load global point annotations: %s", e)
        return {}


def load_global_point_annotations_into(layer):
    """Load global point annotations from disk into the layer class-level store.

    Imports BaseLayer locally to avoid circular imports.

    Args:
        layer: BaseLayer instance
    """
    from yaaat.core.base_layer import BaseLayer
    BaseLayer.global_point_annotations = load_global_point_annotations(
        layer.annotation_dir)


##    <(''<)  <( ' ' )>  (>'')>
# POINT BUCKET ACCESS
##    <(''<)  <( ' ' )>  (>'')>

def get_point_bucket(global_point_annotations, audio_files,
                     current_file_idx, scope, class_name):
    """Return the annotation list for the current file and scope.

    Creates the bucket if it does not yet exist.

    Args:
        global_point_annotations: dict
        audio_files:              list of Path
        current_file_idx:         int
        scope:                    str — 'global' or 'class'
        class_name:               str — calling tab class name

    Returns:
        list or None
    """
    if not audio_files:
        return None

    audio_path = str(audio_files[current_file_idx])
    file_dict  = global_point_annotations.setdefault(audio_path, {})

    if scope == "global":
        return file_dict.setdefault("global", [])

    return file_dict.setdefault(class_name, [])


def add_annotation_point(global_point_annotations, audio_files,
                         current_file_idx, time_s, freq_hz,
                         label, scope, class_name):
    """Append a point annotation to the appropriate bucket.

    Args:
        global_point_annotations: dict
        audio_files:              list of Path
        current_file_idx:         int
        time_s:                   float
        freq_hz:                  float
        label:                    str or None
        scope:                    str
        class_name:               str
    """
    bucket = get_point_bucket(
        global_point_annotations, audio_files,
        current_file_idx, scope, class_name)

    if bucket is None:
        return

    bucket.append({
        "t":     float(time_s),
        "f":     float(freq_hz),
        "label": "" if label is None else str(label),
        "scope": scope,
    })

    logger.debug(
        "Added annotation point: t=%.3f f=%.1f label=%s scope=%s",
        time_s, freq_hz, label, scope)


##    <(''<)  <( ' ' )>  (>'')>
# SHARED POINT ANNOTATION RENDERING HELPER
##    <(''<)  <( ' ' )>  (>'')>

def get_shared_point_annotations_for_file(global_point_annotations,
                                          audio_files, current_file_idx,
                                          class_name):
    """Return global and class-scoped point annotations for the current file.

    Args:
        global_point_annotations: dict
        audio_files:              list of Path
        current_file_idx:         int
        class_name:               str

    Returns:
        tuple(list, list) — (global_points, class_points)
    """
    if not audio_files:
        return [], []

    audio_path   = str(audio_files[current_file_idx])
    file_dict    = global_point_annotations.get(audio_path, {})
    global_pts   = file_dict.get("global", [])
    class_pts    = file_dict.get(class_name, [])

    return global_pts, class_pts


##    <(''<)  <( ' ' )>  (>'')>
# CONTOUR METRICS COMPUTATION
# Computed at save time from contour geometry. Not stored during annotation.
# Called by changepoint_annotator.save_custom_data() before writing.
##    <(''<)  <( ' ' )>  (>'')>

def compute_contour_metrics(contours):
    """Compute per-contour metrics from contour geometry.

    Metrics are derived at save time and never stored during annotation.
    Onset and offset are determined by onset_idx and offset_idx in each
    contour dict.

    Args:
        contours: list of dicts — each with 'points', 'onset_idx', 'offset_idx'

    Returns:
        list of dicts — one metrics dict per contour
    """
    metrics = []

    for i, contour in enumerate(contours):
        points     = contour.get('points', [])
        onset_idx  = contour.get('onset_idx', 0)
        offset_idx = contour.get('offset_idx', len(points) - 1)

        if len(points) < 2:
            continue

        onset_time  = points[onset_idx]['time']
        offset_time = points[offset_idx]['time']
        all_freqs   = [p['freq'] for p in points]

        metrics.append({
            "contour_index":    i,
            "contour_id":       contour.get('id', f"c{i}"),
            "onset_time":       float(onset_time),
            "offset_time":      float(offset_time),
            "contour_duration": float(offset_time - onset_time),
            "frequency_min":    float(min(all_freqs)),
            "frequency_max":    float(max(all_freqs)),
            "frequency_spread": float(max(all_freqs) - min(all_freqs)),
            "num_points":       len(points),
        })

    return metrics


# U S A G I
# from yaaat.core.annotation_io import (
#     resolve_annotation_path, merge_and_save, load_and_check_params,
#     mark_skip, is_skipped, compute_contour_metrics,
#     SUFFIX_CHANGEPOINTS, SUFFIX_PEAKS, SUFFIX_HARMONICS, SUFFIX_BINARY
# )
# path = resolve_annotation_path(audio_file, base_dir, annotation_dir, SUFFIX_CHANGEPOINTS)
# merge_and_save(path, {"contours": [...], "spec_params": {...}})
# data = load_and_check_params(path, layer, SUFFIX_CHANGEPOINTS, annotation_dir)