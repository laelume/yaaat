"""
core/file_nav.py

Standalone file navigation functions for YAAAT annotator tabs.
All functions receive the layer instance explicitly — no inheritance required.

Functions operate on:
    layer.audio_files          : natsorted list of Path objects
    layer.current_file_idx     : int index into audio_files
    layer.annotation_dir       : Path to annotation output directory
    layer.base_audio_dir       : Path to root of loaded audio dataset
    layer.changes_made         : dirty flag
    layer.repeat_id            : after() timer id for continuous nav
    layer._syncing_tabs        : suppresses auto-save during tab sync
"""

import logging
import traceback
from pathlib import Path
import json

import numpy as np
import pysoniq
from natsort import natsorted
from tkinter import filedialog, messagebox

from yaaat.core import audio_utils
from yaaat.core import annotation_io
import time

logger = logging.getLogger(__name__)



# === === === === === === === ===
# verbosity flag — set False to silence debug prints for this loader
_VERBOSE_AUTOLOAD = True
# ==== ==== ==== ====



# (つ -' _ '- )つ    (つ -' _ '- )つ
# TEST AUDIO PATH — relative to package root
# (つ -' _ '- )つ    (つ -' _ '- )つ

_TEST_AUDIO_SUBPATH = ('test_files', 'test_audio', 'syllables', 'kiwi')


##    <(''<)  <( ' ' )>  (>'')>
# DIRECTORY LOADING
##    <(''<)  <( ' ' )>  (>'')>

def load_directory(layer):
    """Open a directory dialog, load .wav files, and prompt for annotation save location.

    Populates layer.audio_files, layer.base_audio_dir, layer.annotation_dir.
    Persists the chosen directory via save_last_directory().
    Falls back to default annotation path if user cancels the save location dialog.
    """
    directory = filedialog.askdirectory(title="Select Audio Directory")
    if not directory:
        return

    layer.audio_files = natsorted(Path(directory).rglob('*.wav'))
    layer.base_audio_dir = Path(directory)

    if not layer.audio_files:
        messagebox.showwarning("No Files", "No .wav files found in selected directory")
        return

    # Prompt user for annotation save location
    response = messagebox.askyesnocancel(
        "Annotation Save Location",
        "Where to save annotations?\n\n"
        "Yes    = Choose existing directory\n"
        "No     = Create new directory\n"
        "Cancel = Use default location"
    )

    dataset_name = Path(directory).name

    if response is True:
        # User chooses an existing directory
        save_dir = filedialog.askdirectory(title="Select Annotation Directory")
        if save_dir:
            layer.annotation_dir = Path(save_dir)
        else:
            # Dialog cancelled — fall back to default
            layer.annotation_dir = _default_annotation_dir(dataset_name)

    elif response is False:
        # User creates a new subdirectory inside a chosen parent
        save_dir = filedialog.askdirectory(title="Select Parent Directory")
        if save_dir:
            layer.annotation_dir = Path(save_dir) / f"{dataset_name}_annotations"
            layer.annotation_dir.mkdir(exist_ok=True)
        else:
            return

    else:
        # Cancel — use default
        layer.annotation_dir = _default_annotation_dir(dataset_name)

    layer.annotation_dir.mkdir(parents=True, exist_ok=True)

    # Load any previously saved shared annotations for this dataset
    annotation_io.load_global_point_annotations_into(layer)

    layer.current_file_idx = 0
    layer.load_current_file()

    logger.info("Loaded %d files from %s", len(layer.audio_files), directory)
    logger.info("Annotations will be saved to: %s", layer.annotation_dir)

    save_last_directory(layer.base_audio_dir)


def load_test_audio(layer):
    """Load bundled test audio files from the package test_files directory.

    Falls back gracefully if test files are not present.
    Populates layer.audio_files, layer.base_audio_dir, layer.annotation_dir.
    """
    # Resolve relative to this file's package location
    test_audio_dir = Path(__file__).parent.parent.joinpath(*_TEST_AUDIO_SUBPATH)

    if not test_audio_dir.exists():
        messagebox.showinfo("No Test Data", f"Test audio not found at:\n{test_audio_dir}")
        logger.warning("Test audio directory not found: %s", test_audio_dir)
        return

    layer.audio_files = natsorted(test_audio_dir.rglob('*.wav'))
    layer.base_audio_dir = test_audio_dir

    if not layer.audio_files:
        messagebox.showwarning("No Files", "No .wav files found in test directory")
        return

    # Default annotation directory for test audio
    layer.annotation_dir = Path.home() / "yaaat_annotations" / "test_audio"
    layer.annotation_dir.mkdir(parents=True, exist_ok=True)

    annotation_io.load_global_point_annotations_into(layer)

    layer.current_file_idx = 0
    layer.load_current_file()

    logger.info("Loaded %d test files", len(layer.audio_files))
    save_last_directory(layer.base_audio_dir)


def auto_load_directory(layer):
    """Auto-load the last used directory on startup, or fall back to test audio.

    Called via root.after() so the UI is fully rendered before disk access.
    """
    last_dir = load_last_directory()

    # D E B U G  — resolved last-directory pointer read from persistence
    if _VERBOSE_AUTOLOAD:
        logger.info("[autoload] last_dir resolved -> %r", last_dir)
        logger.info("[autoload] last_dir exists  -> %s",
                    bool(last_dir and last_dir.exists()))

    if last_dir and last_dir.exists():
        logger.info("Auto-loading: %s", last_dir)

        layer.audio_files = natsorted(last_dir.rglob('*.wav'))
        layer.base_audio_dir = last_dir

        # D E B U G  — wav discovery count under the resolved directory
        if _VERBOSE_AUTOLOAD:
            logger.info("[autoload] wav files found -> %d",
                        len(layer.audio_files))

        if layer.audio_files:
            dataset_name = last_dir.name
            layer.annotation_dir = _default_annotation_dir(dataset_name)

            # D E B U G  — annotation dir resolved from dataset name; log
            #              existence before mkdir to expose missing-from-
            #              last-time state
            if _VERBOSE_AUTOLOAD:
                logger.info("[autoload] annotation_dir -> %s (exists=%s)",
                            layer.annotation_dir,
                            layer.annotation_dir.exists())

            layer.annotation_dir.mkdir(parents=True, exist_ok=True)

            annotation_io.load_global_point_annotations_into(layer)
            
            layer.current_file_idx = 0

            # D E B U G  — handoff to per-file loader; pysoniq loader symbol
            #              failure surfaces inside load_current_file
            if _VERBOSE_AUTOLOAD:
                logger.info("[autoload] loading file idx 0 -> %s",
                            layer.audio_files[0])

            layer.load_current_file()
            return
    
    # F A L L B A C K  [FALLBACK:autoload-test-audio]
    # Reached when no persisted directory, the directory is gone, or it holds
    # no wav files. Temporary path until persistence validation is implemented.
    
    if _VERBOSE_AUTOLOAD:
        logger.info("[autoload] F A L L B A C K -> test audio "
                    "(last_dir missing or empty)")
    

    # No last directory or no files found — fall back to test audio
    load_test_audio(layer)


                
##    <(''<)  <( ' ' )>  (>'')>
# FILE LOADING
##    <(''<)  <( ' ' )>  (>'')>

# D E B U G  — stage timers; elapsed per stage exposes decode vs STFT cost
_t_load_start = time.perf_counter() if _VERBOSE_AUTOLOAD else None

def load_current_file(layer):
    """Load audio and annotations for layer.current_file_idx.

    If layer._skip_reload is True, only recomputes the spectrogram without
    resetting annotations or re-running detection — used for tab switching.
    """
    if not layer.audio_files:
        return

    audio_file = layer.audio_files[layer.current_file_idx]
    logger.info("Loading %s", audio_file.name)


    # D E B U G  — resolved file path and on-disk existence before any load;
    #              isolates missing-file from loader-symbol failure
    if _VERBOSE_AUTOLOAD:
        logger.info("[loadfile] idx %d -> %s", layer.current_file_idx, audio_file)
        logger.info("[loadfile] file exists -> %s", audio_file.is_file())
        logger.info("[loadfile] skip_reload -> %s",
                    getattr(layer, '_skip_reload', False))

    # (つ -' _ '- )つ    (つ -' _ '- )つ
    # TAB SWITCH PATH — recompute display only, preserve annotation state
    # (つ -' _ '- )つ    (つ -' _ '- )つ
    if getattr(layer, '_skip_reload', False):
        layer.compute_spectrogram()
        layer.update_display(recompute_spec=True)
        return

    # (つ -' _ '- )つ    (つ -' _ '- )つ
    # FULL LOAD PATH
    # (つ -' _ '- )つ    (つ -' _ '- )つ

    # Load mono float32 audio via pysoniq
    # D E B U G  — about to call loader symbol; if execution stops after the
    #              prior print and before the next, the loader call is the break
    if _VERBOSE_AUTOLOAD:
        logger.info("[loadfile] calling pysoniq.load_audio")

    layer.y, layer.sr = pysoniq.load_audio(str(audio_file))

    # D E B U G  — loader returned; shape and rate confirm a successful read
    #            — decode elapsed; marks end of disk-read + decode stage

    if _VERBOSE_AUTOLOAD:
        _t_decoded = time.perf_counter()
        logger.info("[loadfile] decode elapsed -> %.4f s",
                    _t_decoded - _t_load_start)
        logger.info("[loadfile] loaded y.shape=%s sr=%s",
                    getattr(layer.y, 'shape', None), layer.sr)

    if layer.y.ndim > 1:
        # Collapse stereo to mono — all audio must be mono throughout the pipeline
        layer.y = np.mean(layer.y, axis=1)

    # D E B U G  — spectrogram stage start
    if _VERBOSE_AUTOLOAD:
        _t_spec_start = time.perf_counter()

    layer.compute_spectrogram()

    # D E B U G  — STFT elapsed; compare against decode to rank the bottleneck
    if _VERBOSE_AUTOLOAD:
        logger.info("[loadfile] spectrogram elapsed -> %.4f s",
                    time.perf_counter() - _t_spec_start)

    layer.spec_image = None

    # Tab-specific annotation load (override in subclass)
    layer.load_custom_data()

    # Tab-specific detection/analysis (override in subclass)
    layer.process_audio()

    layer.changes_made = False
    layer.zoom_stack = []
    layer.update_display(recompute_spec=True)
    update_progress(layer)

    # D E B U G  — total full-load elapsed; sum of all stages for this file
    if _VERBOSE_AUTOLOAD:
        logger.info("[loadfile] total full-load elapsed -> %.4f s",
                    time.perf_counter() - _t_load_start)

##    <(''<)  <( ' ' )>  (>'')>
# NAVIGATION
##    <(''<)  <( ' ' )>  (>'')>

def next_file(layer):
    """Advance to the next file in the list, wrapping around.

    Auto-saves if changes exist and tab sync is not active.
    Resumes playback if loop was active before navigation.
    """
    if not layer.audio_files:
        return

    was_playing = pysoniq.is_looping()

    if layer.changes_made and not getattr(layer, '_syncing_tabs', False):
        layer.save_custom_data()
        annotation_io.save_global_point_annotations(
            type(layer).global_point_annotations, layer.annotation_dir)

    pysoniq.stop()

    layer.current_file_idx = (layer.current_file_idx + 1) % len(layer.audio_files)
    layer.load_current_file()

    if was_playing:
        layer.play_audio()


def previous_file(layer):
    """Go back to the previous file in the list, wrapping around.

    Auto-saves if changes exist and tab sync is not active.
    Resumes playback if loop was active before navigation.
    """
    if not layer.audio_files:
        return

    was_playing = pysoniq.is_looping()

    if layer.changes_made and not getattr(layer, '_syncing_tabs', False):
        layer.save_custom_data()
        annotation_io.save_global_point_annotations(
            type(layer).global_point_annotations, layer.annotation_dir)

    pysoniq.stop()

    layer.current_file_idx = (layer.current_file_idx - 1) % len(layer.audio_files)
    layer.load_current_file()

    if was_playing:
        layer.play_audio()


def jump_to_file(layer):
    """Jump to the file number currently entered in layer.file_number_entry.

    Shows a warning dialog for invalid input. Auto-saves if changes exist.
    """
    from tkinter import messagebox
    try:
        file_num = int(layer.file_number_entry.get())

        if 1 <= file_num <= len(layer.audio_files):
            if layer.changes_made and not getattr(layer, '_syncing_tabs', False):
                layer.save_custom_data()

            layer.current_file_idx = file_num - 1
            layer.load_current_file()

        else:
            messagebox.showwarning(
                "Invalid File Number",
                f"Enter a number between 1 and {len(layer.audio_files)}"
            )
            update_progress(layer)

    except ValueError:
        messagebox.showwarning("Invalid Input", "Enter a valid integer")
        update_progress(layer)


##    <(''<)  <( ' ' )>  (>'')>
# CONTINUOUS NAVIGATION (hold-to-repeat)
##    <(''<)  <( ' ' )>  (>'')>

def start_continuous_nav(layer, direction):
    """Begin hold-to-repeat navigation. First step fires immediately.

    Schedules continue_nav() via after() with an initial 300ms delay.
    """
    if direction == 'next':
        next_file(layer)
    else:
        previous_file(layer)

    layer.repeat_id = layer.root.after(300, continue_nav, layer, direction)


def continue_nav(layer, direction):
    """Continue hold-to-repeat navigation at 150ms intervals."""
    if direction == 'next':
        next_file(layer)
    else:
        previous_file(layer)

    layer.repeat_id = layer.root.after(150, continue_nav, layer, direction)


def stop_continuous_nav(layer):
    """Cancel the hold-to-repeat navigation timer on button release."""
    if layer.repeat_id:
        layer.root.after_cancel(layer.repeat_id)
        layer.repeat_id = None


##    <(''<)  <( ' ' )>  (>'')>
# PROGRESS DISPLAY
##    <(''<)  <( ' ' )>  (>'')>

def update_progress(layer):
    """Update file number entry, total label, and filename label in the nav bar.

    Guards against being called before widgets are fully initialized.
    """
    if not hasattr(layer, 'file_number_entry') or not layer.file_number_entry.winfo_exists():
        logger.warning("update_progress called before widget ready in %s",
                       type(layer).__name__)
        return

    layer.file_number_entry.delete(0, 'end')
    layer.file_number_entry.insert(0, str(layer.current_file_idx + 1))
    layer.file_total_label.config(text=f"/ {len(layer.audio_files)}")
    layer.file_label.config(text=layer.audio_files[layer.current_file_idx].name)


##    <(''<)  <( ' ' )>  (>'')>
# INTERNAL HELPERS
##    <(''<)  <( ' ' )>  (>'')>

def _default_annotation_dir(dataset_name):
    """Return the default annotation directory path for a given dataset name.

    Default location: ~/yaaat_annotations/<dataset_name>
    """
    return Path.home() / "yaaat_annotations" / dataset_name



##    <(''<)  <( ' ' )>  (>'')>
# DIRECTORY PERSISTENCE
# Save and load the last-used audio directory path.
# Stored under 'last_directory' key in ~/.yaaat_config.json.
# Called after any successful directory load so the next startup
# auto-loads the same dataset without a dialog.
##    <(''<)  <( ' ' )>  (>'')>

def save_last_directory(directory):
    """Save last opened directory to ~/.yaaat_config.json."""
    config_file = Path.home() / '.yaaat_config.json'
    try:
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        config['last_directory'] = str(directory)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.error("Could not save last directory: %s", e)


def load_last_directory():
    """Load last opened directory from ~/.yaaat_config.json.

    Returns:
        Path if valid directory exists, None otherwise
    """
    config_file = Path.home() / '.yaaat_config.json'
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            last_dir = config.get('last_directory', '')
            if last_dir:
                last_dir = Path(last_dir)
                if last_dir.exists() and last_dir.is_dir():
                    return last_dir
    except Exception as e:
        logger.error("Could not load last directory: %s", e)
    return None

##    <(''<)  <( ' ' )>  (>'')>
# MANIFEST PERSISTENCE
# Save and load the last-used inference review manifest path.
# Stored under 'last_manifest' key in ~/.yaaat_config.json alongside
# 'last_directory'. Called after pipeline writes a new manifest, or
# after user manually selects one via InferenceReviewLayer file dialog.
#
# TODO: Wire save_last_manifest() call into the full pipeline manifest
# output step so the inference review layer can auto-locate the most
# recent manifest on startup. The pipeline step that generates the
# manifest JSON should call this immediately after writing to disk.
##    <(''<)  <( ' ' )>  (>'')>

def save_last_manifest(manifest_path):
    """Save the last used inference review manifest path to the YAAAT config file.

    Stores path under 'last_manifest' in ~/.yaaat_config.json,
    alongside 'last_directory'. Preserves all other config keys.

    Args:
        manifest_path: Path or str — path to the manifest JSON file
    """
    config_file = Path.home() / '.yaaat_config.json'
    try:
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)

        config['last_manifest'] = str(manifest_path)

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

    except Exception as e:
        logger.error("Could not save manifest path to config: %s", e)


def load_last_manifest():
    """Load the last used inference review manifest path from the YAAAT config file.

    Reads 'last_manifest' from ~/.yaaat_config.json.
    Returns None if config does not exist, key is absent,
    or the file no longer exists on disk.

    Returns:
        Path pointing to the manifest JSON file, or None
    """
    config_file = Path.home() / '.yaaat_config.json'
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)

            last_manifest = config.get('last_manifest', '')
            if last_manifest:
                last_manifest = Path(last_manifest)
                if last_manifest.exists() and last_manifest.is_file():
                    return last_manifest

    except Exception as e:
        logger.error("Could not load manifest path from config: %s", e)

    return None



# U S A G I

# from yaaat.core.file_nav import load_directory, layer.load_current_file, next_file
# load_directory(layer)
# layer.load_current_file()
# next_file(layer)

# from yaaat.core.file_nav import save_last_manifest, load_last_manifest
# save_last_manifest(Path('/path/to/manifest.json'))
# manifest_path = load_last_manifest()

