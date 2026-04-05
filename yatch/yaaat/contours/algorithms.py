"""Core signal processing algorithms for contour extraction."""

import numpy as np
from typing import Tuple, List, Dict


def detect_vertical_impulses(
    S_db: np.ndarray,
    freq_threshold: float = 0.4,
    intensity_threshold_db: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect vertical impulse frames in spectrogram.
    
    Impulse noise appears as vertical lines affecting many frequency bins
    simultaneously. Detection uses per-frame frequency activity ratio.
    
    Parameters
    ----------
    S_db : ndarray, shape (n_freqs, n_frames)
        Spectrogram in dB scale
    freq_threshold : float, default=0.4
        Proportion of frequency bins that must be active (0-1)
    intensity_threshold_db : float, default=10.0
        dB above median to consider a bin "active"
    
    Returns
    -------
    impulse_frames : ndarray
        Frame indices detected as impulses
    freq_activity_per_frame : ndarray
        Frequency activity ratio for each frame
    """
    impulse_frames = []
    freq_activity_per_frame = []
    median_per_freq = np.median(S_db, axis=1, keepdims=True)
    
    for t in range(S_db.shape[1]):
        column = S_db[:, t]
        active_bins = np.sum(column > (median_per_freq[:, 0] + intensity_threshold_db))
        freq_activity = active_bins / S_db.shape[0]
        freq_activity_per_frame.append(freq_activity)
        
        if freq_activity >= freq_threshold:
            impulse_frames.append(t)
    
    return np.array(impulse_frames), np.array(freq_activity_per_frame)


def group_impulse_events(
    impulse_frames: np.ndarray,
    max_gap: int = 3,
    expand: int = 2
) -> Tuple[List[int], List[List[int]]]:
    """
    Group nearby impulse frames into events and expand boundaries.
    
    Parameters
    ----------
    impulse_frames : ndarray
        Frame indices detected as impulses
    max_gap : int, default=3
        Maximum frame gap to group as same event
    expand : int, default=2
        Number of frames to expand on each side of event
    
    Returns
    -------
    all_frames_to_remove : list of int
        All frame indices to remove (expanded events merged)
    events : list of lists
        Grouped impulse events before expansion
    """
    if len(impulse_frames) == 0:
        return [], []
    
    events = []
    current_event = [impulse_frames[0]]
    
    for i in range(1, len(impulse_frames)):
        if impulse_frames[i] - impulse_frames[i-1] <= max_gap:
            current_event.append(impulse_frames[i])
        else:
            events.append(current_event)
            current_event = [impulse_frames[i]]
    
    events.append(current_event)
    
    all_frames_to_remove = []
    for event in events:
        start = max(0, min(event) - expand)
        end = max(event) + expand + 1
        all_frames_to_remove.extend(range(start, end))
    
    all_frames_to_remove = sorted(list(set(all_frames_to_remove)))
    
    return all_frames_to_remove, events


def remove_impulse_columns(
    spectrogram: np.ndarray,
    impulse_frames: List[int]
) -> np.ndarray:
    """
    Remove impulse columns by interpolating from neighboring clean frames.
    
    Parameters
    ----------
    spectrogram : ndarray, shape (n_freqs, n_frames)
        Input spectrogram
    impulse_frames : list of int
        Frame indices to remove
    
    Returns
    -------
    spec_clean : ndarray
        Spectrogram with impulses interpolated
    """
    spec_clean = spectrogram.copy()
    
    for t in impulse_frames:
        if t > 0 and t < spectrogram.shape[1] - 1:
            # Find nearest clean frames
            left = t - 1
            while left in impulse_frames and left > 0:
                left -= 1
            
            right = t + 1
            while right in impulse_frames and right < spectrogram.shape[1] - 1:
                right += 1
            
            if left >= 0 and right < spectrogram.shape[1]:
                spec_clean[:, t] = (spectrogram[:, left] + spectrogram[:, right]) / 2
                
        elif t == 0 and spectrogram.shape[1] > 1:
            spec_clean[:, t] = spectrogram[:, 1]
        elif t == spectrogram.shape[1] - 1 and t > 0:
            spec_clean[:, t] = spectrogram[:, t-1]
    
    return spec_clean


def deimpulse_spectrogram(
    S_db: np.ndarray,
    freq_threshold: float = 0.4,
    intensity_threshold_db: float = 10.0,
    max_gap: int = 3,
    expand: int = 2
) -> Tuple[np.ndarray, Dict]:
    """
    Complete pipeline: detect and remove impulse noise from spectrogram.
    
    Parameters
    ----------
    S_db : ndarray, shape (n_freqs, n_frames)
        Spectrogram in dB scale
    freq_threshold : float, default=0.4
        Proportion of frequency bins for impulse detection (0-1)
    intensity_threshold_db : float, default=10.0
        dB above median for active bin
    max_gap : int, default=3
        Max frames to group as same event
    expand : int, default=2
        Frames to expand on each side
    
    Returns
    -------
    S_db_clean : ndarray
        Denoised spectrogram
    impulse_info : dict
        Dictionary containing:
        - 'impulse_frames_detected': detected impulse frames
        - 'impulse_frames_to_remove': all frames removed (after grouping/expansion)
        - 'impulse_events': grouped events
        - 'freq_activity': frequency activity per frame
        - 'removed_impulses': difference (original - clean)
    """
    impulse_frames_detected, freq_activity = detect_vertical_impulses(
        S_db, 
        freq_threshold=freq_threshold, 
        intensity_threshold_db=intensity_threshold_db
    )
    
    impulse_frames_to_remove, impulse_events = group_impulse_events(
        impulse_frames_detected, 
        max_gap=max_gap, 
        expand=expand
    )
    
    S_db_clean = remove_impulse_columns(S_db, impulse_frames_to_remove)
    
    removed_impulses = S_db - S_db_clean
    
    impulse_info = {
        'impulse_frames_detected': impulse_frames_detected,
        'impulse_frames_to_remove': impulse_frames_to_remove,
        'impulse_events': impulse_events,
        'freq_activity': freq_activity,
        'removed_impulses': removed_impulses
    }
    
    return S_db_clean, impulse_info