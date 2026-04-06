# peak_ratio_harmonics.py 
# uses ratios of prominent harmonics to compare candidates for fundamental frequency. 



"""
Harmonic analysis using peak ratio detection and polar coordinate transforms.

This module provides template-free harmonic detection by analyzing spectral peaks
and their integer ratio relationships, plus polar coordinate transformations for
exploring spectrotemporal patterns relative to detected fundamentals.

Functions for YAAAT integration:
- analyze_harmonics: Main pipeline for detecting fundamental and harmonic series
- find_spectral_peaks: Adaptive peak detection with frequency-scaled prominence
- cartesian_to_logpolar_spectrogram: Transform for radial/angular pattern analysis
"""

import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks


def find_spectral_peaks(S_db, freqs, f_min=300, f_max=None, prominence_base=10, 
                       prominence_scale='log'):
    """
    Find prominent peaks with frequency-scaled threshold.
    
    Higher frequencies receive lower prominence thresholds, matching the natural
    decay of harmonic amplitudes in biological vocalizations.
    
    Parameters
    ----------
    S_db : ndarray, shape (n_freqs, n_times)
        Power spectrogram in dB
    freqs : ndarray, shape (n_freqs,)
        Frequency bins in Hz
    f_min : float
        Minimum frequency to consider
    f_max : float or None
        Maximum frequency to consider (defaults to freqs[-1])
    prominence_base : float
        Base prominence threshold at f_min in dB
    prominence_scale : str
        Scaling method: 'log', 'linear', or 'none'
        
    Returns
    -------
    peak_freqs : ndarray
        Frequencies of detected peaks in Hz
    peak_amps : ndarray
        Amplitudes of detected peaks in dB
    """
    spectrum = np.mean(S_db, axis=1)
    
    if f_max is None:
        f_max = freqs[-1]
    
    valid_idx = (freqs >= f_min) & (freqs <= f_max)
    freqs_valid = freqs[valid_idx]
    spectrum_valid = spectrum[valid_idx]
    
    # Scale prominence by frequency
    if prominence_scale == 'log':
        # Logarithmic: prom = base * log(f_max/f) / log(f_max/f_min)
        prominence_threshold = prominence_base * np.log(f_max / freqs_valid) / np.log(f_max / f_min)
        prominence_threshold = np.maximum(prominence_threshold, 2)  # Floor at 2 dB
        
    elif prominence_scale == 'linear':
        # Linear decay
        prominence_threshold = prominence_base * (f_max - freqs_valid) / (f_max - f_min)
        prominence_threshold = np.maximum(prominence_threshold, 2)
        
    else:  # 'none'
        prominence_threshold = np.full_like(freqs_valid, prominence_base)
    
    # Find peaks with scipy, then filter by adaptive threshold
    peaks, properties = scipy_find_peaks(spectrum_valid, distance=5)
    
    # Filter by adaptive prominence
    valid_peaks = []
    for peak_idx in peaks:
        # Get local baseline
        window = 5
        i_min = max(0, peak_idx - window)
        i_max = min(len(spectrum_valid), peak_idx + window + 1)
        local_min = np.min(spectrum_valid[i_min:i_max])
        
        prominence = spectrum_valid[peak_idx] - local_min
        if prominence >= prominence_threshold[peak_idx]:
            valid_peaks.append(peak_idx)
    
    valid_peaks = np.array(valid_peaks)
    peak_freqs = freqs_valid[valid_peaks]
    peak_amps = spectrum_valid[valid_peaks]
    
    return peak_freqs, peak_amps


def compute_harmonic_score(f_candidate, peak_freqs, max_harmonic=20, tolerance=0.05):
    """
    Count how many detected peaks are explained by this fundamental.
    
    A peak is "explained" if it's close to n*f_candidate for integer n.
    
    Parameters
    ----------
    f_candidate : float
        Candidate fundamental frequency in Hz
    peak_freqs : ndarray
        Detected peak frequencies in Hz
    max_harmonic : int
        Maximum harmonic number to consider
    tolerance : float
        Relative tolerance for harmonic matching (e.g., 0.05 = 5%)
        
    Returns
    -------
    score : int
        Number of peaks explained by this fundamental
    matches : list of tuples
        List of (harmonic_number, peak_frequency) for each match
    """
    matches = []
    explained_peaks = set()
    
    # For each detected peak, check if it's close to ANY harmonic of f_candidate
    for peak_freq in peak_freqs:
        # What harmonic number would this peak be?
        n = peak_freq / f_candidate
        
        # Check nearby integers
        for n_test in [np.floor(n), np.ceil(n)]:
            if n_test < 1 or n_test > max_harmonic:
                continue
            
            f_expected = n_test * f_candidate
            
            # Is this peak within tolerance of this harmonic?
            if np.abs(peak_freq - f_expected) / f_expected < tolerance:
                if peak_freq not in explained_peaks:
                    matches.append((int(n_test), peak_freq))
                    explained_peaks.add(peak_freq)
                    break
    
    return len(matches), matches


def find_all_harmonic_relationships(peak_freqs, tolerance=0.05):
    """
    Find all pairwise harmonic relationships between peaks.
    
    Parameters
    ----------
    peak_freqs : ndarray
        Detected peak frequencies in Hz
    tolerance : float
        Relative tolerance for integer ratio matching
        
    Returns
    -------
    relationships : list of dict
        Each dict contains:
        - f_low: lower frequency
        - f_high: higher frequency
        - ratio: nearest integer ratio
        - ratio_exact: exact ratio value
        - error: relative error from integer
    """
    relationships = []
    
    for i, f1 in enumerate(peak_freqs):
        for j, f2 in enumerate(peak_freqs):
            if i >= j:  # Avoid duplicates and self-comparison
                continue
            
            ratio = f2 / f1
            nearest_int = round(ratio)
            
            if nearest_int > 0 and abs(ratio - nearest_int) / nearest_int < tolerance:
                relationships.append({
                    'f_low': f1,
                    'f_high': f2,
                    'ratio': nearest_int,
                    'ratio_exact': ratio,
                    'error': abs(ratio - nearest_int) / nearest_int
                })
    
    return relationships


def analyze_harmonics(S_db, freqs, f_min=300, f_max=None, prominence_base=10,
                      prominence_scale='log', max_harmonic=None, tolerance=0.05):
    """
    Complete harmonic analysis pipeline using peak ratio detection.
    
    This is the main entry point for YAAAT integration. It detects spectral peaks,
    evaluates each as a potential fundamental, and finds all harmonic relationships.
    
    Parameters
    ----------
    S_db : ndarray, shape (n_freqs, n_times)
        Power spectrogram in dB
    freqs : ndarray, shape (n_freqs,)
        Frequency bins in Hz
    f_min : float
        Minimum frequency for peak detection
    f_max : float or None
        Maximum frequency for peak detection
    prominence_base : float
        Base prominence threshold in dB
    prominence_scale : str
        'log', 'linear', or 'none'
    max_harmonic : int or None
        Maximum harmonic number to test (defaults to auto)
    tolerance : float
        Relative tolerance for harmonic matching
        
    Returns
    -------
    dict with keys:
        - peak_freqs: detected peak frequencies
        - peak_amps: peak amplitudes
        - candidates: list of fundamental candidates with scores
        - relationships: all pairwise harmonic relationships
        - best_candidate: top-ranked fundamental candidate (or None)
    """
    # Find spectral peaks
    peak_freqs, peak_amps = find_spectral_peaks(S_db, freqs, f_min, f_max, 
                                                prominence_base, prominence_scale)
    
    # Auto-determine max_harmonic if not specified
    if max_harmonic is None:
        if len(peak_freqs) > 0:
            max_harmonic = int(peak_freqs[-1] / peak_freqs[0]) + 5
        else:
            max_harmonic = 20
    
    # Test each peak as potential fundamental
    candidates = []
    for i, f_test in enumerate(peak_freqs):
        score, matches = compute_harmonic_score(f_test, peak_freqs, max_harmonic, tolerance)
        candidates.append({
            'f0': f_test,
            'score': score,
            'matches': matches,
            'amplitude': peak_amps[i]
        })
    
    # Rank purely by score (number of peaks explained)
    candidates_sorted = sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    # Find all pairwise relationships
    relationships = find_all_harmonic_relationships(peak_freqs, tolerance)
    
    return {
        'peak_freqs': peak_freqs,
        'peak_amps': peak_amps,
        'candidates': candidates_sorted,
        'relationships': relationships,
        'best_candidate': candidates_sorted[0] if len(candidates_sorted) > 0 else None
    }


def cartesian_to_logpolar_spectrogram(S_db, freqs, times, f_center, t_center=None,
                                     n_log_radii=256, n_angles=360):
    """
    Transform spectrogram to log-polar coordinates centered on a frequency.
    
    This reveals radial and angular patterns relative to the fundamental frequency.
    Harmonic structures become radial lines, and temporal modulations become
    angular patterns.
    
    Parameters
    ----------
    S_db : ndarray, shape (n_freqs, n_times)
        Power spectrogram in dB
    freqs : ndarray, shape (n_freqs,)
        Frequency bins in Hz
    times : ndarray, shape (n_times,)
        Time bins in seconds
    f_center : float
        Center frequency (typically the detected fundamental)
    t_center : float or None
        Center time (defaults to middle of time axis)
    n_log_radii : int
        Number of radial bins (log-spaced)
    n_angles : int
        Number of angular bins
        
    Returns
    -------
    S_logpolar : ndarray, shape (n_log_radii, n_angles)
        Transformed spectrogram in log-polar space
    log_radii : ndarray, shape (n_log_radii,)
        Log-radius values
    angles : ndarray, shape (n_angles,)
        Angle values in radians
    """
    if t_center is None:
        t_center = times[len(times)//2]
    
    # Normalize to [0, 1] for easier polar transform
    f_norm = (freqs - freqs.min()) / (freqs.max() - freqs.min())
    t_norm = (times - times.min()) / (times.max() - times.min())
    f_center_norm = (f_center - freqs.min()) / (freqs.max() - freqs.min())
    t_center_norm = (t_center - times.min()) / (times.max() - times.min())
    
    # Define log-polar grid
    min_radius = 0.01
    max_radius = np.sqrt(
        max((1 - f_center_norm)**2, f_center_norm**2) + 
        max((1 - t_center_norm)**2, t_center_norm**2)
    )
    log_radii = np.linspace(np.log(min_radius), np.log(max_radius), n_log_radii)
    angles = np.linspace(0, 2*np.pi, n_angles)
    
    S_logpolar = np.zeros((n_log_radii, n_angles))
    
    # Sample original spectrogram in polar coordinates
    for i, log_r in enumerate(log_radii):
        r = np.exp(log_r)
        for j, theta in enumerate(angles):
            # Convert polar to normalized cartesian
            f_n = f_center_norm + r * np.cos(theta)
            t_n = t_center_norm + r * np.sin(theta)
            
            # Convert back to original frequency/time
            f = f_n * (freqs.max() - freqs.min()) + freqs.min()
            t = t_n * (times.max() - times.min()) + times.min()
            
            # Sample if within bounds
            if freqs.min() <= f <= freqs.max() and times.min() <= t <= times.max():
                f_idx = np.argmin(np.abs(freqs - f))
                t_idx = np.argmin(np.abs(times - t))
                S_logpolar[i, j] = S_db[f_idx, t_idx]
    
    return S_logpolar, log_radii, angles


def analyze_angular_symmetry(S_logpolar, angles):
    """
    Detect rotational patterns in polar space using angular autocorrelation.
    
    Useful for identifying tremolo/vibrato and periodic frequency modulation.
    Based on circular statistics (Mardia & Jupp, 2000).
    
    Parameters
    ----------
    S_logpolar : ndarray, shape (n_log_radii, n_angles)
        Polar-transformed spectrogram
    angles : ndarray, shape (n_angles,)
        Angle bins in radians
        
    Returns
    -------
    autocorr : ndarray, shape (n_angles,)
        Angular autocorrelation (normalized)
    """
    angular_profile = np.mean(S_logpolar, axis=0)
    autocorr = np.correlate(angular_profile, angular_profile, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    return autocorr / autocorr[0]


def compute_radial_flow(S_logpolar):
    """
    Measure energy flow toward/away from fundamental using radial gradient.
    
    Positive values indicate energy increasing with radius (upward frequency sweep),
    negative values indicate energy decreasing (downward sweep or harmonic decay).
    Based on optical flow methods (Beauchemin & Barron, 1995).
    
    Parameters
    ----------
    S_logpolar : ndarray, shape (n_log_radii, n_angles)
        Polar-transformed spectrogram
        
    Returns
    -------
    radial_gradient : ndarray, shape (n_log_radii,)
        Average radial gradient across all angles
    """
    radial_gradient = np.gradient(S_logpolar, axis=0)
    return np.mean(radial_gradient, axis=1)


def harmonic_alignment_score(S_logpolar, expected_harmonic_angles):
    """
    Quantify how well harmonics align at expected angles in polar space.
    
    Based on template matching in polar coordinates (Wolberg & Zokai, 2000).
    
    Parameters
    ----------
    S_logpolar : ndarray, shape (n_log_radii, n_angles)
        Polar-transformed spectrogram
    expected_harmonic_angles : list of int
        Angle indices where harmonics should appear
        
    Returns
    -------
    score : float
        Mean maximum amplitude at expected harmonic angles
    """
    scores = []
    for angle_idx in expected_harmonic_angles:
        radial_slice = S_logpolar[:, angle_idx]
        scores.append(np.max(radial_slice))
    return np.mean(scores)


def compute_harmonic_ratio_matrix(peak_freqs, tolerance=0.05):
    """
    Create matrix of harmonic ratios between all peak pairs.
    
    Useful for visualizing harmonic consistency and detecting complex
    multi-fundamental structures.
    
    Parameters
    ----------
    peak_freqs : ndarray
        Detected peak frequencies in Hz
    tolerance : float
        Relative tolerance for integer ratio detection
        
    Returns
    -------
    ratio_matrix : ndarray, shape (n_peaks, n_peaks)
        Matrix where element [i,j] contains integer ratio if peaks i and j
        are harmonically related, otherwise 0
    """
    n_peaks = len(peak_freqs)
    ratio_matrix = np.zeros((n_peaks, n_peaks))
    
    for i in range(n_peaks):
        for j in range(n_peaks):
            if i != j:
                ratio = peak_freqs[j] / peak_freqs[i]
                nearest_int = round(ratio)
                if nearest_int > 0 and abs(ratio - nearest_int) / nearest_int < tolerance:
                    ratio_matrix[i, j] = nearest_int
    
    return ratio_matrix