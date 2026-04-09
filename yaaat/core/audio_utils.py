"""
core/audio_utils.py

Signal processing utilities for YAAAT annotator tabs.
No filesystem access, no tkinter dependencies, no side effects.

Responsibilities:
    - Mel scale conversion
    - Mel filterbank construction
    - Unified spectrogram computation (linear and mel)
    - Welch PSD computation
    - Frame-to-time conversion

All functions are stateless and operate on explicit numpy array arguments.

Dependencies:
    numpy, scipy.signal only — no librosa, no pysoniq, no tkinter
"""

import logging

import numpy as np
from scipy.signal import spectrogram, welch

logger = logging.getLogger(__name__)


# (つ -' _ '- )つ    (つ -' _ '- )つ
# MEL SCALE CONSTANTS
# HTK mel formula: m = 2595 * log10(1 + f/700)
# (つ -' _ '- )つ    (つ -' _ '- )つ

_MEL_BREAK_HZ   = 700.0
_MEL_SCALE      = 2595.0


##    <(''<)  <( ' ' )>  (>'')>
# MEL SCALE CONVERSION
##    <(''<)  <( ' ' )>  (>'')>

def hz_to_mel(hz):
    """Convert frequency in Hz to mel scale using the HTK formula.

    Args:
        hz: float or np.ndarray — frequency in Hz

    Returns:
        float or np.ndarray — frequency in mel
    """
    return _MEL_SCALE * np.log10(1 + hz / _MEL_BREAK_HZ)


def mel_to_hz(mel):
    """Convert mel scale value to frequency in Hz using the HTK formula.

    Args:
        mel: float or np.ndarray — frequency in mel

    Returns:
        float or np.ndarray — frequency in Hz
    """
    return _MEL_BREAK_HZ * (10 ** (mel / _MEL_SCALE) - 1)


##    <(''<)  <( ' ' )>  (>'')>
# MEL FILTERBANK
##    <(''<)  <( ' ' )>  (>'')>

def create_mel_filterbank(sr, n_fft, n_mels=128, fmin=0, fmax=None):
    """Construct a triangular mel filterbank matrix.

    Filterbank is built over mel-spaced center frequencies between
    fmin and fmax. Each filter is a triangular window in linear frequency.

    Args:
        sr:     int   — sample rate in Hz
        n_fft:  int   — FFT size; determines number of frequency bins
        n_mels: int   — number of mel filter bands
        fmin:   float — minimum frequency in Hz
        fmax:   float or None — maximum frequency in Hz; defaults to sr/2

    Returns:
        mel_basis: np.ndarray shape (n_mels, n_fft//2 + 1) — filterbank matrix
        mel_freqs: np.ndarray shape (n_mels,) — center frequency of each mel band in Hz
    """
    if fmax is None:
        fmax = sr / 2.0

    # Mel-spaced points including two boundary points
    mel_min    = hz_to_mel(fmin)
    mel_max    = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = mel_to_hz(mel_points)

    # Linear FFT bin frequencies
    fft_freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)

    mel_basis = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(n_mels):
        left   = hz_points[i]
        center = hz_points[i + 1]
        right  = hz_points[i + 2]

        # Rising slope from left to center
        rising  = np.clip((fft_freqs - left)   / (center - left),  0.0, 1.0)
        # Falling slope from center to right
        falling = np.clip((right - fft_freqs)  / (right - center), 0.0, 1.0)

        mel_basis[i] = rising * falling

    # Center frequencies exclude the two boundary points
    mel_freqs = hz_points[1:-1]

    return mel_basis, mel_freqs


def apply_mel_scale(S, mel_basis):
    """Apply a mel filterbank matrix to a linear magnitude spectrogram.

    Args:
        S:          np.ndarray shape (n_fft//2+1, n_frames) — linear magnitude spectrogram
        mel_basis:  np.ndarray shape (n_mels, n_fft//2+1)  — filterbank matrix

    Returns:
        np.ndarray shape (n_mels, n_frames) — mel-scaled spectrogram
    """
    return np.dot(mel_basis, S)


##    <(''<)  <( ' ' )>  (>'')>
# UNIFIED SPECTROGRAM COMPUTATION
##    <(''<)  <( ' ' )>  (>'')>

def compute_spectrogram_unified(y, sr, nfft, hop, fmin=0, fmax=None,
                                scale='linear', n_mels=256,
                                orientation='horizontal'):
    """Compute a spectrogram with optional mel scaling and orientation.

    Adapts nperseg and noverlap to signal length to handle short clips
    without raising scipy errors.

    Args:
        y:           np.ndarray — mono audio signal (float32)
        sr:          int        — sample rate in Hz
        nfft:        int        — FFT size
        hop:         int        — hop length in samples
        fmin:        float      — minimum frequency for output in Hz
        fmax:        float/None — maximum frequency for output in Hz; defaults to sr/2
        scale:       str        — 'linear' or 'mel'
        n_mels:      int        — number of mel bands (used only when scale='mel')
        orientation: str        — 'horizontal' (time on x) or 'vertical' (freq on x)

    Returns:
        S_db:   np.ndarray — spectrogram in dB
        freqs:  np.ndarray — frequency axis array in Hz (or mel if scale='mel')
        times:  np.ndarray — time axis array in seconds
    """
    if fmax is None:
        fmax = sr / 2.0

    # (つ -' _ '- )つ    (つ -' _ '- )つ
    # Adapt window and overlap to signal length to handle short clips
    # (つ -' _ '- )つ    (つ -' _ '- )つ
    L        = len(y)
    nperseg  = min(nfft, max(16, L))
    noverlap = max(0, min(nperseg - hop, nperseg - 1))

    freqs, times, S = spectrogram(
        y, fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        mode='magnitude'
    )

    if scale == 'mel':
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # Mel path: apply filterbank over full frequency range
        # then return mel-scaled output without masking
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        mel_basis, mel_freqs = create_mel_filterbank(sr, nfft, n_mels, fmin, fmax)

        # Expand mel_basis to match actual freq bins from scipy (may differ from nfft//2+1)
        n_fft_bins = len(freqs)
        mel_basis_adapted = np.zeros((n_mels, n_fft_bins))

        # Map filterbank columns onto actual scipy frequency bins
        for i in range(n_mels):
            # Recompute filter over actual freqs array
            left   = mel_to_hz(hz_to_mel(fmin) + i       * (hz_to_mel(fmax) - hz_to_mel(fmin)) / (n_mels + 1))
            center = mel_to_hz(hz_to_mel(fmin) + (i + 1) * (hz_to_mel(fmax) - hz_to_mel(fmin)) / (n_mels + 1))
            right  = mel_to_hz(hz_to_mel(fmin) + (i + 2) * (hz_to_mel(fmax) - hz_to_mel(fmin)) / (n_mels + 1))

            rising  = np.clip((freqs - left)   / (center - left  + 1e-10), 0.0, 1.0)
            falling = np.clip((right - freqs)  / (right  - center + 1e-10), 0.0, 1.0)
            mel_basis_adapted[i] = rising * falling

        S_final      = np.dot(mel_basis_adapted, S)
        freqs_final  = mel_freqs

    else:
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # Linear path: mask to fmin/fmax range
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        freq_mask   = (freqs >= fmin) & (freqs <= fmax)
        S_final     = S[freq_mask, :]
        freqs_final = freqs[freq_mask]

    # Convert magnitude to dB with floor to suppress -inf
    S_db = 20 * np.log10(S_final + 1e-12)

    # Rotate for vertical orientation (freq on x-axis, time on y-axis)
    if orientation == 'vertical':
        S_db = np.fliplr(np.rot90(S_db, k=-1))

    return S_db, freqs_final, times


##    <(''<)  <( ' ' )>  (>'')>
# WELCH PSD
##    <(''<)  <( ' ' )>  (>'')>

def compute_psd(y, sr, nfft_psd=None, noverlap_psd=None, hop_psd=None):
    """Compute a normalized Welch PSD with auto-adapted window length.

    Adapts nperseg to signal length to handle short clips safely.
    hop_psd takes precedence over noverlap_psd if both are provided.

    Args:
        y:            np.ndarray — mono audio signal
        sr:           int        — sample rate in Hz
        nfft_psd:     int/None   — FFT size; defaults to 1024
        noverlap_psd: int/None   — overlap in samples; defaults to 512
        hop_psd:      int/None   — hop length in samples (overrides noverlap_psd)

    Returns:
        freqs:    np.ndarray — frequency array in Hz
        psd_norm: np.ndarray — PSD normalized to [0, 1]
    """
    L        = len(y)
    nfft_psd = nfft_psd     or 1024
    noverlap_psd = noverlap_psd or 512

    # Adapt to signal length
    nperseg = min(nfft_psd, max(16, L // 2))

    if hop_psd is not None:
        noverlap = nperseg - hop_psd
    else:
        noverlap = noverlap_psd

    noverlap = max(0, min(noverlap, nperseg - 1))

    freqs, psd = welch(
        y, fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    # Normalize to [0, 1]
    psd_norm = psd / (psd.max() + 1e-12)

    return freqs, psd_norm


##    <(''<)  <( ' ' )>  (>'')>
# FRAME UTILITIES
##    <(''<)  <( ' ' )>  (>'')>

def frames_to_time(frames, sr, hop_length):
    """Convert frame indices to time in seconds.

    Args:
        frames:     int or np.ndarray — frame index or array of indices
        sr:         int               — sample rate in Hz
        hop_length: int               — hop length in samples

    Returns:
        float or np.ndarray — time in seconds
    """
    return frames * hop_length / sr


# (つ -' _ '- )つ    (つ -' _ '- )つ
# TODO: Refactor mel filterbank to use a single consistent implementation
# across create_mel_filterbank() and the inline adaptation in
# compute_spectrogram_unified() mel path. Currently duplicated.
# (つ -' _ '- )つ    (つ -' _ '- )つ


# U S A G I
# from yaaat.core.audio_utils import compute_spectrogram_unified, compute_psd, hz_to_mel
# S_db, freqs, times = compute_spectrogram_unified(y, sr, nfft=256, hop=64)
# freqs, psd = compute_psd(y, sr)
# mel = hz_to_mel(1000.0)