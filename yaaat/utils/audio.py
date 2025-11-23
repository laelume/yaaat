"""Shared utility functions for YAAAT annotation tools"""

import numpy as np
from scipy.signal import spectrogram, welch

# ===== Mel Scale Utilities =====

def hz_to_mel(hz):
    """Convert Hz to mel scale"""
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    """Convert mel scale to Hz"""
    return 700 * (10**(mel / 2595) - 1)

def create_mel_filterbank(sr, n_fft, n_mels=128, fmin=0, fmax=None):
    """
    Create mel filterbank matrix
    
    Args:
        sr: Sample rate
        n_fft: FFT size
        n_mels: Number of mel bands
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz), defaults to sr/2
    
    Returns:
        mel_basis: (n_mels, n_fft//2 + 1) filterbank matrix
        mel_freqs: Center frequencies of mel bands
    """
    if fmax is None:
        fmax = sr / 2
    
    # Create mel-spaced frequencies
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Create FFT bin frequencies
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    
    # Create filterbank
    mel_basis = np.zeros((n_mels, n_fft // 2 + 1))
    
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]
        
        # Rising slope
        rising = (fft_freqs - left) / (center - left)
        rising = np.maximum(0, rising)
        rising = np.minimum(1, rising)
        
        # Falling slope
        falling = (right - fft_freqs) / (right - center)
        falling = np.maximum(0, falling)
        falling = np.minimum(1, falling)
        
        # Combine
        mel_basis[i] = rising * falling
    
    mel_freqs = hz_points[1:-1]  # Center frequencies
    
    return mel_basis, mel_freqs

def apply_mel_scale(S, mel_basis):
    """
    Apply mel filterbank to linear spectrogram
    
    Args:
        S: Linear magnitude spectrogram (freq_bins, time_frames)
        mel_basis: Mel filterbank matrix (n_mels, freq_bins)
    
    Returns:
        S_mel: Mel-scaled spectrogram (n_mels, time_frames)
    """
    return np.dot(mel_basis, S)

# ===== Unified Spectrogram Computation =====

def compute_spectrogram_unified(y, sr, nfft, hop, fmin=0, fmax=None, 
                                scale='linear', n_mels=256, orientation='horizontal'):
    """
    Unified spectrogram computation with mel scale support
    
    Args:
        y: Audio signal
        sr: Sample rate
        nfft: FFT size
        hop: Hop length
        fmin: Min frequency
        fmax: Max frequency
        scale: 'linear' or 'mel'
        n_mels: Number of mel bands (only used if scale='mel')
        orientation: 'horizontal' (time=x) or 'vertical' (freq=x)
    
    Returns:
        S_db: Spectrogram in dB
        freqs: Frequency array
        times: Time array
    """
    if fmax is None:
        fmax = sr / 2
    
    # Adapt for short signals
    L = len(y)
    nperseg = min(nfft, max(16, L))
    noverlap = nperseg - hop
    noverlap = max(0, min(noverlap, nperseg - 1))
    
    # Compute linear spectrogram
    freqs, times, S = spectrogram(
        y, fs=sr, nperseg=nperseg, noverlap=noverlap,
        scaling='density', mode='magnitude'
    )
    
    # Apply frequency mask
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    S_masked = S[freq_mask, :]
    freqs_masked = freqs[freq_mask]
    
    # Apply mel scale if requested
    if scale == 'mel':
        mel_basis, mel_freqs = create_mel_filterbank(sr, nfft, n_mels, fmin, fmax)
        # Apply mel basis only to masked frequencies
        mel_basis_full = np.zeros((n_mels, len(freqs)))
        mel_basis_full[:, freq_mask] = mel_basis[:, freq_mask]
        S_final = apply_mel_scale(S, mel_basis_full)
        freqs_final = mel_freqs
    else:
        S_final = S_masked
        freqs_final = freqs_masked
    
    # Convert to dB
    S_db = 20 * np.log10(S_final + 1e-12)
    
    # Rotate if vertical orientation requested
    if orientation == 'vertical':
        S_db = np.fliplr(np.rot90(S_db, k=-1))
    
    return S_db, freqs_final, times

def compute_psd(y, sr, nfft_psd=None, hop_psd=None):
    """
    Welch PSD with auto-adjusted nperseg/noverlap
    
    Args:
        y: Audio signal
        sr: Sample rate
        nfft_psd: FFT size for PSD
        hop_psd: Hop length in samples
    
    Returns:
        freqs: Frequency array
        psd_norm: Normalized PSD
    """
    L = len(y)
    nfft_psd = nfft_psd or 1024
    
    # Adapt nperseg for signal length
    nperseg = min(nfft_psd, max(16, L // 2))
    
    # Calculate noverlap from hop
    if hop_psd is not None:
        noverlap = nperseg - hop_psd
    else:
        noverlap = nperseg // 2
    
    # Ensure noverlap is valid
    noverlap = max(0, min(noverlap, nperseg - 1))
    
    # Compute Welch PSD
    freqs, psd = welch(
        y, fs=sr, nperseg=nperseg, noverlap=noverlap, scaling="density"
    )
    
    # Normalize PSD
    psd_norm = psd / (psd.max() + 1e-12)
    
    return freqs, psd_norm

def frames_to_time(frames, sr, hop_length):
    """Convert frame indices to time (replaces librosa.frames_to_time)"""
    return frames * hop_length / sr
