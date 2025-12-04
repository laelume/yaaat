# yaaat/utils.py

"""Shared utility functions for YAAAT annotation tools"""

import numpy as np
from scipy.signal import spectrogram, welch
import json
from pathlib import Path


# ----------------------------------- #
# ******** SIGNAL PROCESSING ******** #
# ----------------------------------- #

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


# ===== Unified Spectrogram Stuff =====

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
        # Create mel basis for full frequency range
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


# **************** Dual-resolution??? PSD **************** #


def compute_psd(y, sr, nfft_psd=None, noverlap_psd=None, hop_psd=None):
    """
    Welch PSD with auto-adjusted nperseg/noverlap to fit signal length
    
    Args:
        y: Audio signal
        sr: Sample rate
        nfft_psd: FFT size for PSD
        noverlap_psd: Overlap samples (alternative to hop_psd)
        hop_psd: Hop length in samples (takes precedence over noverlap)
    
    Returns:
        freqs: Frequency array
        psd_norm: Normalized PSD
    """
    L = len(y)
    
    nfft_psd = nfft_psd or 1024
    noverlap_psd = noverlap_psd or 512
    
    # Adapt nperseg for signal length
    nperseg = min(nfft_psd, max(16, L // 2))
    
    # Calculate noverlap from hop if provided
    if hop_psd is not None:
        noverlap = nperseg - hop_psd
    else:
        noverlap = min(noverlap_psd, nperseg - 1)
    
    # Ensure noverlap is valid
    noverlap = max(0, min(noverlap, nperseg - 1))
    
    # Compute Welch PSD
    freqs, psd = welch(
        y,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density"
    )
    
    # Normalize PSD
    psd_norm = psd / (psd.max() + 1e-12)
    
    return freqs, psd_norm


def frames_to_time(frames, sr, hop_length):
    """Convert frame indices to time (replaces librosa.frames_to_time)"""
    return frames * hop_length / sr


# ===== Configuration Management =====

def save_last_directory(directory):
    """Save last opened directory to config file
    
    Args:
        directory: Path object or string
    """    
    config_file = Path.home() / '.yaaat_config.json'
    try:
        # Load existing config if it exists
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Update last directory
        config['last_directory'] = str(directory)
        
        # Save config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save config: {e}")

def load_last_directory():
    """Load last opened directory from config file
    
    Returns:
        Path object if valid directory exists, None otherwise
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
        print(f"Warning: Could not load config: {e}")
    return None





# ******** LEGACY ******** #


# # ----------------------------------------------------------------------
# # Dual-resolution spectrogram and PSD computation functions
# # ----------------------------------------------------------------------

# def compute_vertical_spectrogram(y, sr, nfft_spect=None, noverlap_spect=None, hop_spect=None):
#     """
#     Spectrogram rotated vertically (freq->x, time->y), safe for short signals
#     and flipped horizontally (high freq -> right).
    
#     Args:
#         y: Audio signal
#         sr: Sample rate
#         nfft_spect: FFT size for spectrogram
#         noverlap_spect: Overlap samples (alternative to hop_spect)
#         hop_spect: Hop length in samples (takes precedence over noverlap)
    
#     Returns:
#         S_db_rot: Rotated spectrogram in dB
#         freqs: Frequency array
#         times: Time array
#     """
#     nfft_spect = nfft_spect or 512
#     noverlap_spect = noverlap_spect or 256

#     L = len(y)

#     # Adapt nperseg for short signals
#     nperseg = min(nfft_spect, max(16, L))
    
#     # Calculate noverlap from hop if provided
#     if hop_spect is not None:
#         noverlap = nperseg - hop_spect
#     else:
#         noverlap = min(noverlap_spect, nperseg - 1)
    
#     # Ensure noverlap is valid
#     noverlap = max(0, min(noverlap, nperseg - 1))

#     # Compute spectrogram
#     freqs, times, S = spectrogram(
#         y,
#         fs=sr,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         scaling="density",
#         mode="magnitude"
#     )

#     # Convert to dB
#     S_db = 20 * np.log10(S + 1e-12)
    
#     # Rotate so freq->x, time->y, then flip horizontally
#     S_db_rot = np.fliplr(np.rot90(S_db, k=-1))

#     return S_db_rot, freqs, times


# def compute_psd(y, sr, nfft_psd=None, noverlap_psd=None, hop_psd=None):
#     """
#     Welch PSD with auto-adjusted nperseg/noverlap to fit signal length
    
#     Args:
#         y: Audio signal
#         sr: Sample rate
#         nfft_psd: FFT size for PSD
#         noverlap_psd: Overlap samples (alternative to hop_psd)
#         hop_psd: Hop length in samples (takes precedence over noverlap)
    
#     Returns:
#         freqs: Frequency array
#         psd_norm: Normalized PSD
#     """
#     L = len(y)
    
#     nfft_psd = nfft_psd or 1024
#     noverlap_psd = noverlap_psd or 512
    
#     # Adapt nperseg for signal length
#     nperseg = min(nfft_psd, max(16, L // 2))
    
#     # Calculate noverlap from hop if provided
#     if hop_psd is not None:
#         noverlap = nperseg - hop_psd
#     else:
#         noverlap = min(noverlap_psd, nperseg - 1)
    
#     # Ensure noverlap is valid
#     noverlap = max(0, min(noverlap, nperseg - 1))
    
#     # Compute Welch PSD
#     freqs, psd = welch(
#         y,
#         fs=sr,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         scaling="density"
#     )
    
#     # Normalize PSD
#     psd_norm = psd / (psd.max() + 1e-12)
    
#     return freqs, psd_norm
