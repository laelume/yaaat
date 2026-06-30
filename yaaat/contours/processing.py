"""Audio file processing orchestration."""

import numpy as np

import pysoniq
from yaaat.audio_utils import compute_spectrogram_unified


from pathlib import Path
from typing import Dict
from natsort import natsorted
from scipy.ndimage import label
from skimage import measure, restoration
from skimage.filters import hessian, frangi, sobel, threshold_local
# import cv2

from .algorithms import deimpulse_spectrogram


def process_audio_file(
    audio_file, 
    y=None, 
    sr=None, 
    # Spectrogram parameters
    fmin=0, 
    fmax=16000, 
    n_fft=512, 
    hop_length=64,
    # Impulse denoising parameters
    denoise_impulses=True, 
    impulse_freq_threshold=0.4, 
    impulse_intensity_threshold_db=10,
    impulse_max_gap=3,
    impulse_expand=2,
    # Frequency band parameters (for rolling ball)
    low_freq_boundary_hz=1000,
    mid_freq_boundary_hz=3000,
    # Rolling ball kernel parameters
    kernel_low_freq=(7, 20),
    kernel_low_flatness=0.1,
    kernel_mid_freq=(5, 20),
    kernel_mid_flatness=0.1,
    kernel_high_freq=(3, 20),
    kernel_high_flatness=0.1,
    # CLAHE parameters
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    # Ridge detection parameters
    ridge_filter='hessian',
    hessian_sigmas=(1, 5),
    frangi_sigmas=(1, 5),
    frangi_scale_step=0.5,
    ridge_threshold=0.3,
    # Frangi fallback parameters
    use_frangi_fallback=False,
    fallback_edge_threshold=100,
    # Component selection parameters
    time_dist_divisor=50.0,
    freq_dist_divisor=100.0,
    size_score_divisor=5.0,
    energy_score_divisor=10.0,
    time_weight=5.0,
    freq_weight=1.0,
    size_weight=1.0,
    energy_weight=1.0
):
    """
    Process a single audio file with full parameter control.
    
    Returns dict with keys: 'clahe', 'component', 'max_energy_idx', 
    'num_components', 'n_pixels', 'used_fallback', 'ridge_filter'
    """
    
    # Load audio only if not provided
    if y is None:
        y, sr = pysoniq.load_audio(audio_file)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    
    # Generate mel spectrogram (already in dB)
    
    S_db_original, freqs, times = compute_spectrogram_unified(
        y=y,
        sr=sr,
        nfft=n_fft,
        hop=hop_length,
        fmin=fmin,
        fmax=fmax,
        scale='mel',
        n_mels=128
    )
    
    # Normalize to match ref=np.max behavior
    S_db_original = S_db_original - np.max(S_db_original)
      
    # Apply impulse denoising if enabled
    if denoise_impulses:
        S_db_denoised, impulse_info = deimpulse_spectrogram(
            S_db_original,
            freq_threshold=impulse_freq_threshold,
            intensity_threshold_db=impulse_intensity_threshold_db,
            max_gap=impulse_max_gap,
            expand=impulse_expand
        )
        S_db_to_use = S_db_denoised
    else:
        S_db_to_use = S_db_original
    
    # Normalize
    img_original = (S_db_to_use - S_db_to_use.min()) / (S_db_to_use.max() - S_db_to_use.min())
    
    # Find point with maximum energy
    max_energy_idx = np.unravel_index(np.argmax(S_db_to_use), S_db_to_use.shape)
    freq_bin = max_energy_idx[0]
    max_energy_frame = max_energy_idx[1]
    
    # Get mel frequencies for band boundaries
    mel_freqs = pysoniq.fourier.mel_frequencies(n_mels=S_db_original.shape[0], fmin=fmin, fmax=fmax)
    low_freq_end = np.searchsorted(mel_freqs, low_freq_boundary_hz)
    mid_freq_end = np.searchsorted(mel_freqs, mid_freq_boundary_hz)
    
    # Setup rolling ball kernels
    kernel_low = restoration.ellipsoid_kernel(kernel_low_freq, kernel_low_flatness)
    kernel_mid = restoration.ellipsoid_kernel(kernel_mid_freq, kernel_mid_flatness)
    kernel_high = restoration.ellipsoid_kernel(kernel_high_freq, kernel_high_flatness)
    
    def apply_freq_adaptive(img_input):
        """Apply frequency-adaptive rolling ball background subtraction."""
        background = np.zeros_like(img_input)
        background[:low_freq_end, :] = restoration.rolling_ball(img_input[:low_freq_end, :], kernel=kernel_low)
        background[low_freq_end:mid_freq_end, :] = restoration.rolling_ball(img_input[low_freq_end:mid_freq_end, :], kernel=kernel_mid)
        background[mid_freq_end:, :] = restoration.rolling_ball(img_input[mid_freq_end:, :], kernel=kernel_high)
        return img_input - background
    
    # Baseline: Original → Cubed → Freq-Adaptive
    img_cubed = img_original * img_original * img_original
    result_baseline = apply_freq_adaptive(img_cubed)

    # USING CV2    
    # # Convert to uint8 and apply CLAHE
    # result_uint8 = (result_baseline * 255).astype(np.uint8)
    # clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    # clahe_result = clahe.apply(result_uint8) / 255.0
    
    # USING SKIMAGE
    from skimage import exposure
    clahe_result = exposure.equalize_adapthist(
        result_baseline, 
        kernel_size=clahe_tile_grid_size,
        clip_limit=clahe_clip_limit / 100.0,  # Convert from OpenCV scale
        nbins=256
    )

    # Apply detection method based on ridge_filter choice
    if ridge_filter == 'hessian':
        ridge_result = hessian(clahe_result, sigmas=range(hessian_sigmas[0], hessian_sigmas[1]))
        binary = ridge_result > ridge_threshold
        
    elif ridge_filter == 'frangi':
        ridge_result = frangi(clahe_result, sigmas=frangi_sigmas, scale_step=frangi_scale_step)
        binary = ridge_result > ridge_threshold
        
    elif ridge_filter == 'threshold':
        # Simple intensity threshold on CLAHE
        binary = clahe_result > ridge_threshold
        
    elif ridge_filter == 'edge':
        # Edge detection with Sobel
        edges = sobel(clahe_result)
        binary = edges > ridge_threshold
        
    elif ridge_filter == 'adaptive':
        # Adaptive local thresholding
        block_size = int(ridge_threshold * 100) if ridge_threshold < 1 else 35
        block_size = block_size if block_size % 2 == 1 else block_size + 1  # Must be odd
        adaptive_thresh = threshold_local(clahe_result, block_size=block_size, method='gaussian')
        binary = clahe_result > adaptive_thresh

    else:
        raise ValueError(f"Unknown ridge_filter: {ridge_filter}")
    
    # Label connected components
    labeled, num_features = label(binary)
    
    # Check for Frangi fallback if using Hessian
    used_fallback = False
    if use_frangi_fallback and ridge_filter == 'hessian':
        hessian_target_label = labeled[max_energy_idx[0], max_energy_idx[1]]
        
        if hessian_target_label != 0:
            component_mask_temp = (labeled == hessian_target_label)
            time_coords = np.where(component_mask_temp.any(axis=0))[0]
            
            if len(time_coords) > 0:
                component_start = time_coords[0]
                component_end = time_coords[-1]
                total_frames = binary.shape[1]
                
                starts_near_edge = component_start < fallback_edge_threshold
                ends_near_edge = component_end > (total_frames - fallback_edge_threshold)
                
                if not starts_near_edge or not ends_near_edge:
                    # Switch to Frangi
                    frangi_result = frangi(clahe_result, sigmas=frangi_sigmas, scale_step=frangi_scale_step)
                    binary = frangi_result > ridge_threshold
                    labeled, num_features = label(binary)
                    used_fallback = True
    
    # Component selection
    target_label = labeled[max_energy_idx[0], max_energy_idx[1]]
    
    if target_label == 0:
        best_score = -float('inf')
        nearest_label = 0
        
        for region in measure.regionprops(labeled):
            time_dist = abs(region.centroid[1] - max_energy_idx[1])
            freq_dist = abs(region.centroid[0] - max_energy_idx[0])
            
            component_mask = (labeled == region.label)
            mean_energy = S_db_original[component_mask].mean()
            size = region.area
            
            # Scoring
            time_score = 1.0 / (1.0 + time_dist / time_dist_divisor)
            freq_score = 1.0 / (1.0 + freq_dist / freq_dist_divisor)
            size_score = np.log1p(size) / size_score_divisor
            energy_score = mean_energy / energy_score_divisor
            
            # Weighted combination
            score = (time_weight * time_score + 
                    freq_weight * freq_score + 
                    size_weight * size_score + 
                    energy_weight * energy_score)
            
            if score > best_score:
                best_score = score
                nearest_label = region.label
        
        target_label = nearest_label
    
    component_mask = (labeled == target_label).astype(float)
    
    return {
        'clahe': clahe_result,
        'component': component_mask,
        'max_energy_idx': max_energy_idx,
        'num_components': num_features,
        'n_pixels': int(component_mask.sum()),
        'used_fallback': used_fallback,
        'ridge_filter': 'frangi' if used_fallback else ridge_filter
    }


def process_directory(audio_dir, file_extension='.wav', **kwargs):
    """Process all audio files in a directory with customizable parameters."""
    
    audio_dir = Path(audio_dir)
    audio_files = natsorted(list(audio_dir.glob(f'*{file_extension}')))
    
    if not audio_files:
        print(f"No {file_extension} files found in {audio_dir}")
        return {}
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Processing with parameters: {kwargs.get('ridge_filter', 'hessian')} filter")
    
    results = {}
    for i, audio_file in enumerate(audio_files):
        print(f"Processing {i+1}/{len(audio_files)}: {audio_file.name}")
        try:
            results[audio_file.name] = process_audio_file(str(audio_file), **kwargs)
            if results[audio_file.name].get('used_fallback'):
                print(f"  → Used Frangi fallback")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results