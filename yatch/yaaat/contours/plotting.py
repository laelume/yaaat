"""Visualization functions for contour extraction results."""

import numpy as np
import matplotlib.pyplot as plt
import pysoniq
from yaaat.audio_utils import compute_spectrogram_unified

from pathlib import Path
from skimage import restoration
# import cv2

from .algorithms import deimpulse_spectrogram


def plot_results(results, output_path=None, dir_name=None):
    """Plot results for all files."""
    
    n_files = len(results)
    if n_files == 0:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(n_files, 2, figsize=(12, 3*n_files))
    
    if n_files == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (filename, data) in enumerate(results.items()):
        max_energy_idx = data['max_energy_idx']
        filter_used = data.get('ridge_filter', 'hessian')
        
        # Plot CLAHE
        axes[idx, 0].imshow(data['clahe'], cmap='gray', aspect='auto', origin='lower')
        axes[idx, 0].plot([max_energy_idx[1] - 5, max_energy_idx[1] + 5], 
                          [max_energy_idx[0], max_energy_idx[0]], color='lime', lw=2)
        axes[idx, 0].plot([max_energy_idx[1], max_energy_idx[1]], 
                          [max_energy_idx[0] - 5, max_energy_idx[0] + 5], color='lime', lw=2)
        axes[idx, 0].set_title(f'{filename}\n{filter_used}', fontsize=8)
        axes[idx, 0].axis('off')
        
        # Plot Component
        axes[idx, 1].imshow(data['component'], cmap='hot', aspect='auto', origin='lower')
        axes[idx, 1].plot([max_energy_idx[1] - 5, max_energy_idx[1] + 5], 
                          [max_energy_idx[0], max_energy_idx[0]], color='lime', lw=2)
        axes[idx, 1].plot([max_energy_idx[1], max_energy_idx[1]], 
                          [max_energy_idx[0] - 5, max_energy_idx[0] + 5], color='lime', lw=2)
        axes[idx, 1].set_title(f'{data["n_pixels"]} px', fontsize=8)
        axes[idx, 1].axis('off')
    
    fig.tight_layout()
    
    if output_path:
        if dir_name:
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_{dir_name}{ext}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


def print_summary(results):
    """Print summary statistics."""
    print("\n=== Summary ===")
    print(f"Total files processed: {len(results)}")
    fallback_count = sum(1 for r in results.values() if r.get('used_fallback', False))
    if fallback_count > 0:
        print(f"Frangi fallbacks used: {fallback_count}")
    print("\nComponent sizes:")
    for filename, data in results.items():
        fallback_str = " (fallback)" if data.get('used_fallback') else ""
        print(f"  {filename}: {data['n_pixels']} px{fallback_str}")


def plot_multi_comparison(
    results_dict, 
    audio_dir, 
    denoise_params=None, 
    output_path=None, 
    n_energy_peaks=5, 
    show_titles=False, 
    show_plots=True, 
    verbose=False
):
    """
    Plot with visualization of preprocessing steps including impulse denoising.
    Uses frequency-adaptive rolling ball kernels from the real pipeline.
    
    Parameters
    ----------
    results_dict : OrderedDict
        Structure: {config_name: {filename: process_audio_file_result}}
    audio_dir : str or Path
        Directory containing audio files for reloading
    denoise_params : dict, optional
        Impulse denoising params: 'freq_threshold', 'intensity_threshold_db', 
        'max_gap', 'expand'
    output_path : Path, optional
        Save path for plot
    n_energy_peaks : int, default=5
        Number of energy peaks to mark
    show_titles : bool, default=False
        Show column titles
    show_plots : bool, default=True
        Display plots
    verbose : bool, default=False
        Print summary
    """
    
    # Default denoise params if not provided
    if denoise_params is None:
        denoise_params = {
            'freq_threshold': 0.7,
            'intensity_threshold_db': 10,
            'max_gap': 3,
            'expand': 2
        }

    first_results = list(results_dict.values())[0]
    n_files = len(first_results)
    n_configs = len(results_dict)
    
    # FULL PIPELINE: Original → Denoised → Cubed → RollingBall → CLAHE → Results
    fig, axes = plt.subplots(n_files, n_configs + 5, 
                            figsize=((n_configs + 5) * 3, n_files * 2.5))
    
    if n_files == 1:
        axes = axes.reshape(1, -1)
    
    filenames = list(first_results.keys())
    audio_dir_path = Path(audio_dir)
    
    for row_idx, filename in enumerate(filenames):
        
        # Reload and reprocess to show intermediate steps
        audio_file = audio_dir_path / filename
        y, sr = pysoniq.load_audio(str(audio_file))
        
        # Convert to mono if needed
        if y.ndim > 1:
            y = y.mean(axis=1)
        
        # Step 1: Original
        S_db_original, freqs, times = compute_spectrogram_unified(
            y=y, sr=sr, nfft=512, hop=128, fmin=1000, fmax=8000,
            scale='mel', n_mels=128
        )
        # Normalize to match ref=np.max behavior
        S_db_original = S_db_original - np.max(S_db_original)

        # Step 2: Impulse Denoising 
        S_db_denoised, impulse_info = deimpulse_spectrogram(
            S_db_original,
            freq_threshold=denoise_params['freq_threshold'],
            intensity_threshold_db=denoise_params['intensity_threshold_db'],
            max_gap=denoise_params['max_gap'],
            expand=denoise_params['expand']
        )
        
        # Step 3: Normalize again
        img_normalized = (S_db_denoised - S_db_denoised.min()) / (S_db_denoised.max() - S_db_denoised.min())
        
        # Step 4: Cubed
        img_cubed = img_normalized * img_normalized * img_normalized
        
        # Step 5: REAL frequency-adaptive rolling ball
        mel_freqs = pysoniq.fourier.mel_frequencies(n_mels=S_db_original.shape[0], fmin=1000, fmax=8000)
        low_freq_end = np.searchsorted(mel_freqs, 1000)
        mid_freq_end = np.searchsorted(mel_freqs, 3000)
        
        # REAL kernels from actual pipeline
        kernel_low = restoration.ellipsoid_kernel((7, 20), 0.1)
        kernel_mid = restoration.ellipsoid_kernel((5, 20), 0.1)
        kernel_high = restoration.ellipsoid_kernel((3, 20), 0.1)
        
        # REAL frequency-adaptive rolling ball
        background = np.zeros_like(img_cubed)
        background[:low_freq_end, :] = restoration.rolling_ball(img_cubed[:low_freq_end, :], kernel=kernel_low)
        background[low_freq_end:mid_freq_end, :] = restoration.rolling_ball(img_cubed[low_freq_end:mid_freq_end, :], kernel=kernel_mid)
        background[mid_freq_end:, :] = restoration.rolling_ball(img_cubed[mid_freq_end:, :], kernel=kernel_high)
        img_rollingball = img_cubed - background
        
        # # Step 6: CLAHE
        # result_uint8 = (img_rollingball * 255).astype(np.uint8)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # img_clahe = clahe.apply(result_uint8) / 255.0

        # Apply CLAHE using scikit-image (no cv2 needed)
        from skimage import exposure
        clahe_result = exposure.equalize_adapthist(
            result_baseline, 
            kernel_size=clahe_tile_grid_size,
            clip_limit=clahe_clip_limit / 100.0,  # Convert from OpenCV scale
            nbins=256
        )

        # Get max energy point from results
        first_config = list(results_dict.values())[0]
        data = first_config[filename]
        max_energy_idx = data['max_energy_idx']

        # Find top N energy peaks (spatially distinct)
        mask_radius = 15
        peak_indices = [max_energy_idx]
        peak_colors = ['magenta', 'darkorange', 'yellow', 'lime', 'cyan', 'blue', 'violet']
        img_clahe_copy = img_clahe.copy()

        for i in range(1, n_energy_peaks):
            # Mask out region around previous peak
            prev_peak = peak_indices[-1]
            y_min = max(0, prev_peak[0] - mask_radius)
            y_max = min(img_clahe.shape[0], prev_peak[0] + mask_radius + 1)
            x_min = max(0, prev_peak[1] - mask_radius)
            x_max = min(img_clahe.shape[1], prev_peak[1] + mask_radius + 1)
            img_clahe_copy[y_min:y_max, x_min:x_max] = 0
            peak_indices.append(np.unravel_index(np.argmax(img_clahe_copy), img_clahe_copy.shape))

        # Column 0: Original (raw dB)
        axes[row_idx, 0].imshow(S_db_original, cmap='gray', aspect='auto', origin='lower')
        if show_titles and row_idx == 0:
            axes[row_idx, 0].set_title('1. Original Mel dB', fontsize=8, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Column 1: Denoised (show what was removed in red overlay)
        axes[row_idx, 1].imshow(S_db_denoised, cmap='gray', aspect='auto', origin='lower')
        n_impulse_frames = len(impulse_info['impulse_frames_to_remove'])
        if n_impulse_frames > 0:
            impulse_mask = np.zeros_like(S_db_original)
            for frame in impulse_info['impulse_frames_to_remove']:
                if frame < impulse_mask.shape[1]:
                    impulse_mask[:, frame] = 1
            axes[row_idx, 1].contour(impulse_mask, levels=[0.5], colors='red', linewidths=1, alpha=0.7)
        
        if show_titles: 
            if row_idx == 0:
                axes[row_idx, 1].set_title(f'2. Deimpulsed ({n_impulse_frames} frames)', fontsize=8, fontweight='bold')
            else:
                axes[row_idx, 1].set_title(f'{n_impulse_frames} frames', fontsize=7)
        axes[row_idx, 1].axis('off')
        
        # Column 2: Cubed
        axes[row_idx, 2].imshow(img_cubed, cmap='gray', aspect='auto', origin='lower')
        if show_titles and row_idx == 0:
            axes[row_idx, 2].set_title('3. Normalized & Cubed', fontsize=8, fontweight='bold')
        axes[row_idx, 2].axis('off')
        
        # Column 3: After Rolling Ball
        axes[row_idx, 3].imshow(img_rollingball, cmap='gray', aspect='auto', origin='lower')
        if show_titles and row_idx == 0:
            axes[row_idx, 3].set_title('4. Rolling Ball (Freq-Adaptive)', fontsize=8, fontweight='bold')
        axes[row_idx, 3].axis('off')
        
        # Column 4: After CLAHE
        axes[row_idx, 4].imshow(img_clahe, cmap='gray', aspect='auto', origin='lower')
        
        # Plot all peak crosshairs for N peaks
        for peak_idx, color in zip(peak_indices, peak_colors[:n_energy_peaks]):
            axes[row_idx, 4].plot([peak_idx[1] - 5, peak_idx[1] + 5], 
                                [peak_idx[0], peak_idx[0]], color=color, lw=2)
            axes[row_idx, 4].plot([peak_idx[1], peak_idx[1]], 
                                [peak_idx[0] - 5, peak_idx[0] + 5], color=color, lw=2)

        if show_titles and row_idx == 0:
            axes[row_idx, 4].set_title('5. CLAHE Enhanced', fontsize=8, fontweight='bold')
        axes[row_idx, 4].text(-0.15, 0.5, filename, transform=axes[row_idx, 4].transAxes,
                             fontsize=7, rotation=90, va='center', ha='right')
        axes[row_idx, 4].axis('off')

        # Columns 5+: Results from each config
        for col_idx, (label, results) in enumerate(results_dict.items(), start=5):
            data = results[filename]
            max_energy_idx = data['max_energy_idx']
            
            axes[row_idx, col_idx].imshow(data['component'], cmap='gray', aspect='auto', origin='lower')
            
            # Plot all peak crosshairs
            for peak_idx, color in zip(peak_indices, peak_colors[:n_energy_peaks]):
                axes[row_idx, col_idx].plot([peak_idx[1] - 5, peak_idx[1] + 5], 
                                            [peak_idx[0], peak_idx[0]], color=color, lw=2)
                axes[row_idx, col_idx].plot([peak_idx[1], peak_idx[1]], 
                                            [peak_idx[0] - 5, peak_idx[0] + 5], color=color, lw=2)            

            px_count = data["n_pixels"]
            fallback_marker = "*" if data.get('used_fallback') else ""
            
            if show_titles: 
                if row_idx == 0:
                    axes[row_idx, col_idx].set_title(f'{label}\n{px_count}px{fallback_marker}', fontsize=8, fontweight='bold')
                else:
                    axes[row_idx, col_idx].set_title(f'{px_count}px{fallback_marker}', fontsize=7)
            axes[row_idx, col_idx].axis('off')
    
    if show_titles: 
        plt.suptitle('Full Pipeline: Original → Impulse Denoise → Cube → Rolling Ball (Freq-Adaptive) → CLAHE → Ridge Detection', 
                     fontsize=10, fontweight='bold', y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved multiplot to {Path(output_path)}")
    
    if show_plots:
        plt.show()
    
    if verbose: 
        print_summary(list(results_dict.values())[0])