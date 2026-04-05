"""
Parameter optimization for contour extraction.
Finds optimal parameters using scipy.optimize with quality metrics.
"""

import numpy as np
import pysoniq
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import label as scipy_label
import json
from pathlib import Path
import sys
from typing import Dict, Optional

from yaaat.audio_utils import compute_spectrogram_unified

# Add yaaat to path
sys.path.insert(0, str(Path.cwd()))

# Import from contours
try:
    from .processing import process_audio_file
except ImportError:
    from processing import process_audio_file


def compute_spectrogram_for_eval(y, sr, fmin=1000, fmax=8000, n_fft=512, hop_length=128):
    """Compute spectrogram for evaluation"""   
    S_db, freqs, times = compute_spectrogram_unified( # returns db directly
        y=y, 
        sr=sr, 
        nfft=n_fft, 
        hop=hop_length, 
        fmin=fmin, 
        fmax=fmax,
        scale='mel',
        n_mels=128  # Default, adjust if needed
    )
    
    # Normalize 
    S_db = S_db - np.max(S_db)
    
    return S_db


def evaluate_contour_quality(result: Dict, S_db: np.ndarray, weights: Optional[Dict] = None) -> Dict:
    """
    Evaluate contour quality using multiple heuristic metrics.
    
    Parameters
    ----------
    result : dict
        Output from process_audio_file()
    S_db : ndarray
        Spectrogram in dB for energy calculations
    weights : dict, optional
        Custom weights for combining metrics
    
    Returns
    -------
    metrics : dict
        Dictionary with individual metrics and overall_score
    """
    if weights is None:
        weights = {
            'time_coverage': 0.30,
            'compactness': 0.25,
            'energy_ratio': 0.25,
            'snr': 0.20
        }
    
    component = result['component']
    
    if component.sum() == 0:
        return {
            'overall_score': 0.0,
            'time_coverage': 0.0,
            'compactness': 0.0,
            'energy_ratio': 0.0,
            'thickness_consistency': 0.0,
            'snr': 0.0,
            'n_components': 0,
            'n_pixels': 0
        }
    
    # 1. Time coverage
    time_coverage = np.any(component, axis=0).sum() / component.shape[1]
    
    # 2. Compactness
    labeled, n_components = scipy_label(component)
    compactness = 1.0 / max(1, n_components)
    
    # 3. Energy ratio
    total_energy = np.abs(S_db).sum()
    contour_energy = np.abs(S_db[component > 0]).sum()
    energy_ratio = contour_energy / total_energy if total_energy > 0 else 0
    
    # 4. Thickness consistency
    freq_spans = []
    for t_idx in range(component.shape[1]):
        active_freqs = np.where(component[:, t_idx])[0]
        if len(active_freqs) > 0:
            freq_spans.append(active_freqs.max() - active_freqs.min())
    
    if freq_spans:
        thickness_mean = np.mean(freq_spans)
        thickness_std = np.std(freq_spans)
        thickness_consistency = 1.0 / (1.0 + thickness_std / max(1, thickness_mean))
    else:
        thickness_consistency = 0.0
    
    # 5. SNR
    contour_mean = S_db[component > 0].mean()
    background_mean = S_db[component == 0].mean() if (component == 0).any() else S_db.min()
    snr = contour_mean - background_mean
    snr_normalized = min(1.0, max(0.0, snr / 20.0))
    
    # Overall score
    overall_score = (
        weights['time_coverage'] * time_coverage +
        weights['compactness'] * compactness +
        weights['energy_ratio'] * energy_ratio +
        weights['snr'] * snr_normalized
    )
    
    return {
        'overall_score': overall_score,
        'time_coverage': time_coverage,
        'compactness': compactness,
        'energy_ratio': energy_ratio,
        'thickness_consistency': thickness_consistency,
        'snr': snr,
        'snr_normalized': snr_normalized,
        'n_components': n_components,
        'n_pixels': int(component.sum())
    }


def optimize_parameters(
    audio_path: str,
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    method: str = 'differential_evolution',
    target_metric: str = 'overall_score',
    param_bounds: Optional[Dict] = None,
    fixed_params: Optional[Dict] = None,
    metric_weights: Optional[Dict] = None,
    verbose: bool = True
) -> Dict:
    """Find optimal contour extraction parameters using scipy optimization."""
    
    # Load audio if not provided
    if y is None:
        y, sr = pysoniq.load_audio(audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    
    # Default parameter bounds
    if param_bounds is None:
        param_bounds = {
            'impulse_freq_threshold': (0.1, 0.9),
            'ridge_threshold': (0.1, 0.5),
            'hessian_sigma_min': (1, 5),
            'hessian_sigma_max': (5, 15),
        }
    
    # Default fixed parameters
    if fixed_params is None:
        fixed_params = {
            'fmin': 1000, 'fmax': 8000, 'n_fft': 512, 'hop_length': 128,
            'denoise_impulses': True,
            'impulse_intensity_threshold_db': 10.0,
            'impulse_max_gap': 3, 'impulse_expand': 2,
            'low_freq_boundary_hz': 1000, 'mid_freq_boundary_hz': 3000,
            'kernel_low_freq': (7, 20), 'kernel_low_flatness': 0.1,
            'kernel_mid_freq': (5, 20), 'kernel_mid_flatness': 0.1,
            'kernel_high_freq': (3, 20), 'kernel_high_flatness': 0.1,
            'clahe_clip_limit': 2.0, 'clahe_tile_grid_size': (8, 8),
            'ridge_filter': 'hessian',
            'frangi_sigmas': (1, 5), 'frangi_scale_step': 0.5,
            'use_frangi_fallback': False, 'fallback_edge_threshold': 100,
            'time_dist_divisor': 50.0, 'freq_dist_divisor': 100.0,
            'size_score_divisor': 5.0, 'energy_score_divisor': 10.0,
            'time_weight': 5.0, 'freq_weight': 1.0,
            'size_weight': 1.0, 'energy_weight': 1.0,
        }
    
    # Precompute spectrogram
    S_db = compute_spectrogram_for_eval(
        y, sr, fmin=fixed_params['fmin'], fmax=fixed_params['fmax'],
        n_fft=fixed_params['n_fft'], hop_length=fixed_params['hop_length']
    )
    
    iteration_count = [0]
    
    def objective(params_array):
        iteration_count[0] += 1
        param_names = list(param_bounds.keys())
        param_dict = dict(zip(param_names, params_array))
        
        full_params = fixed_params.copy()
        full_params['impulse_freq_threshold'] = param_dict.get('impulse_freq_threshold', 0.4)
        full_params['ridge_threshold'] = param_dict.get('ridge_threshold', 0.3)
        
        if 'hessian_sigma_min' in param_dict and 'hessian_sigma_max' in param_dict:
            sigma_min = int(param_dict['hessian_sigma_min'])
            sigma_max = int(param_dict['hessian_sigma_max'])
            full_params['hessian_sigmas'] = (sigma_min, sigma_max)
        
        try:
            result = process_audio_file(audio_path, y=y, sr=sr, **full_params)
            quality = evaluate_contour_quality(result, S_db, weights=metric_weights)
            score = quality[target_metric]
            
            if verbose and iteration_count[0] % 10 == 0:
                print(f"Iteration {iteration_count[0]}: score={score:.4f}")
            
            return -score
        except Exception as e:
            if verbose:
                print(f"Error in iteration {iteration_count[0]}: {e}")
            return 1e6
    
    bounds_list = [param_bounds[name] for name in param_bounds.keys()]
    
    if verbose:
        print(f"Starting optimization: method={method}, target={target_metric}")
    
    if method == 'differential_evolution':
        opt_result = differential_evolution(
            objective, bounds_list, maxiter=100, popsize=15,
            strategy='best1bin', seed=42, disp=verbose
        )
    elif method == 'minimize':
        x0 = [(b[0] + b[1]) / 2 for b in bounds_list]
        opt_result = minimize(
            objective, x0, bounds=bounds_list,
            method='L-BFGS-B', options={'disp': verbose}
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Extract optimal parameters
    param_names = list(param_bounds.keys())
    optimal_params = dict(zip(param_names, opt_result.x))
    
    if 'hessian_sigma_min' in optimal_params:
        optimal_params['hessian_sigma_min'] = int(optimal_params['hessian_sigma_min'])
    if 'hessian_sigma_max' in optimal_params:
        optimal_params['hessian_sigma_max'] = int(optimal_params['hessian_sigma_max'])
    
    # Final evaluation
    full_params = fixed_params.copy()
    full_params.update(optimal_params)
    
    if 'hessian_sigma_min' in optimal_params and 'hessian_sigma_max' in optimal_params:
        full_params['hessian_sigmas'] = (optimal_params['hessian_sigma_min'], 
                                          optimal_params['hessian_sigma_max'])
    
    final_result = process_audio_file(audio_path, y=y, sr=sr, **full_params)
    final_quality = evaluate_contour_quality(final_result, S_db, weights=metric_weights)
    
    if verbose:
        print(f"\n{'='*60}\nOPTIMIZATION COMPLETE\n{'='*60}")
        print(f"Optimal {target_metric}: {final_quality[target_metric]:.4f}")
        print(f"Optimal parameters: {optimal_params}")
    
    return {
        'optimal_params': optimal_params,
        'optimal_score': final_quality[target_metric],
        'all_metrics': final_quality,
        'iterations': iteration_count[0],
        'scipy_result': opt_result,
        'contour_result': final_result
    }


def batch_optimize(audio_files: list, output_dir: Optional[Path] = None, **optimize_kwargs) -> Dict:
    """Optimize parameters for multiple audio files."""
    results = {}
    
    for i, audio_path in enumerate(audio_files):
        print(f"\n{'='*60}\nProcessing {i+1}/{len(audio_files)}: {Path(audio_path).name}\n{'='*60}")
        
        result = optimize_parameters(audio_path, **optimize_kwargs)
        results[audio_path] = result
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            save_path = output_dir / f"{Path(audio_path).stem}_optimization.json"
            save_data = {
                'audio_file': audio_path,
                'optimal_params': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                   for k, v in result['optimal_params'].items()},
                'optimal_score': float(result['optimal_score']),
                'all_metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                for k, v in result['all_metrics'].items()},
                'iterations': result['iterations']
            }
            
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"✓ Saved to {save_path}")
    
    return results



def plot_optimization_results(results_dict: Dict, audio_dir: Path, output_path: Optional[Path] = None):
    """
    Plot comparison of original vs optimized parameters.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary from batch_optimize() or optimize_parameters()
    audio_dir : Path
        Directory containing audio files
    output_path : Path, optional
        Save path for plot
    """
    from pathlib import Path
    import pysoniq
    from .processing import process_audio_file
    from .optimization import compute_spectrogram_for_eval, evaluate_contour_quality
    
    n_files = len(results_dict)
    fig, axes = plt.subplots(n_files, 3, figsize=(15, 5*n_files))
    
    if n_files == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (audio_path, opt_result) in enumerate(results_dict.items()):
        # Load audio
        y, sr = pysoniq.load_audio(audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        
        # Compute spectrogram
        S_db = compute_spectrogram_for_eval(y, sr)
        times = pysoniq.fourier.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=128)
        freqs = pysoniq.fourier.mel_frequencies(n_mels=S_db.shape[0], fmin=1000, fmax=8000)
        extent = [times[0], times[-1], freqs[0], freqs[-1]]
        
        # Original (default) parameters
        default_params = {
            'fmin': 1000, 'fmax': 8000, 'n_fft': 512, 'hop_length': 128,
            'denoise_impulses': True,
            'impulse_freq_threshold': 0.4,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.3,
            'hessian_sigmas': (1, 5)
        }
        
        original_result = process_audio_file(audio_path, y=y, sr=sr, **default_params)
        original_quality = evaluate_contour_quality(original_result, S_db)
        
        # Optimized result
        optimized_result = opt_result['contour_result']
        optimized_quality = opt_result['all_metrics']
        
        # Plot original
        axes[idx, 0].imshow(S_db, aspect='auto', origin='lower', extent=extent, 
                           cmap='magma', alpha=0.5)
        if original_result['component'].sum() > 0:
            axes[idx, 0].imshow(original_result['component'], aspect='auto', origin='lower',
                               extent=extent, cmap='hot', alpha=0.7)
        axes[idx, 0].set_title(f"Original\nScore: {original_quality['overall_score']:.3f}", fontsize=10)
        axes[idx, 0].set_ylabel('Frequency (Hz)')
        
        # Plot optimized
        axes[idx, 1].imshow(S_db, aspect='auto', origin='lower', extent=extent,
                           cmap='magma', alpha=0.5)
        if optimized_result['component'].sum() > 0:
            axes[idx, 1].imshow(optimized_result['component'], aspect='auto', origin='lower',
                               extent=extent, cmap='hot', alpha=0.7)
        axes[idx, 1].set_title(f"Optimized\nScore: {optimized_quality['overall_score']:.3f}", fontsize=10)
        
        # Plot improvement metrics
        metrics = ['time_coverage', 'compactness', 'energy_ratio', 'snr_normalized']
        original_vals = [original_quality[m] for m in metrics]
        optimized_vals = [optimized_quality[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[idx, 2].bar(x - width/2, original_vals, width, label='Original', alpha=0.7)
        axes[idx, 2].bar(x + width/2, optimized_vals, width, label='Optimized', alpha=0.7)
        axes[idx, 2].set_xticks(x)
        axes[idx, 2].set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=8)
        axes[idx, 2].set_ylim([0, 1])
        axes[idx, 2].set_title('Metric Comparison', fontsize=10)
        axes[idx, 2].legend()
        axes[idx, 2].grid(axis='y', alpha=0.3)
        
        # Add filename as text
        axes[idx, 0].text(0.02, 0.98, Path(audio_path).name, transform=axes[idx, 0].transAxes,
                         fontsize=8, va='top', ha='left', bbox=dict(boxstyle='round', 
                         facecolor='white', alpha=0.7))
    
    fig.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved optimization plot to {output_path}")
    
    plt.show()


    from contours.optimization import optimize_parameters, batch_optimize

# from contours.plotting import plot_optimization_results # located in this file for now

# # Single file
# result = optimize_parameters(
#     'path/to/audio.wav',
#     method='differential_evolution',
#     target_metric='overall_score',
#     verbose=True
# )

# # Multiple files
# results = batch_optimize(
#     ['file1.wav', 'file2.wav', 'file3.wav'],
#     output_dir='optimization_results/'
# )

# # Visualize
# plot_optimization_results(results, audio_dir='path/to/audio')