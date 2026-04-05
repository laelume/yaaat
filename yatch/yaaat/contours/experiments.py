"""Test case experiments for comparing contour extraction methods."""

import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict

from .processing import process_audio_file, process_directory
from .plotting import plot_multi_comparison
from .config import DEFAULT_CONFIG, NO_DENOISE_CONFIG, DTRAIN_CONFIG, IMPULSE_CONFIG


def run_single_file_comparison(
    audio_file: str,
    output_dir: Optional[Path] = None,
    show_plots: bool = True
) -> Dict:
    """
    Run comprehensive comparison on a single audio file.
    
    Tests 7 different configurations:
    1. Hessian without denoising
    2. Hessian with denoising
    3. Hessian with lower threshold (0.2)
    4. Hessian with Frangi fallback
    5. Frangi filter (threshold 0.1)
    6. Frangi filter (threshold 0.15)
    7. Hessian with wider sigma range (1-10)
    
    Parameters
    ----------
    audio_file : str
        Path to audio file
    output_dir : Path, optional
        Directory to save results
    show_plots : bool, default=True
        Display plots
    
    Returns
    -------
    results : OrderedDict
        Results from all configurations
    """
    print("="*80)
    print(f"SINGLE FILE COMPARISON: {os.path.basename(audio_file)}")
    print("="*80)
    
    # Base parameters WITH denoising
    base_params = DEFAULT_CONFIG.to_dict()
    
    # Base parameters WITHOUT denoising
    base_params_no_denoise = NO_DENOISE_CONFIG.to_dict()
    
    # Extract denoise params for plot visualization
    denoise_params = {
        'freq_threshold': base_params['impulse_freq_threshold'],
        'intensity_threshold_db': base_params['impulse_intensity_threshold_db'],
        'max_gap': 3,
        'expand': 2
    }
    
    # Create ordered dict to control column order
    results = OrderedDict()
    
    filename = os.path.basename(audio_file)
    
    print("\n1. Processing Hessian (NO denoise)...")
    results['Hessian No Denoise'] = {
        filename: process_audio_file(audio_file, **{
            **base_params_no_denoise,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.3,
        })
    }
    
    print("\n2. Processing Hessian (WITH denoise)...")
    results['Hessian Denoise'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.3,
        })
    }
    
    print("\n3. Processing Hessian (thresh=0.2)...")
    results['Hessian 0.2'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.2,
        })
    }
    
    print("\n4. Processing Hessian + Fallback...")
    results['Hessian Fallback'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.3,
            'use_frangi_fallback': True,
            'fallback_edge_threshold': 100,
        })
    }
    
    print("\n5. Processing Frangi (thresh=0.1)...")
    results['Frangi 0.1'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'frangi',
            'ridge_threshold': 0.1,
        })
    }
    
    print("\n6. Processing Frangi (thresh=0.15)...")
    results['Frangi 0.15'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'frangi',
            'ridge_threshold': 0.15,
        })
    }
    
    print("\n7. Processing Hessian (σ=1-10)...")
    results['Hessian σ:1-10'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.2,
            'hessian_sigmas': (1, 10),
        })
    }
    
    # Print summary
    print("\n" + "="*80)
    print("PIXEL COUNT SUMMARY")
    print("="*80)
    print(f"\n{filename}:")
    for config_name, config_results in results.items():
        px = config_results[filename]['n_pixels']
        fallback = "*" if config_results[filename].get('used_fallback') else " "
        print(f"  {config_name.replace(chr(10), ' ')}: {px:4d} px {fallback}")
    
    # Plot comparison
    file_dir = os.path.dirname(audio_file)
    output_path = None
    if output_dir:
        output_path = output_dir / f"{os.path.splitext(filename)[0]}_comparison.png"
    
    plot_multi_comparison(
        results, 
        file_dir, 
        denoise_params=denoise_params,
        output_path=output_path,
        show_plots=show_plots
    )
    
    return results


def run_directory_comparison(
    audio_dir: str,
    output_dir: Optional[Path] = None,
    show_plots: bool = False,
    n_energy_peaks: int = 5
) -> Dict:
    """
    Run comprehensive comparison on all files in a directory.
    
    Tests same 7 configurations as single file comparison.
    
    Parameters
    ----------
    audio_dir : str
        Path to directory containing audio files
    output_dir : Path, optional
        Directory to save results
    show_plots : bool, default=False
        Display plots
    n_energy_peaks : int, default=5
        Number of energy peaks to mark in plots
    
    Returns
    -------
    results : OrderedDict
        Results from all configurations
    """
    print("="*80)
    print("COMPREHENSIVE DIRECTORY COMPARISON")
    print("="*80)
    
    # Base parameters WITH denoising (implementation_1 uses higher fmax)
    base_params = {
        'fmin': 1000, 
        'fmax': 16000, 
        'n_fft': 512, 
        'hop_length': 128,
        'denoise_impulses': True,
        'impulse_freq_threshold': 0.7,
        'impulse_intensity_threshold_db': 10,
    }
    
    # Denoising params for visualization
    denoise_params = {
        'freq_threshold': 0.9, 
        'intensity_threshold_db': 10,
        'max_gap': 3,
        'expand': 2
    }

    # Base parameters WITHOUT denoising
    base_params_no_denoise = {
        'fmin': 1000, 
        'fmax': 16000, 
        'n_fft': 512, 
        'hop_length': 128,
        'denoise_impulses': False,
    }

    # Create ordered dict to control column order
    results = OrderedDict()
    
    print("\n1. Processing Hessian (NO denoise, thresh=0.3)...")
    results['Hessian No Denoise'] = process_directory(audio_dir, **{
        **base_params_no_denoise,
        'ridge_filter': 'hessian',
        'ridge_threshold': 0.3,
    })
    
    print("\n2. Processing Hessian (WITH denoise, thresh=0.3)...")
    results['Hessian Denoise'] = process_directory(audio_dir, **{
        **base_params,
        'ridge_filter': 'hessian',
        'ridge_threshold': 0.3,
    })
    
    print("\n3. Processing Hessian (denoise, thresh=0.2)...")
    results['Hessian 0.2'] = process_directory(audio_dir, **{
        **base_params,
        'ridge_filter': 'hessian',
        'ridge_threshold': 0.2,
    })
    
    print("\n4. Processing Hessian + Frangi Fallback...")
    results['Hessian Fallback'] = process_directory(audio_dir, **{
        **base_params,
        'ridge_filter': 'hessian',
        'ridge_threshold': 0.3,
        'use_frangi_fallback': True,
        'fallback_edge_threshold': 100,
    })
    
    print("\n5. Processing Frangi (denoise, thresh=0.1)...")
    results['Frangi 0.1'] = process_directory(audio_dir, **{
        **base_params,
        'ridge_filter': 'frangi',
        'ridge_threshold': 0.1,
    })
    
    print("\n6. Processing Frangi (denoise, thresh=0.15)...")
    results['Frangi 0.15'] = process_directory(audio_dir, **{
        **base_params,
        'ridge_filter': 'frangi',
        'ridge_threshold': 0.15,
    })
    
    print("\n7. Processing Hessian (denoise, σ=1-10)...")
    results['Hessian σ:1-10'] = process_directory(audio_dir, **{
        **base_params,
        'ridge_filter': 'hessian',
        'ridge_threshold': 0.2,
        'hessian_sigmas': (1, 10),
    })
    
    # Print summary comparison
    print("\n" + "="*80)
    print("PIXEL COUNT SUMMARY")
    print("="*80)
    
    filenames = list(results['Hessian No Denoise'].keys())
    for filename in filenames:
        print(f"\n{filename}:")
        for config_name, config_results in results.items():
            px = config_results[filename]['n_pixels']
            fallback = "*" if config_results[filename].get('used_fallback') else " "
            print(f"  {config_name.replace(chr(10), ' ')}: {px:4d} px {fallback}")
    
    # Plot comprehensive comparison
    dir_basename = os.path.basename(audio_dir)
    output_path = None
    if output_dir:
        output_path = output_dir / f"{dir_basename}_comparison.png"
    
    plot_multi_comparison(
        results, 
        audio_dir, 
        denoise_params=denoise_params,
        n_energy_peaks=n_energy_peaks, 
        show_titles=False, 
        output_path=output_path, 
        show_plots=show_plots, 
        verbose=False
    )
    
    return results


def run_dtrain_test(audio_file: str, output_dir: Optional[Path] = None) -> Dict:
    """
    Test case for d-train vocalizations with sharp transitions.
    
    Uses more aggressive temporal bridging to handle discontinuities.
    
    Parameters
    ----------
    audio_file : str
        Path to d-train audio file
    output_dir : Path, optional
        Directory to save results
    
    Returns
    -------
    results : OrderedDict
        Results from all configurations
    """
    print("="*80)
    print(f"D-TRAIN TEST: {os.path.basename(audio_file)}")
    print("="*80)
    
    # Use d-train configuration with aggressive temporal bridging
    base_params = DTRAIN_CONFIG.to_dict()
    base_params_no_denoise = NO_DENOISE_CONFIG.to_dict()
    
    denoise_params = {
        'freq_threshold': base_params['impulse_freq_threshold'],
        'intensity_threshold_db': base_params['impulse_intensity_threshold_db'],
        'max_gap': 3,
        'expand': 2
    }
    
    results = OrderedDict()
    filename = os.path.basename(audio_file)
    
    print("\n1. Processing Hessian (NO denoise)...")
    results['Hessian No Denoise'] = {
        filename: process_audio_file(audio_file, **{
            **base_params_no_denoise,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.2,
        })
    }
    
    print("\n2. Processing Hessian (WITH denoise)...")
    results['Hessian Denoise'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.3,
        })
    }
    
    print("\n3. Processing Hessian (thresh=0.2)...")
    results['Hessian 0.2'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.2,
        })
    }
    
    print("\n4. Processing Hessian + Fallback...")
    results['Hessian Fallback'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.3,
            'use_frangi_fallback': True,
            'fallback_edge_threshold': 100,
        })
    }
    
    print("\n5. Processing Frangi (thresh=0.2)...")
    results['Frangi 0.2'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'frangi',
            'ridge_threshold': 0.2,
        })
    }
    
    print("\n6. Processing Frangi (thresh=0.15)...")
    results['Frangi 0.15'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'frangi',
            'ridge_threshold': 0.15,
        })
    }
    
    print("\n7. Processing Hessian (σ=1-10)...")
    results['Hessian σ:1-10'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.2,
            'hessian_sigmas': (1, 10),
        })
    }
    
    # Print summary
    print("\n" + "="*80)
    print("PIXEL COUNT SUMMARY")
    print("="*80)
    print(f"\n{filename}:")
    for config_name, config_results in results.items():
        px = config_results[filename]['n_pixels']
        fallback = "*" if config_results[filename].get('used_fallback') else " "
        print(f"  {config_name.replace(chr(10), ' ')}: {px:4d} px {fallback}")
    
    file_dir = os.path.dirname(audio_file)
    output_path = None
    if output_dir:
        output_path = output_dir / f"{os.path.splitext(filename)[0]}_dtrain_test.png"
    
    plot_multi_comparison(results, file_dir, denoise_params=denoise_params, output_path=output_path)
    
    return results


def run_impulse_test(audio_file: str, output_dir: Optional[Path] = None) -> Dict:
    """
    Test case for impulse-contaminated recordings.
    
    Uses lower frequency threshold for sensitive impulse detection.
    
    Parameters
    ----------
    audio_file : str
        Path to impulse-contaminated audio file
    output_dir : Path, optional
        Directory to save results
    
    Returns
    -------
    results : OrderedDict
        Results from all configurations
    """
    print("="*80)
    print(f"IMPULSE TEST: {os.path.basename(audio_file)}")
    print("="*80)
    
    # Use impulse configuration with lower frequency threshold
    base_params = IMPULSE_CONFIG.to_dict()
    base_params_no_denoise = NO_DENOISE_CONFIG.to_dict()
    
    denoise_params = {
        'freq_threshold': base_params['impulse_freq_threshold'],
        'intensity_threshold_db': base_params['impulse_intensity_threshold_db'],
        'max_gap': 3,
        'expand': 2
    }
    
    results = OrderedDict()
    filename = os.path.basename(audio_file)
    
    print("\n1. Processing Hessian (NO denoise)...")
    results['Hessian No Denoise'] = {
        filename: process_audio_file(audio_file, **{
            **base_params_no_denoise,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.3,
        })
    }
    
    print("\n2. Processing Hessian (WITH denoise)...")
    results['Hessian Denoise'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.3,
        })
    }
    
    print("\n3. Processing Hessian (thresh=0.2)...")
    results['Hessian 0.2'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.2,
        })
    }
    
    print("\n4. Processing Hessian + Fallback...")
    results['Hessian Fallback'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.3,
            'use_frangi_fallback': True,
            'fallback_edge_threshold': 100,
        })
    }
    
    print("\n5. Processing Frangi (thresh=0.1)...")
    results['Frangi 0.1'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'frangi',
            'ridge_threshold': 0.1,
        })
    }
    
    print("\n6. Processing Frangi (thresh=0.15)...")
    results['Frangi 0.15'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'frangi',
            'ridge_threshold': 0.15,
        })
    }
    
    print("\n7. Processing Hessian (σ=1-10)...")
    results['Hessian σ:1-10'] = {
        filename: process_audio_file(audio_file, **{
            **base_params,
            'ridge_filter': 'hessian',
            'ridge_threshold': 0.2,
            'hessian_sigmas': (1, 10),
        })
    }
    
    # Print summary
    print("\n" + "="*80)
    print("PIXEL COUNT SUMMARY")
    print("="*80)
    print(f"\n{filename}:")
    for config_name, config_results in results.items():
        px = config_results[filename]['n_pixels']
        fallback = "*" if config_results[filename].get('used_fallback') else " "
        print(f"  {config_name.replace(chr(10), ' ')}: {px:4d} px {fallback}")
    
    file_dir = os.path.dirname(audio_file)
    output_path = None
    if output_dir:
        output_path = output_dir / f"{os.path.splitext(filename)[0]}_impulse_test.png"
    
    plot_multi_comparison(results, file_dir, denoise_params=denoise_params, output_path=output_path)
    
    return results