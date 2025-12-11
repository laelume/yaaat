"""
Harmonic Stacking and Compression Analysis (Analysis-Only Version)
Plotting removed - metrics and statistics preserved
This will be for refactoring and feeding into ML pipe
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import minimum_filter1d
from fuzzyvalley import FlexibleSpectralValleyTracker


def extract_harmonic_alignment(tracker, valleys, harmonic_tracks, 
                              # Analysis filtering parameters
                              analysis_fmin=100, analysis_fmax=16000,
                              # Plotting parameters  
                              fmin_plot=100, fmax_plot=16000,
                              # Valley detection parameters
                              valley_margin=0.25, min_gap=50,
                              # Boundary calculation parameters
                              boundary_smoothing_window=7):
    """
    Comprehensive harmonic alignment and stacking analysis
    
    Extracts harmonic ribbons using valley boundaries and performs 5 types of analysis:
    1. Harmonic Compression - Map all harmonics to F0 range
    2. Phase-Aligned Summation - Align harmonics by phase relationships
    3. Energy Ratio Analysis - Examine relative energy distribution
    4. Spectral Centroid Analysis - Compute spectral center of mass
    5. Harmonic Superposition - Stack harmonics to augment signal contour
    
    Parameters
    ----------
    tracker : FlexibleSpectralValleyTracker
        Tracker with spectrogram data
    valleys : dict
        Valley tracks from tracker.find_valleys_between_harmonics()
    harmonic_tracks : list
        Harmonic tracks from tracker.track_harmonics_with_template()
    analysis_fmin, analysis_fmax : float
        Frequency range for analysis
    fmin_plot, fmax_plot : float
        Frequency range for plotting (kept for compatibility)
    valley_margin : float
        Margin for valley detection
    min_gap : float
        Minimum gap between valleys
    boundary_smoothing_window : int
        Savitzky-Golay window length for boundary smoothing
        
    Returns
    -------
    dict
        Complete results including all analysis outputs and metrics
    """
    
    print("\n" + "="*80)
    print("HARMONIC ALIGNMENT ANALYSIS")
    print("="*80)
    
    # ====================================================================
    # STEP 1: Filter valley boundaries to analysis range
    # ====================================================================
    
    valley_boundaries = []
    
    for pair_key, valley_data in valleys.items():
        if valley_data:
            times = np.array([v['time'] for v in valley_data])
            freqs = np.array([v['frequency'] for v in valley_data])
            
            mask = (freqs >= analysis_fmin) & (freqs <= analysis_fmax)
            if np.sum(mask) > 3:
                times_filtered = times[mask]
                freqs_filtered = freqs[mask]
                
                if len(freqs_filtered) >= boundary_smoothing_window:
                    freqs_smooth = savgol_filter(freqs_filtered, 
                                               window_length=boundary_smoothing_window, 
                                               polyorder=2)
                else:
                    freqs_smooth = freqs_filtered
                
                valley_interp = np.interp(tracker.times, times_filtered, freqs_smooth)
                valley_boundaries.append(valley_interp)
    
    valley_boundaries = sorted(valley_boundaries, key=lambda x: np.mean(x))
    
    time_min = tracker.times[0]
    time_max = tracker.times[-1]
    
    # ====================================================================
    # STEP 2: Dynamic boundary detection
    # ====================================================================
    
    if len(valley_boundaries) > 0 and harmonic_tracks:
        first_valley = valley_boundaries[0]
        last_valley = valley_boundaries[-1]
        
        # Extract F0 (H1) harmonic ridge frequencies
        f0_ridge_freqs = []
        f0_times = []
        
        for frame_data in harmonic_tracks:
            for harmonic in frame_data['harmonics']:
                if harmonic['harmonic_number'] == 1:
                    f0_ridge_freqs.append(harmonic['actual_frequency'])
                    f0_times.append(frame_data['time'])
                    break
        
        if f0_ridge_freqs:
            # Interpolate F0 ridge to time grid
            f0_ridge_interp = np.interp(tracker.times, f0_times, f0_ridge_freqs)
            
            # Find valley between fmin_plot and F0 ridge
            lower_valley_data = []
            
            for t_idx in range(len(tracker.times)):
                search_start_freq = fmin_plot
                search_end_freq = f0_ridge_interp[t_idx]
                
                search_start_idx = np.searchsorted(tracker.freqs, search_start_freq)
                search_end_idx = np.searchsorted(tracker.freqs, search_end_freq)
                
                if search_end_idx > search_start_idx:
                    valley_spectrum = tracker.log_magnitude[search_start_idx:search_end_idx, t_idx]
                    valley_local_idx = np.argmin(valley_spectrum)
                    valley_freq_idx = search_start_idx + valley_local_idx
                    lower_valley_data.append(tracker.freqs[valley_freq_idx])
                else:
                    lower_valley_data.append(search_start_freq)
            
            # Smooth lower boundary
            if len(lower_valley_data) >= boundary_smoothing_window:
                dynamic_lower = savgol_filter(lower_valley_data, 
                                             window_length=boundary_smoothing_window, 
                                             polyorder=2)
            else:
                dynamic_lower = np.array(lower_valley_data)
        else:
            dynamic_lower = np.full_like(tracker.times, fmin_plot)
        
        # Find valley above highest harmonic
        max_harm_freqs = []
        max_harm_times = []
        
        for frame_data in harmonic_tracks:
            if frame_data['harmonics']:
                highest_harmonic = max(frame_data['harmonics'], 
                                      key=lambda h: h['actual_frequency'])
                max_harm_freqs.append(highest_harmonic['actual_frequency'])
                max_harm_times.append(frame_data['time'])
        
        if max_harm_freqs:
            max_harm_interp = np.interp(tracker.times, max_harm_times, max_harm_freqs)
            
            # Find valley between highest harmonic and fmax_plot
            upper_valley_data = []
            
            for t_idx in range(len(tracker.times)):
                search_start_freq = max_harm_interp[t_idx]
                search_end_freq = fmax_plot
                
                search_start_idx = np.searchsorted(tracker.freqs, search_start_freq)
                search_end_idx = np.searchsorted(tracker.freqs, search_end_freq)
                
                if search_end_idx > search_start_idx:
                    valley_spectrum = tracker.log_magnitude[search_start_idx:search_end_idx, t_idx]
                    valley_local_idx = np.argmin(valley_spectrum)
                    valley_freq_idx = search_start_idx + valley_local_idx
                    upper_valley_data.append(tracker.freqs[valley_freq_idx])
                else:
                    upper_valley_data.append(search_end_freq)
            
            # Smooth upper boundary
            if len(upper_valley_data) >= boundary_smoothing_window:
                dynamic_upper = savgol_filter(upper_valley_data, 
                                             window_length=boundary_smoothing_window, 
                                             polyorder=2)
            else:
                dynamic_upper = np.array(upper_valley_data)
        else:
            dynamic_upper = np.full_like(tracker.times, fmax_plot)
        
        dynamic_upper = np.minimum(dynamic_upper, fmax_plot)
    else:
        dynamic_lower = np.full_like(tracker.times, fmin_plot)
        dynamic_upper = np.full_like(tracker.times, fmax_plot)
    
    boundaries_with_edges = [dynamic_lower] + valley_boundaries + [dynamic_upper]
    
    # ====================================================================
    # STEP 3: Harmonic ribbon extraction
    # ====================================================================
    
    harmonic_ribbons = []
    n_ribbons = len(boundaries_with_edges) - 1
    
    for ribbon_idx in range(n_ribbons):
        ribbon_lower = boundaries_with_edges[ribbon_idx]
        ribbon_upper = boundaries_with_edges[ribbon_idx + 1]
        
        ribbon_spectrogram = []
        ribbon_freqs_list = []
        
        for t_idx in range(len(tracker.times)):
            lower_freq_idx = np.searchsorted(tracker.freqs, ribbon_lower[t_idx])
            upper_freq_idx = np.searchsorted(tracker.freqs, ribbon_upper[t_idx])
            
            if upper_freq_idx > lower_freq_idx:
                ribbon_spectrum = tracker.magnitude[lower_freq_idx:upper_freq_idx, t_idx]
                ribbon_freqs = tracker.freqs[lower_freq_idx:upper_freq_idx]
                ribbon_spectrogram.append(ribbon_spectrum)
                ribbon_freqs_list.append(ribbon_freqs)
            else:
                ribbon_spectrogram.append(np.array([]))
                ribbon_freqs_list.append(np.array([]))
        
        harmonic_ribbons.append((ribbon_spectrogram, ribbon_freqs_list))
    
    print(f"Extracted {n_ribbons} harmonic ribbons")
    
    # ====================================================================
    # STEP 4: Extract F0 track for alignment
    # ====================================================================
    
    f0_track_freqs = []
    f0_track_times = []
    
    for frame_data in harmonic_tracks:
        for harmonic in frame_data['harmonics']:
            if harmonic['harmonic_number'] == 1:
                f0_track_freqs.append(harmonic['actual_frequency'])
                f0_track_times.append(frame_data['time'])
                break
    
    if f0_track_freqs:
        f0_interp = np.interp(tracker.times, f0_track_times, f0_track_freqs)
    else:
        f0_interp = (boundaries_with_edges[0] + boundaries_with_edges[1]) / 2
    
    # ====================================================================
    # ANALYSIS 1: Harmonic Compression
    # ====================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS 1: HARMONIC COMPRESSION")
    print("="*80)
    
    f0_lower = boundaries_with_edges[0]
    f0_upper = boundaries_with_edges[1]
    
    f0_min = np.min(f0_lower)
    f0_max = np.max(f0_upper)
    freq_resolution = tracker.freqs[1] - tracker.freqs[0]
    n_compressed_bins = int((f0_max - f0_min) / freq_resolution)
    compressed_freq_axis = np.linspace(f0_min, f0_max, n_compressed_bins)
    
    compressed_harmonics = []
    compressed_harmonics_db = []
    
    for ribbon_idx in range(n_ribbons):
        ribbon_specs, ribbon_freqs_list = harmonic_ribbons[ribbon_idx]
        
        compressed_magnitude = np.zeros((n_compressed_bins, len(tracker.times)))
        
        for t_idx in range(len(tracker.times)):
            if len(ribbon_specs[t_idx]) > 0:
                actual_freqs = ribbon_freqs_list[t_idx]
                actual_spectrum = ribbon_specs[t_idx]
                
                if len(actual_freqs) > 1:
                    harm_min = np.min(actual_freqs)
                    harm_max = np.max(actual_freqs)
                    
                    if harm_max > harm_min:
                        normalized_positions = (actual_freqs - harm_min) / (harm_max - harm_min)
                        mapped_freqs = f0_min + normalized_positions * (f0_max - f0_min)
                        
                        compressed_spectrum = np.interp(compressed_freq_axis,
                                                       mapped_freqs,
                                                       actual_spectrum,
                                                       left=0, right=0)
                        
                        compressed_magnitude[:, t_idx] = compressed_spectrum
        
        compressed_harmonics.append(compressed_magnitude)
        compressed_harmonics_db.append(20 * np.log10(compressed_magnitude + 1e-10))
    
    summed_compressed = np.sum(compressed_harmonics, axis=0)
    summed_compressed_db = 20 * np.log10(summed_compressed + 1e-10)
    
    # Compression metrics
    compression_stats = {
        'n_ribbons': n_ribbons,
        'f0_range': (f0_min, f0_max),
        'compressed_freq_axis': compressed_freq_axis,
        'mean_compressed_magnitudes': []
    }
    
    for i in range(n_ribbons):
        mean_compressed = np.mean(compressed_harmonics[i][compressed_harmonics[i] > 0])
        compression_stats['mean_compressed_magnitudes'].append(mean_compressed)
        print(f"   H{i+1} mean compressed magnitude: {mean_compressed:.4f}")
    
    # ====================================================================
    # ANALYSIS 2: Phase-aligned summation
    # ====================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS 2: PHASE-ALIGNED SUMMATION")
    print("="*80)
    
    phase_aligned_magnitude = np.zeros_like(tracker.magnitude)
    harmonic_peak_phases = []
    
    for ribbon_idx in range(n_ribbons):
        ribbon_specs, ribbon_freqs_list = harmonic_ribbons[ribbon_idx]
        harmonic_number = ribbon_idx + 1
        
        peak_phases = []
        
        for t_idx in range(len(tracker.times)):
            if len(ribbon_specs[t_idx]) > 0:
                spectrum = ribbon_specs[t_idx]
                freqs = ribbon_freqs_list[t_idx]
                
                if len(spectrum) > 0:
                    peak_idx = np.argmax(spectrum)
                    peak_freq = freqs[peak_idx]
                    
                    if t_idx < len(f0_interp):
                        f0_freq = f0_interp[t_idx]
                        expected_freq = f0_freq * harmonic_number
                        freq_deviation = peak_freq - expected_freq
                        phase_shift = freq_deviation / expected_freq * 2 * np.pi
                        
                        freq_shift_bins = int(freq_deviation / freq_resolution)
                        target_start_idx = np.searchsorted(tracker.freqs, expected_freq - (peak_freq - freqs[0]))
                        
                        for i, (f, mag) in enumerate(zip(freqs, spectrum)):
                            target_idx = target_start_idx + i
                            if 0 <= target_idx < len(tracker.freqs):
                                harmonic_weight = 1.0 / np.sqrt(harmonic_number)
                                phase_aligned_magnitude[target_idx, t_idx] += mag * harmonic_weight
                    
                    peak_phases.append(peak_freq)
                else:
                    peak_phases.append(np.nan)
            else:
                peak_phases.append(np.nan)
        
        harmonic_peak_phases.append(peak_phases)
    
    phase_aligned_db = 20 * np.log10(phase_aligned_magnitude + 1e-10)
    
    # Phase alignment metrics
    phase_aligned_energy = np.sum(phase_aligned_magnitude**2)
    original_energy = np.sum(tracker.magnitude**2)
    phase_retention = phase_aligned_energy / original_energy * 100
    
    phase_stats = {
        'energy_retention_pct': phase_retention,
        'peak_phases': harmonic_peak_phases
    }
    
    print(f"   Phase-aligned energy retention: {phase_retention:.1f}%")
    
    # ====================================================================
    # ANALYSIS 3: Energy ratio analysis
    # ====================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS 3: ENERGY RATIO ANALYSIS")
    print("="*80)
    
    harmonic_energies = []
    
    for ribbon_idx in range(n_ribbons):
        ribbon_specs, _ = harmonic_ribbons[ribbon_idx]
        
        energy_trace = np.zeros(len(tracker.times))
        
        for t_idx in range(len(tracker.times)):
            if len(ribbon_specs[t_idx]) > 0:
                energy = np.sum(ribbon_specs[t_idx]**2)
                energy_trace[t_idx] = energy
        
        harmonic_energies.append(energy_trace)
    
    # Calculate ratios relative to fundamental
    harmonic_ratios = []
    harmonic_ratios_db = []
    
    fundamental_energy = harmonic_energies[0] + 1e-10
    
    for ribbon_idx in range(n_ribbons):
        ratio = harmonic_energies[ribbon_idx] / fundamental_energy
        ratio_db = 10 * np.log10(ratio + 1e-10)
        
        harmonic_ratios.append(ratio)
        harmonic_ratios_db.append(ratio_db)
    
    # Calculate spectral rolloff metrics
    rolloff_percentiles = [25, 50, 75, 90]
    rolloff_traces = {p: [] for p in rolloff_percentiles}
    
    for t_idx in range(len(tracker.times)):
        cumulative_energy = 0
        total_energy = sum(h[t_idx] for h in harmonic_energies)
        
        if total_energy > 0:
            for percentile in rolloff_percentiles:
                target_energy = total_energy * (percentile / 100.0)
                cumsum = 0
                
                for ribbon_idx, energy in enumerate([h[t_idx] for h in harmonic_energies]):
                    cumsum += energy
                    if cumsum >= target_energy:
                        harmonic_number = ribbon_idx + 1
                        rolloff_freq = f0_interp[t_idx] * harmonic_number if t_idx < len(f0_interp) else 0
                        rolloff_traces[percentile].append(rolloff_freq)
                        break
                else:
                    rolloff_traces[percentile].append(np.nan)
        else:
            for percentile in rolloff_percentiles:
                rolloff_traces[percentile].append(np.nan)
    
    # Energy ratio metrics
    energy_stats = {
        'harmonic_energies': harmonic_energies,
        'harmonic_ratios_db': harmonic_ratios_db,
        'rolloff_traces': rolloff_traces,
        'mean_ratios_db': []
    }
    
    for i in range(min(5, n_ribbons)):
        mean_ratio = np.mean(harmonic_ratios_db[i])
        energy_stats['mean_ratios_db'].append(mean_ratio)
        print(f"   H{i+1}/H1 mean ratio: {mean_ratio:.1f} dB")
    
    # ====================================================================
    # ANALYSIS 4: Spectral centroid
    # ====================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS 4: SPECTRAL CENTROID ANALYSIS")
    print("="*80)
    
    individual_centroids = []
    
    for ribbon_idx in range(n_ribbons):
        ribbon_specs, ribbon_freqs_list = harmonic_ribbons[ribbon_idx]
        
        centroid_trace = np.zeros(len(tracker.times))
        
        for t_idx in range(len(tracker.times)):
            if len(ribbon_specs[t_idx]) > 0:
                spectrum = ribbon_specs[t_idx]
                freqs = ribbon_freqs_list[t_idx]
                
                total_mag = np.sum(spectrum)
                if total_mag > 0:
                    centroid = np.sum(freqs * spectrum) / total_mag
                    centroid_trace[t_idx] = centroid
                else:
                    centroid_trace[t_idx] = np.mean(freqs) if len(freqs) > 0 else np.nan
            else:
                centroid_trace[t_idx] = np.nan
        
        individual_centroids.append(centroid_trace)
    
    # Overall spectral centroid
    overall_centroid = np.zeros(len(tracker.times))
    spectral_spread = np.zeros(len(tracker.times))
    spectral_skewness = np.zeros(len(tracker.times))
    
    for t_idx in range(len(tracker.times)):
        all_freqs = []
        all_mags = []
        
        for ribbon_idx in range(n_ribbons):
            if len(harmonic_ribbons[ribbon_idx][0][t_idx]) > 0:
                all_freqs.extend(harmonic_ribbons[ribbon_idx][1][t_idx])
                all_mags.extend(harmonic_ribbons[ribbon_idx][0][t_idx])
        
        if all_freqs:
            all_freqs = np.array(all_freqs)
            all_mags = np.array(all_mags)
            
            total_mag = np.sum(all_mags)
            if total_mag > 0:
                centroid = np.sum(all_freqs * all_mags) / total_mag
                overall_centroid[t_idx] = centroid
                
                variance = np.sum(all_mags * (all_freqs - centroid)**2) / total_mag
                spectral_spread[t_idx] = np.sqrt(variance)
                
                if spectral_spread[t_idx] > 0:
                    skewness = np.sum(all_mags * (all_freqs - centroid)**3) / (total_mag * spectral_spread[t_idx]**3)
                    spectral_skewness[t_idx] = skewness
    
    # Centroid metrics
    valid_mask = overall_centroid > 0
    mean_centroid = np.mean(overall_centroid[valid_mask])
    mean_spread = np.mean(spectral_spread[spectral_spread > 0])
    mean_skewness = np.mean(spectral_skewness[~np.isnan(spectral_skewness)])
    
    centroid_stats = {
        'individual_centroids': individual_centroids,
        'overall_centroid': overall_centroid,
        'spectral_spread': spectral_spread,
        'spectral_skewness': spectral_skewness,
        'mean_centroid': mean_centroid,
        'mean_spread': mean_spread,
        'mean_skewness': mean_skewness
    }
    
    print(f"   Mean centroid: {mean_centroid:.1f} Hz")
    print(f"   Mean spread: {mean_spread:.1f} Hz")
    print(f"   Mean skewness: {mean_skewness:.3f}")
    
    # ====================================================================
    # ANALYSIS 5: Harmonic superposition
    # ====================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS 5: HARMONIC SUPERPOSITION")
    print("="*80)
    
    superposed_freq_axis = compressed_freq_axis
    superimposed_magnitude = np.zeros((n_compressed_bins, len(tracker.times)))
    
    for ribbon_idx in range(n_ribbons):
        ribbon_specs, ribbon_freqs_list = harmonic_ribbons[ribbon_idx]
        ribbon_lower = boundaries_with_edges[ribbon_idx]
        ribbon_upper = boundaries_with_edges[ribbon_idx + 1]
        
        for t_idx in range(len(tracker.times)):
            if len(ribbon_specs[t_idx]) > 0:
                spectrum = ribbon_specs[t_idx]
                freqs = ribbon_freqs_list[t_idx]
                
                ribbon_min = ribbon_lower[t_idx]
                ribbon_max = ribbon_upper[t_idx]
                ribbon_range = ribbon_max - ribbon_min
                
                f0_min_t = f0_lower[t_idx]
                f0_max_t = f0_upper[t_idx]
                f0_range = f0_max_t - f0_min_t
                
                if ribbon_range > 0 and f0_range > 0:
                    normalized = (freqs - ribbon_min) / ribbon_range
                    mapped_freqs = f0_min_t + normalized * f0_range
                    
                    interp_spectrum = np.interp(superposed_freq_axis,
                                               mapped_freqs,
                                               spectrum,
                                               left=0, right=0)
                    
                    superimposed_magnitude[:, t_idx] += interp_spectrum
    
    superimposed_db = 20 * np.log10(superimposed_magnitude + 1e-10)
    
    # F0 only for comparison
    f0_specs, f0_freqs = harmonic_ribbons[0]
    f0_image = np.zeros((n_compressed_bins, len(tracker.times)))
    
    for t_idx, (spec, freqs) in enumerate(zip(f0_specs, f0_freqs)):
        if len(spec) > 0:
            f0_interp_spec = np.interp(superposed_freq_axis, freqs, spec, left=0, right=0)
            f0_image[:, t_idx] = f0_interp_spec
    
    f0_db = 20 * np.log10(f0_image + 1e-10)
    enhancement = superimposed_db - f0_db
    
    # Superposition metrics
    superposition_stats = {
        'peak_augmentation_db': np.max(enhancement),
        'mean_augmentation_db': np.mean(enhancement),
        'f0_only_db': f0_db
    }
    
    print(f"   Superposition complete: {n_ribbons} harmonics stacked")
    print(f"   Peak augmentation: {np.max(enhancement):.1f} dB")
    print(f"   Mean augmentation: {np.mean(enhancement):.1f} dB")
    
    # ====================================================================
    # Summary statistics
    # ====================================================================
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Ribbon energy statistics
    print("\nHarmonic Ribbon Energy Statistics:")
    print("-" * 50)
    for ribbon_idx, energy_trace in enumerate(harmonic_energies):
        mean_energy = np.mean(energy_trace)
        max_energy = np.max(energy_trace)
        total_energy_ribbon = np.sum(energy_trace)
        
        label = f'H{ribbon_idx+1}'
        print(f"{label:5} - Mean: {mean_energy:10.2f}, Max: {max_energy:10.2f}, Total: {total_energy_ribbon:10.2f}")
    
    print(f"\nTotal System Energy: {np.sum(harmonic_energies):10.2f}")
    
    # ====================================================================
    # Return complete results
    # ====================================================================
    
    results = {
        # Raw data
        'boundaries': boundaries_with_edges,
        'harmonic_ribbons': harmonic_ribbons,
        'f0_interp': f0_interp,
        'time_range': (time_min, time_max),
        
        # Compression results
        'compressed_harmonics': compressed_harmonics,
        'compressed_harmonics_db': compressed_harmonics_db,
        'summed_compressed': summed_compressed,
        'summed_compressed_db': summed_compressed_db,
        'compressed_freq_axis': compressed_freq_axis,
        'compression_stats': compression_stats,
        
        # Phase alignment results
        'phase_aligned_magnitude': phase_aligned_magnitude,
        'phase_aligned_db': phase_aligned_db,
        'harmonic_peak_phases': harmonic_peak_phases,
        'phase_stats': phase_stats,
        
        # Energy ratio results
        'harmonic_energies': harmonic_energies,
        'harmonic_ratios': harmonic_ratios,
        'harmonic_ratios_db': harmonic_ratios_db,
        'rolloff_traces': rolloff_traces,
        'energy_stats': energy_stats,
        
        # Centroid results
        'individual_centroids': individual_centroids,
        'overall_centroid': overall_centroid,
        'spectral_spread': spectral_spread,
        'spectral_skewness': spectral_skewness,
        'centroid_stats': centroid_stats,
        
        # Superposition results
        'superimposed_magnitude': superimposed_magnitude,
        'superimposed_db': superimposed_db,
        'f0_db': f0_db,
        'enhancement': enhancement,
        'superposition_stats': superposition_stats
    }
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return results


# tracker = FlexibleSpectralValleyTracker()

# results = extract_harmonic_alignment(
#     tracker, valleys, harm_tracks,
#     analysis_fmin=500,
#     analysis_fmax=16000,
#     boundary_smoothing_window=30
# )

# # Access any result:
# summed_compressed_db = results['summed_compressed_db']
# superimposed_db = results['superimposed_db']
# compression_stats = results['compression_stats']
# energy_stats = results['energy_stats']