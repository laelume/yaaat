

import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
import pysoniq

try:
    from yaaat import utils
except ImportError:
    import utils

# ============================================================================
# FUZZY VALLEY BACKEND
# ============================================================================

class FuzzyValley:
    def __init__(self):
        """Initialize without hardcoded parameters"""
        self.y = None
        self.sr = None
        self.magnitude = None
        self.log_magnitude = None
        self.freqs = None
        self.times = None
        self.hop_length = None
        
    def load_audio(self, filepath, sr=None):
        """Load audio file"""
        self.y, self.sr = pysoniq.load(str(filepath))
        if self.y.ndim > 1:
            self.y = np.mean(self.y, axis=1)
        return self.y, self.sr
    
    def compute_spectrogram(self, n_fft=2048, hop_length=None, window='hann'):
        """Compute spectrogram"""
        if hop_length is None:
            hop_length = n_fft // 4
        
        from scipy import signal
        
        if window == 'hann':
            win = np.hanning(n_fft)
        elif window == 'hamming':
            win = np.hamming(n_fft)
        elif window == 'blackman':
            win = np.blackman(n_fft)
        else:
            win = np.ones(n_fft)
        
        f, t, Zxx = signal.stft(self.y, 
                                self.sr,
                                nperseg=n_fft, 
                                noverlap=n_fft - hop_length,
                                window=win,
                                return_onesided=True,
                                boundary=None,
                                padded=False)
        
        self.magnitude = np.abs(Zxx)
        self.log_magnitude = 20.0 * np.log10(np.maximum(1e-10, self.magnitude))
        self.freqs = f
        self.times = t
        self.hop_length = hop_length
        
        return self.magnitude, self.log_magnitude
    
    def analyze_psd(self, nperseg=None, noverlap=None, window='hann'):
        """Compute PSD"""
        from scipy import signal
        
        if nperseg is None:
            nperseg = min(2048, len(self.y) // 8)
        if noverlap is None:
            noverlap = nperseg // 2
        
        freqs, psd = signal.welch(self.y, fs=self.sr, 
                                 nperseg=nperseg, 
                                 noverlap=noverlap,
                                 window=window, 
                                 scaling='density')
        
        psd_db = 10 * np.log10(psd + 1e-12)
        
        return freqs, psd, psd_db
    
    def detect_fundamental_and_harmonics_from_psd(self, freqs, psd_db, 
                                                 fmin=100, fmax=None,
                                                 prominence_threshold=5,
                                                 max_harmonics=10,
                                                 harmonic_tolerance=0.05):
        """Detect fundamental frequency and harmonics from PSD"""
        if fmax is None:
            fmax = self.sr / 2
        
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs_subset = freqs[freq_mask]
        psd_subset = psd_db[freq_mask]
        
        # Find peaks in PSD
        peaks, properties = find_peaks(psd_subset, 
                                      prominence=prominence_threshold,
                                      distance=max(5, int(50 / (freqs[1] - freqs[0]))))
        
        if len(peaks) == 0:
            return None, []
        
        peak_freqs = freqs_subset[peaks]
        peak_mags = psd_subset[peaks]
        
        
        print(f"\nDEBUG - PSD Peak Detection:")
        print(f"  Found {len(peaks)} peaks in PSD")
        print(f"  Top 10 peaks by magnitude:")
        sorted_indices = np.argsort(peak_mags)[::-1]
        for i in sorted_indices[:10]:
            print(f"    {peak_freqs[i]:.1f} Hz: {peak_mags[i]:.1f} dB")


        # Sort by magnitude
        sorted_indices = np.argsort(peak_mags)[::-1]
        peak_freqs_sorted = peak_freqs[sorted_indices]
        
        
        # Find fundamental - prioritize lowest frequency with strong harmonic series
        best_fundamental = None
        best_harmonic_series = []
        best_score = 0
        
        # Sort candidates by frequency (low to high), not magnitude
        peak_freqs_by_freq = sorted(zip(peak_freqs, peak_mags), key=lambda x: x[0])
        
        print(f"\nDEBUG - Testing F0 candidates (low to high freq):")
        
        for f0_candidate, mag in peak_freqs_by_freq[:min(10, len(peak_freqs_by_freq))]:
            if f0_candidate < fmin:
                print(f"\n  Testing F0={f0_candidate:.1f} Hz")
                continue
            
            harmonic_series = [f0_candidate]
            tolerance = f0_candidate * harmonic_tolerance
            
            for harmonic_num in range(2, max_harmonics + 1):
                expected_freq = f0_candidate * harmonic_num
                if expected_freq > fmax:
                    break
                
                freq_diffs = np.abs(peak_freqs - expected_freq)
                closest_idx = np.argmin(freq_diffs)
                
                if freq_diffs[closest_idx] < tolerance:
                    harmonic_series.append(peak_freqs[closest_idx])
            
            print(f"    Found {len(harmonic_series)} harmonics: {[f'{f:.1f}' for f in harmonic_series[:5]]}")

            # Require at least 2 harmonics (F0 + one harmonic)
            score = len(harmonic_series)
            if score > best_score:
                best_score = score
                best_fundamental = f0_candidate
                best_harmonic_series = harmonic_series
        
        print(f"\nDEBUG - Selected F0={best_fundamental:.1f} Hz with {len(best_harmonic_series)} harmonics")
        print(f"  Harmonic series: {[f'{f:.1f}' for f in best_harmonic_series]}")
        
        return best_fundamental, best_harmonic_series

    def track_harmonics_with_template(self, harmonic_series, 
                                    fmin=None, fmax=None,
                                    freq_tolerance=0.08,
                                    prominence_factor=0.05, 
                                    curve_smoothing_window=7):
        """Track harmonics across time using PSD-derived template"""

        print(f"DEBUG: track_harmonics_with_template received: {harmonic_series}")

        if fmin is None:
            fmin = min(harmonic_series) * 0.8
        if fmax is None:
            fmax = max(harmonic_series) * 1.2
        
        harmonic_tracks = []
        
        for t_idx in range(self.magnitude.shape[1]):
            spectrum = self.log_magnitude[:, t_idx]
            frame_harmonics = {
                'time': self.times[t_idx],
                'harmonics': []
            }
            
            for i, expected_freq in enumerate(harmonic_series):
                if expected_freq < fmin or expected_freq > fmax:
                    continue
                
                tolerance = expected_freq * freq_tolerance
                freq_range = (expected_freq - tolerance, expected_freq + tolerance)
                
                freq_mask = (self.freqs >= freq_range[0]) & (self.freqs <= freq_range[1])
                
                if not np.any(freq_mask):
                    continue
                
                freq_indices = np.where(freq_mask)[0]
                spectrum_slice = spectrum[freq_mask]
                
                # Find peaks in this frequency range
                spectrum_range = np.max(spectrum_slice) - np.min(spectrum_slice)
                if spectrum_range > 0:
                    prominence_threshold = spectrum_range * prominence_factor
                    
                    peaks, _ = find_peaks(spectrum_slice, prominence=prominence_threshold)
                    
                    if len(peaks) > 0:
                        strongest_peak_idx = peaks[np.argmax(spectrum_slice[peaks])]
                        actual_freq_idx = freq_indices[strongest_peak_idx]
                        
                        harmonic_data = {
                            'harmonic_number': i + 1,
                            'expected_frequency': expected_freq,
                            'actual_frequency': self.freqs[actual_freq_idx],
                            'magnitude': spectrum[actual_freq_idx],
                            'freq_idx': actual_freq_idx
                        }
                        frame_harmonics['harmonics'].append(harmonic_data)
            
            harmonic_tracks.append(frame_harmonics)
                    
        # Smooth harmonic trajectories
        if curve_smoothing_window >= 3:
            for harm_num in range(1, len(harmonic_series) + 1):
                harm_times = []
                harm_freqs = []
                harm_indices = []
                
                for t_idx, frame_data in enumerate(harmonic_tracks):
                    for harmonic in frame_data['harmonics']:
                        if harmonic['harmonic_number'] == harm_num:
                            harm_times.append(frame_data['time'])
                            harm_freqs.append(harmonic['actual_frequency'])
                            harm_indices.append(t_idx)
                            break
                
                if len(harm_freqs) >= curve_smoothing_window:
                    window_len = min(curve_smoothing_window, len(harm_freqs))
                    if window_len % 2 == 0:
                        window_len -= 1
                    if window_len >= 3:
                        harm_freqs_smooth = savgol_filter(harm_freqs, window_length=window_len, polyorder=2)
                        
                        # Update actual frequencies
                        for i, t_idx in enumerate(harm_indices):
                            for harmonic in harmonic_tracks[t_idx]['harmonics']:
                                if harmonic['harmonic_number'] == harm_num:
                                    harmonic['actual_frequency'] = harm_freqs_smooth[i]
                                    break
        
        return harmonic_tracks
    

    def find_valleys_between_harmonics(self, harmonic_tracks, valley_margin=0.25,min_gap=50):
        """Find valleys between consecutive harmonics"""
        valley_tracks = {}
        
        for t_idx, frame_data in enumerate(harmonic_tracks):
            harmonics = frame_data['harmonics']
            
            if len(harmonics) < 2:
                continue
            
            harmonics_sorted = sorted(harmonics, key=lambda x: x['actual_frequency'])
            
            for i in range(len(harmonics_sorted) - 1):
                h1 = harmonics_sorted[i]
                h2 = harmonics_sorted[i + 1]
                
                h1_freq = h1['actual_frequency']
                h2_freq = h2['actual_frequency']
                
                if h2_freq - h1_freq < min_gap:
                    continue
                
                freq_gap = h2_freq - h1_freq
                margin = freq_gap * valley_margin
                
                search_start_freq = h1_freq + margin
                search_end_freq = h2_freq - margin
                
                if search_start_freq >= search_end_freq:
                    continue
                
                search_start_idx = np.searchsorted(self.freqs, search_start_freq)
                search_end_idx = np.searchsorted(self.freqs, search_end_freq)
                
                valley_spectrum = self.log_magnitude[search_start_idx:search_end_idx, t_idx]
                valley_local_idx = np.argmin(valley_spectrum)
                valley_freq_idx = search_start_idx + valley_local_idx
                
                pair_key = f"H{h1['harmonic_number']}-H{h2['harmonic_number']}"
                
                if pair_key not in valley_tracks:
                    valley_tracks[pair_key] = []
                
                valley_data = {
                    'time': frame_data['time'],
                    'frequency': self.freqs[valley_freq_idx],
                    'magnitude': valley_spectrum[valley_local_idx],
                    'freq_idx': valley_freq_idx,
                    'between_harmonics': (h1_freq, h2_freq),
                    'harmonic_numbers': (h1['harmonic_number'], h2['harmonic_number']),
                    'valley_depth': (h1['magnitude'] + h2['magnitude']) / 2 - valley_spectrum[valley_local_idx]
                }
                
                valley_tracks[pair_key].append(valley_data)
        
        return valley_tracks


    def apply_learned_corrections(self, corrector_model):
        """Apply learned model to refine harmonic tracks
        
        Args:
            corrector_model: HarmonicCorrector instance
        """
        if not corrector_model.is_trained:
            print("Model not trained, skipping corrections")
            return
        
        corrections_applied = 0
        
        for t_idx, frame_data in enumerate(self.harmonic_tracks):
            for harmonic in frame_data['harmonics']:
                old_freq = harmonic['actual_frequency']
                
                # Extract context for this detection
                context_width = 5
                t_start = max(0, t_idx - context_width)
                t_end = min(self.magnitude.shape[1], t_idx + context_width + 1)
                
                freq_idx = harmonic['freq_idx']
                freq_context = 20  # bins
                f_start = max(0, freq_idx - freq_context)
                f_end = min(len(self.freqs), freq_idx + freq_context + 1)
                
                spec_slice = self.log_magnitude[f_start:f_end, t_start:t_end]
                freqs_slice = self.freqs[f_start:f_end]
                
                # Predict correction
                new_freq = corrector_model.predict_correction(spec_slice, freqs_slice, old_freq)
                
                # Only apply if shift is significant (>5 Hz)
                if abs(new_freq - old_freq) > 5:
                    harmonic['actual_frequency'] = new_freq
                    harmonic['model_corrected'] = True
                    corrections_applied += 1
        
        print(f"✓ Applied {corrections_applied} learned corrections")