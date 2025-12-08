"""
Minimal noise-robust harmonic correction model
Learns from manual annotations to improve peak detection
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.ndimage import maximum_filter1d
from sklearn.ensemble import RandomForestRegressor

class HarmonicCorrector:
    """Predicts frequency corrections based on spectral features"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        self.is_trained = False
    
    def extract_features(self, spectrogram, freqs, old_freq):
        """Extract features from spectrogram slice
        
        Args:
            spectrogram: (freq, time) magnitude array
            freqs: frequency axis
            old_freq: initially detected frequency
        
        Returns:
            1D feature vector
        """
        # Time-averaged spectrum
        avg_spectrum = np.mean(spectrogram, axis=1)
        
        # Find old_freq index
        old_idx = np.argmin(np.abs(freqs - old_freq))
        
        # Local features around detected peak
        context = 10  # bins
        start = max(0, old_idx - context)
        end = min(len(avg_spectrum), old_idx + context + 1)
        
        local_spectrum = avg_spectrum[start:end]
        
        features = [
            np.max(local_spectrum),  # peak magnitude
            np.mean(local_spectrum),  # mean magnitude
            np.std(local_spectrum),   # spectral spread
            old_freq,                 # initial detection
            np.argmax(local_spectrum) - (old_idx - start),  # offset from peak to detection
        ]
        
        # Add spectral shape features
        if len(local_spectrum) > 5:
            # Local maxima count (noise indicator)
            local_max = maximum_filter1d(local_spectrum, size=3)
            n_peaks = np.sum(local_spectrum == local_max)
            features.append(n_peaks)
            
            # Spectral slope
            freq_bins = np.arange(len(local_spectrum))
            slope = np.polyfit(freq_bins, local_spectrum, 1)[0]
            features.append(slope)
        
        return np.array(features)
    
    def train(self, training_data):
        """Train on correction pairs
        
        Args:
            training_data: list of dicts from prepare_correction_pairs()
        """
        if not training_data:
            print("No training data")
            return
        
        X = []
        y = []
        
        for sample in training_data:
            features = self.extract_features(
                sample['spectrogram'],
                sample['freqs'],
                sample['old_freq']
            )
            
            # Target is frequency shift
            freq_shift = sample['new_freq'] - sample['old_freq']
            
            X.append(features)
            y.append(freq_shift)
        
        X = np.array(X)
        y = np.array(y)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        print(f"✓ Trained on {len(X)} corrections")
        print(f"  Mean absolute correction: {np.mean(np.abs(y)):.1f} Hz")
    
    def predict_correction(self, spectrogram, freqs, old_freq):
        """Predict frequency correction
        
        Returns:
            corrected_freq (float)
        """
        if not self.is_trained:
            return old_freq
        
        features = self.extract_features(spectrogram, freqs, old_freq)
        predicted_shift = self.model.predict(features.reshape(1, -1))[0]
        
        return old_freq + predicted_shift
    
    def save(self, filepath):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Batch training script
def train_from_annotations(annotation_dir):
    """Train model from accumulated annotations"""
    annotation_dir = Path(annotation_dir)
    training_file = annotation_dir / "training_corrections.pkl"
    
    if not training_file.exists():
        print(f"No training data at {training_file}")
        return
    
    with open(training_file, 'rb') as f:
        training_data = pickle.load(f)
    
    print(f"Loaded {len(training_data)} corrections from {len(set(d['audio_file'] for d in training_data))} files")
    
    model = HarmonicCorrector()
    model.train(training_data)
    
    # Save trained model
    model_file = annotation_dir / "harmonic_corrector.pkl"
    model.save(model_file)
    
    print(f"✓ Saved model to {model_file}")
    
    return model


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        train_from_annotations(sys.argv[1])
    else:
        print("Usage: python harmonic_learner.py <annotation_directory>")