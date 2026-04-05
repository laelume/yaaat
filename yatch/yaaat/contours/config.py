"""Configuration dataclasses and default parameter sets."""

from dataclasses import dataclass, asdict
from typing import Tuple, Optional


@dataclass
class SpectrogramConfig:
    """Spectrogram generation parameters."""
    fmin: float = 1000
    fmax: float = 8000
    n_fft: int = 512
    hop_length: int = 128


@dataclass
class DenoiseConfig:
    """Impulse denoising parameters."""
    denoise_impulses: bool = True
    impulse_freq_threshold: float = 0.4
    impulse_intensity_threshold_db: float = 10.0
    impulse_max_gap: int = 3
    impulse_expand: int = 2


@dataclass
class RollingBallConfig:
    """Frequency-adaptive rolling ball background subtraction parameters."""
    low_freq_boundary_hz: float = 1000
    mid_freq_boundary_hz: float = 3000
    kernel_low_freq: Tuple[int, int] = (7, 20)
    kernel_low_flatness: float = 0.1
    kernel_mid_freq: Tuple[int, int] = (5, 20)
    kernel_mid_flatness: float = 0.1
    kernel_high_freq: Tuple[int, int] = (3, 20)
    kernel_high_flatness: float = 0.1


@dataclass
class CLAHEConfig:
    """CLAHE enhancement parameters."""
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)


@dataclass
class RidgeDetectionConfig:
    """Ridge detection and fallback parameters."""
    ridge_filter: str = 'hessian'  # 'hessian', 'frangi', 'threshold', 'edge', 'adaptive'
    ridge_threshold: float = 0.3
    hessian_sigmas: Tuple[int, int] = (1, 5)
    frangi_sigmas: Tuple[int, int] = (1, 5)
    frangi_scale_step: float = 0.5
    use_frangi_fallback: bool = False
    fallback_edge_threshold: int = 100


@dataclass
class ComponentSelectionConfig:
    """Component selection scoring parameters."""
    time_dist_divisor: float = 50.0
    freq_dist_divisor: float = 100.0
    size_score_divisor: float = 5.0
    energy_score_divisor: float = 10.0
    time_weight: float = 5.0
    freq_weight: float = 1.0
    size_weight: float = 1.0
    energy_weight: float = 1.0


@dataclass
class ProcessingConfig:
    """Complete processing configuration combining all parameter groups."""
    spectrogram: SpectrogramConfig
    denoise: DenoiseConfig
    rolling_ball: RollingBallConfig
    clahe: CLAHEConfig
    ridge: RidgeDetectionConfig
    component: ComponentSelectionConfig
    
    def to_dict(self):
        """Convert all nested configs to flat dict for process_audio_file."""
        return {
            **asdict(self.spectrogram),
            **asdict(self.denoise),
            **asdict(self.rolling_ball),
            **asdict(self.clahe),
            **asdict(self.ridge),
            **asdict(self.component),
        }


# Default configurations for kiwi vocalizations
DEFAULT_CONFIG = ProcessingConfig(
    spectrogram=SpectrogramConfig(
        fmin=1000,
        fmax=8000,
        n_fft=512,
        hop_length=128
    ),
    denoise=DenoiseConfig(
        denoise_impulses=True,
        impulse_freq_threshold=0.4,
        impulse_intensity_threshold_db=10.0,
        impulse_max_gap=3,
        impulse_expand=2
    ),
    rolling_ball=RollingBallConfig(
        low_freq_boundary_hz=1000,
        mid_freq_boundary_hz=3000,
        kernel_low_freq=(7, 20),
        kernel_low_flatness=0.1,
        kernel_mid_freq=(5, 20),
        kernel_mid_flatness=0.1,
        kernel_high_freq=(3, 20),
        kernel_high_flatness=0.1
    ),
    clahe=CLAHEConfig(
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8)
    ),
    ridge=RidgeDetectionConfig(
        ridge_filter='hessian',
        ridge_threshold=0.3,
        hessian_sigmas=(1, 5),
        frangi_sigmas=(1, 5),
        frangi_scale_step=0.5,
        use_frangi_fallback=False,
        fallback_edge_threshold=100
    ),
    component=ComponentSelectionConfig(
        time_dist_divisor=50.0,
        freq_dist_divisor=100.0,
        size_score_divisor=5.0,
        energy_score_divisor=10.0,
        time_weight=5.0,
        freq_weight=1.0,
        size_weight=1.0,
        energy_weight=1.0
    )
)

# Configuration without denoising (for comparison)
NO_DENOISE_CONFIG = ProcessingConfig(
    spectrogram=SpectrogramConfig(fmin=1000, fmax=8000, n_fft=512, hop_length=128),
    denoise=DenoiseConfig(denoise_impulses=False),
    rolling_ball=RollingBallConfig(),
    clahe=CLAHEConfig(),
    ridge=RidgeDetectionConfig(ridge_filter='hessian', ridge_threshold=0.3),
    component=ComponentSelectionConfig()
)

# Configuration for d-train vocalizations (sharp transitions)
DTRAIN_CONFIG = ProcessingConfig(
    spectrogram=SpectrogramConfig(fmin=1000, fmax=8000, n_fft=512, hop_length=128),
    denoise=DenoiseConfig(
        denoise_impulses=True,
        impulse_freq_threshold=0.4,
        impulse_intensity_threshold_db=10.0
    ),
    rolling_ball=RollingBallConfig(),
    clahe=CLAHEConfig(),
    ridge=RidgeDetectionConfig(ridge_filter='hessian', ridge_threshold=0.3),
    component=ComponentSelectionConfig(
        time_dist_divisor=20.0,  # More aggressive time bridging
        freq_dist_divisor=100.0,
        time_weight=10.0,  # Prioritize temporal proximity
        freq_weight=1.0,
        size_weight=1.0,
        energy_weight=1.0
    )
)

# Configuration for impulse-contaminated recordings
IMPULSE_CONFIG = ProcessingConfig(
    spectrogram=SpectrogramConfig(fmin=1000, fmax=8000, n_fft=512, hop_length=128),
    denoise=DenoiseConfig(
        denoise_impulses=True,
        impulse_freq_threshold=0.3,  # Lower threshold for sensitive detection
        impulse_intensity_threshold_db=10.0
    ),
    rolling_ball=RollingBallConfig(),
    clahe=CLAHEConfig(),
    ridge=RidgeDetectionConfig(ridge_filter='hessian', ridge_threshold=0.3),
    component=ComponentSelectionConfig()
)


def create_config_variant(base_config: ProcessingConfig, **overrides) -> ProcessingConfig:
    """
    Create configuration variant by overriding specific parameters.
    
    Parameters
    ----------
    base_config : ProcessingConfig
        Base configuration to start from
    **overrides : dict
        Parameter overrides in format: section__param=value
        Example: ridge__ridge_threshold=0.2, denoise__impulse_freq_threshold=0.5
    
    Returns
    -------
    ProcessingConfig
        New configuration with overrides applied
    """
    config_dict = base_config.to_dict()
    
    for key, value in overrides.items():
        if key in config_dict:
            config_dict[key] = value
    
    return ProcessingConfig(
        spectrogram=SpectrogramConfig(**{k: v for k, v in config_dict.items() 
                                         if k in SpectrogramConfig.__annotations__}),
        denoise=DenoiseConfig(**{k: v for k, v in config_dict.items() 
                                 if k in DenoiseConfig.__annotations__}),
        rolling_ball=RollingBallConfig(**{k: v for k, v in config_dict.items() 
                                          if k in RollingBallConfig.__annotations__}),
        clahe=CLAHEConfig(**{k: v for k, v in config_dict.items() 
                             if k in CLAHEConfig.__annotations__}),
        ridge=RidgeDetectionConfig(**{k: v for k, v in config_dict.items() 
                                      if k in RidgeDetectionConfig.__annotations__}),
        component=ComponentSelectionConfig(**{k: v for k, v in config_dict.items() 
                                              if k in ComponentSelectionConfig.__annotations__})
    )