"""
Contour extraction pipeline for bioacoustic vocalizations.

Uses ridge detection (Hessian/Frangi filters) with impulse denoising,
frequency-adaptive rolling ball background subtraction, and CLAHE enhancement.
"""

from .algorithms import (
    detect_vertical_impulses,
    group_impulse_events,
    remove_impulse_columns,
    deimpulse_spectrogram,
)
from .processing import (
    process_audio_file,
    process_directory,
)
from .plotting import (
    plot_multi_comparison,
    plot_results,
    print_summary,
)
from .config import (
    SpectrogramConfig,
    DenoiseConfig,
    RollingBallConfig,
    CLAHEConfig,
    RidgeDetectionConfig,
    ComponentSelectionConfig,
    ProcessingConfig,
    DEFAULT_CONFIG,
    NO_DENOISE_CONFIG,
    DTRAIN_CONFIG,
    IMPULSE_CONFIG,
    create_config_variant,
)
from .experiments import (
    run_single_file_comparison,
    run_directory_comparison,
    run_dtrain_test,
    run_impulse_test,
)
from .optimize import (
    optimize_parameters,
    batch_optimize,
    evaluate_contour_quality,
    compute_spectrogram_for_eval,
)

__version__ = "0.1.0"

__all__ = [
    # Algorithms
    'detect_vertical_impulses',
    'group_impulse_events',
    'remove_impulse_columns',
    'deimpulse_spectrogram',
    # Processing
    'process_audio_file',
    'process_directory',
    # Plotting
    'plot_multi_comparison',
    'plot_results',
    'print_summary',
    # Config classes
    'SpectrogramConfig',
    'DenoiseConfig',
    'RollingBallConfig',
    'CLAHEConfig',
    'RidgeDetectionConfig',
    'ComponentSelectionConfig',
    'ProcessingConfig',
    # Predefined configs
    'DEFAULT_CONFIG',
    'NO_DENOISE_CONFIG',
    'DTRAIN_CONFIG',
    'IMPULSE_CONFIG',
    'create_config_variant',
    # Experiments
    'run_single_file_comparison',
    'run_directory_comparison',
    'run_dtrain_test',
    'run_impulse_test',
    # Optimization  # ADD THIS
    'optimize_parameters',
    'batch_optimize',
    'evaluate_contour_quality',
    'compute_spectrogram_for_eval',
]