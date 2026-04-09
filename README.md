# YAAAT — Yet Another Audio Annotation Tool

Multi-tab Tkinter GUI for spectrogram-based annotation of animal vocalizations.

---

## Installation

```bash
pip install yaaat
```

WIP: optional detection backend:

```bash
pip install yaaat[ai]
```

---

## Requirements

- Python 3.9+
- numpy, scipy, matplotlib, natsort, pandas
- pysoniq (audio I/O)
- ffmpeg (MP3 support via pysoniq, optional)

---

## Usage

Launch the full 4-tab interface:

```bash
python -m yaaat
```

Or from Python:

```python
from yaaat import main
main()
```

---

## Tabs

**Base**
Spectrogram and waveform viewer. No annotation logic. Shared point
annotation API available for programmatic use.

**Changepoint**
Click-to-add contour points for syllable segmentation. Supports lasso
selection, Ctrl+Click dual-endpoint marking, harmonic bounding boxes,
skip file dialog, and find-next-skipped navigation.

**Peak**
Dual-resolution display — vertical spectrogram (temporal resolution)
with PSD overlay (frequency resolution). Click-to-mark spectral peaks.
Auto-detection with prominence threshold.

**Harmonic**
Auto-detects F0 from mean spectrum. Builds harmonic series with
draggable correction lines. Computes inter-harmonic valley boundaries
(min energy search) for spectral band visualization. Multiple ridge
detection methods: max, peaks, centroid, parabolic, peak ratio.

**Binary**
Paginated grid view of mel spectrograms across a dataset. Define named
binary annotation columns (e.g. has_noise, has_bifurcation). Batch
annotate selected files True/False. Export to CSV.

## License

MIT