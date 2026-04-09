# YAAAT! Yet Another Audio Annotation Tool

Interactive bioacoustic annotation tool for measuring animal vocalizations. 

Features: 
1. Changepoint Annotator, for marking temporal onset, offset, and changepoints in vocalizations. Useful for describing rapid fluctuations and identifying nonlinear phenomena. 
2. Peak Annotator, for marking dominant frequency peaks on the power spectrum. Useful for describing spectrally complex vocalizations. 
3. Harmonic Annotator, for identifying harmonics. 

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/laelume/yaaat/main/yaaat/images/changepoint_annotator_screenshot.jpg" alt="Changepoint Annotator" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/laelume/yaaat/main/yaaat/images/peak_annotator_screenshot.jpg" alt="Peak Annotator" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/laelume/yaaat/main/yaaat/images/harmonic_annotator_screenshot.jpg" alt="Harmonic Annotator" width="400"/></td>
  </tr>
  <tr>
    <td align="center">Changepoint Annotator</td>
    <td align="center">Peak Annotator</td>
    <td align="center">Harmonic Annotator</td>
  </tr>
</table>

## Installation

```bash
pip install yaaat
```

## WIP: 
Optional detection/classification backend:

```bash
pip install yaaat[ai]
```

---

## Requirements

- Python 3.9+
- numpy, scipy, matplotlib, natsort, pandas
- pysoniq (custom minimal audio I/O)
- ffmpeg (adds MP3 support via pysoniq, optional)

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

#### Base Annotator
Spectrogram and waveform viewer. No annotation logic. Shared point
annotation API available for programmatic use.

#### Changepoint Annotator
Click-to-add contour points for syllable segmentation. Supports lasso
selection, Ctrl+Click dual-endpoint marking, harmonic bounding boxes,
skip file dialog, and find-next-skipped navigation.

#### Peak Annotator
Dual-resolution display — vertical spectrogram (temporal resolution)
with PSD overlay (frequency resolution). Click-to-mark spectral peaks.
Auto-detection with prominence threshold.

#### Harmonic Annotator
Auto-detects F0 from mean spectrum. Builds harmonic series with
draggable correction lines. Computes inter-harmonic valley boundaries
(min energy search) for spectral band visualization. Multiple ridge
detection methods: max, peaks, centroid, parabolic, peak ratio.

#### Binary Annotator
Paginated grid view of mel spectrograms across a dataset. Define named
binary annotation columns (e.g. has_noise, has_bifurcation). Batch
annotate selected files True/False. Export to CSV.

## License

MIT