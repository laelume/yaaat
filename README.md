# YAAAT! Yet Another Audio Annotation Tool

Interactive bioacoustic annotation tool for measuring vocalizations. 

Features: 
1. Changepoint Annotator, for marking temporal onset, offset, and changepoints in vocalizations. Useful for describing rapid fluctuations and identifying nonlinear phenomena. 
2. Peak Annotator, for marking dominant frequency peaks on the power spectrum. Useful for describing spectrally complex vocalizations. 
3. Harmonic Annotator, for identifying harmonics. 

If YAAAT is useful in your work, please ⭐ the repository or open an issue with feedback !!!

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

### Via Pip (probably easiest)
```bash
pip install yaaat
```

### From Source
```bash
git clone https://github.com/laelume/yaaat.git
cd yaaat
pip install -e .
```

## Usage

### Launch the Application
```bash
yaaat
```
Opens a tabbed interface with various annotators. Includes test audio files to get started. For some reason, auto-load is a little buggy, so clicking **Load Audio Files** and selecting the included test_audio yourself lets the interface work as-intended. 

### Use Individual Annotators
```bash
python -m yaaat.changepoint_annotator
python -m yaaat.peak_annotator
python -m yaaat.harmonic_annotator
```

### Use in Python Scripts
```python
from yaaat import ChangepointAnnotator, PeakAnnotator, HarmonicAnnotator
import tkinter as tk

# Launch annotator
root = tk.Tk()
app = ChangepointAnnotator(root) # or PeakAnnotator or HarmonicAnnotator
root.mainloop()
```

## Getting Started (Any Tab)

1. Click **Load Audio Directory** to select files or **Load Test Audio** to explore test audio
2. Choose where to save annotations (existing, new, or default directory)
3. Click on the spectrogram to add annotations points 
4. Click **Finish Syllable** when done with annotation
5. Move between files using **Next/Previous** buttons
6. Annotations auto-save on file navigation or **Finish syllable**

## Navigation & Features

- Intuitive real-time interactive visualization with zoom, pan, and keycommand + mousewheel navigation
- Visualize harmonics with adjustable multipliers and draggable bounding boxes
- JSON annotations saved per-file to minimize corruption
- Mark and track unusable files
- Adjust spectrogram resolution for accuracy comparison

- TODO: implement ranking system for annotation quality; inject as learning feedback mechanism


## v0.1.9 - Contour Extraction Tools

### New Features
- **Lasso Selection**: Ctrl+Click+Drag to select regions
- **Endpoint Marking**: Ctrl+Click two points to extract time range
- **Cross-contour extraction**: Select points from multiple contours at once
- **Auto-sorted table**: Contours sorted by onset time

### Improvements
- Zoom only activates on normal Click+Drag (Ctrl excluded)
- Table updates immediately after modifications
- Better error handling and user feedback



## Requirements

- Python ≥3.8 (built using 3.11)
- numpy
- matplotlib
- scipy
- natsort
- pysoniq # custom minimal audio package

## License

MIT License - Copyright (c) 2025 laelume

## Contributing

Contributions welcome! Please open an issue or submit a pull request. I'm especially interested in talking to people about using this in their existing AI workflows, so please feel free to reach out !!
