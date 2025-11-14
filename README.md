# YAAAT - Yet Another Audio Annotation Tool

Interactive spectrogram annotation tool for marking onset, offset, and changepoints in vocalizations.

![YAAAT Screenshot](changepoint_annotator_demo.jpg)

## Features

- **Syllable-based workflow**: Click to add temporal points with automatic color coding
- **Interactive spectrogram visualization**: Real-time visualization with zoom, pan, and navigation
- **Harmonic projection guides**: Visualize 2nd and 3rd harmonics with adjustable multipliers
- **Multiple bounding box shapes**: Rectangle, ellipse, or polygon overlays
- **Flexible annotation storage**: JSON-based with syllable metrics
- **Skip tracking**: Mark and track unusable files with reasons
- **Spectrogram customization**: Adjust n_fft, hop_length, frequency scales (linear/mel)

## Installation
```bash
pip install yaaat
```

## Usage

### Command Line
```bash
yaaat
```

### Python
```python
from yaaat import ChangepointAnnotator
import tkinter as tk

root = tk.Tk()
app = ChangepointAnnotator(root)
root.mainloop()
```

## Quick Start

1. Click **Load Audio Directory** to select your audio files
2. Choose where to save annotations (existing directory, new directory, or default)
3. Click on the spectrogram to add annotation points
4. Click **Finish Syllable** when done with each syllable
5. Navigate between files using **Next/Previous** buttons
6. Annotations auto-save on file navigation

## Controls

- **Click**: Add annotation point
- **Click near existing point**: Remove point
- **Click + Drag**: Zoom to region
- **Right-click**: Undo zoom
- **Ctrl + Scroll**: Horizontal zoom
- **Ctrl + Shift + Scroll**: Vertical zoom
- **Scroll**: Vertical pan

## Annotation Workflow

YAAAT uses a syllable-based approach:
1. Click to place points for onset, changepoints, and offset
2. Points are automatically labeled and color-coded:
   - Green: Onset (first point)
   - Cyan: Changepoints (middle points)
   - Magenta: Offset (last point)
3. Click "Finish Syllable" to complete and start a new syllable
4. Annotations save automatically when navigating files

## Output Format

Annotations are saved as JSON files with:
- Time-frequency coordinates for each point
- Syllable groupings
- Syllable metrics (duration, frequency range, etc.)
- Spectrogram parameters
- Skip status and reasons

## Requirements

- Python ≥3.8
- numpy
- matplotlib
- librosa
- natsort
- sounddevice

## License

MIT License - Copyright (c) 2025 laelume

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
