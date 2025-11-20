# YAAAT: Yet Another Audio Annotation Tool

Lightweight interactive bioacoustics annotation tool for marking onset, offset, and changepoints in vocalizations.

![YAAAT Screenshot](changepoint_annotator_demo.jpg)

## Getting Started

1. Click **Load Audio Directory** to select your audio files
2. Choose where to save annotations (existing directory, new directory, or default)
3. Click on the spectrogram to add annotation points
4. Click **Finish Syllable** when done with each syllable
5. Move between files using **Next/Previous** buttons
6. Annotations auto-save on file navigation or 'Finish syllable'

## Navigation & Features

- Intuitive real-time interactive visualization with zoom, pan, and keycommand + mousewheel navigation
- Visualize harmonics with adjustable multipliers and draggable bounding boxes
- JSON annotations saved per-file to minimize corruption
- Mark and track unusable files
- Adjust spectrogram resolution for accuracy comparison
- TODO: implement ranking system for annotation quality; add PSD views; inject as learning feedback mechanism

## Installation From Command Line
```bash
git clone https://github.com/laelume/yaaat.git
cd yaaat
pip install -r requirements.txt
cd yaaat
python changepoint_annotator.py
```

## Installation As Package
```bash
pip install yaaat
```

## Usage

### Run As Standalone Application
```bash
download .exe file
```

### Start From Command Line
```bash
yaaat
```

### Use in Python, Jupyter, &.c
```python
from yaaat import ChangepointAnnotator
import tkinter as tk

root = tk.Tk()
app = ChangepointAnnotator(root)
root.mainloop()
```

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
