# YAAAT User Guide

Comprehensive documentation for Yet Another Audio Annotation Tool.

---

## Table of Contents
1. [Installation](#installation)
2. [Launching YAAAT](#launching-yaaat)
3. [Interface Overview](#interface-overview)
4. [File Management](#file-management)
5. [Changepoint Annotator](#changepoint-annotator)
6. [Peak Annotator](#peak-annotator)
7. [Harmonic Annotator](#harmonic-annotator)
8. [Spectrogram Controls](#spectrogram-controls)
9. [Keyboard Shortcuts](#keyboard-shortcuts)
10. [Annotation File Format](#annotation-file-format)
11. [Troubleshooting](#troubleshooting)

---

## Installation

### Via Pip (Recommended)
```bash
pip install yaaat
```

### From Source
```bash
git clone https://github.com/laelume/yaaat.git
cd yaaat
pip install -e .
```

### Requirements
- Python ≥3.8 (developed on 3.11)
- numpy, matplotlib, scipy, natsort, pysoniq

---

## Launching YAAAT

### Main Tabbed Interface
```bash
yaaat
```
Opens all three annotators in tabs.

### Individual Annotators
```bash
python -m yaaat.changepoint_annotator
python -m yaaat.peak_annotator
python -m yaaat.harmonic_annotator
```

### Python API
```python
from yaaat import ChangepointAnnotator, PeakAnnotator, HarmonicAnnotator
import tkinter as tk

root = tk.Tk()
app = ChangepointAnnotator(root)
root.mainloop()
```

---

## Interface Overview

All annotators share common interface elements:

### Top Control Panel
- **Load Audio Directory**: Select folder containing WAV files
- **Load Test Audio**: Load bundled example files
- **Save Directory Selection**: Choose where annotations are saved
- **File Navigation**: Previous/Next buttons with file counter
- **Mark Unusable**: Flag problematic files

### Spectrogram Display
- Center canvas shows time-frequency representation
- X-axis: Time (seconds)
- Y-axis: Frequency (Hz)
- Color intensity: Power (dB)

### Right Control Panel
- Spectrogram parameters (NFFT, overlap, window)
- Zoom controls
- Play audio buttons
- Annotation management

---

## File Management

### Loading Audio Files

**Method 1: Load Audio Directory**
1. Click "Load Audio Directory"
2. Select folder containing WAV files
3. Files sorted naturally (file1, file2, ..., file10)

**Method 2: Load Test Audio**
- Click "Load Test Audio"
- Loads bundled example files
- Good for learning interface

**Known Issue**: Auto-load occasionally fails. Manually click "Load Audio Directory" and reselect if files don't appear.

### Save Directory Selection

When loading audio, you'll be prompted:

**Option 1: Use Existing Directory**
- Load previously saved annotations
- Continue annotation session

**Option 2: Create New Directory**
- Start fresh annotation set
- Creates `annotations_[timestamp]` folder

**Option 3: Use Default Directory**
- Saves to `annotations` subfolder in audio directory

### File Navigation

- **Previous Button**: Go to previous file (auto-saves current)
- **Next Button**: Go to next file (auto-saves current)
- **File Counter**: Shows `N/Total` position
- **Auto-finish**: Incomplete annotations auto-finish on navigation

### Marking Unusable Files

1. Click "Mark as Unusable" button
2. File flagged in JSON: `"unusable": true`
3. Use for corrupted audio, poor quality, etc.

---

## Changepoint Annotator

For marking temporal features in vocalizations: onsets, offsets, and rapid fluctuations.

### Annotation Modes

**Contour Mode** (default)
- Click points to define contour of a vocalization
- Each contour = one syllable
- Points define time-frequency trajectory

**Sequence Mode**
- View completed contours in table format
- Columns: #, t_onset, t_offset, f_min, f_max

Toggle modes via radio buttons in right panel.

### Basic Workflow (Contour Mode)

1. **Click to add points** on spectrogram
   - Click along frequency contour of vocalization
   - Each click adds point at (time, frequency)
   - Points appear as colored markers

2. **Finish contour**
   - Click "Finish Contour" button
   - Current contour → completed contour
   - Points relabeled: first=onset(green), last=offset(magenta), middle=changepoint(cyan)

3. **Start next contour**
   - After finishing, new clicks start new contour
   - No overlap detection—contours are independent

4. **Auto-save**
   - Saves on file navigation
   - Saves when finishing contour (if changes made)

### Point Labeling Logic

For each completed contour:
- Points sorted by time
- **First point** (earliest time) → onset (green)
- **Last point** (latest time) → offset (magenta)  
- **All middle points** → changepoint (cyan)

### Syllable Definition

Each contour defines ONE syllable via bounding box:
- `t_onset`: Time of first point
- `t_offset`: Time of last point
- `f_min`: Minimum frequency across ALL points
- `f_max`: Maximum frequency across ALL points

### Advanced: Ctrl+Click Segmentation

If you forget to finish contour and click many points consecutively:

1. **First Ctrl+Click**: Mark onset point
   - Ctrl+Click near an existing point in current contour
   - Console prints: "✓ Marked ONSET at point N"

2. **Second Ctrl+Click**: Mark offset point
   - Ctrl+Click near another point
   - Extracts contour from onset→offset
   - Creates completed contour
   - Remaining points stay in current_contour

Example:
```
Current contour has 10 points
Ctrl+Click point 3 → onset
Ctrl+Click point 7 → offset
→ Creates contour with points 3-7 (5 points)
→ Points 1,2,8,9,10 remain in current contour
```

### Removing Points

**Click near existing point** (within threshold)
- Point removed from current contour or completed contours
- Updates display immediately
- If threshold too small, increase in code (see Troubleshooting)

### Clear Functions

**Clear Previous**
- Removes last action
- If current contour has points → removes last point
- If current contour empty → restores last completed contour for editing

**Clear All**
- Removes ALL contours and points for current file
- Cannot undo—use carefully

### Sequence Mode Display

Switch to Sequence Mode to view table:
```
#   t_onset   t_offset   f_min    f_max
1   0.123     0.456      2000     8000
2   0.789     1.234      1500     7500
```

Values calculated from contour bounding boxes.

---

## Peak Annotator

For marking dominant frequency peaks on power spectrum. Useful for harmonically rich or spectrally complex vocalizations.

### Basic Workflow

1. **Click on spectrogram** at frequency peak locations
2. Each click adds marker at (time, frequency)
3. Markers shown as colored points
4. Auto-save on file navigation

### Features

- Mark fundamental frequency (f0) and overtones
- Good for steady vocalizations with clear spectral peaks
- Annotate time-varying peak structures

### Controls

- **Remove point**: Click near existing marker
- **Clear all**: Remove all peaks for current file
- Navigation: Previous/Next (auto-saves)

---

## Harmonic Annotator

For identifying and tracking harmonic series in vocalizations.

### Harmonic Detection

Harmonics = integer multiples of fundamental frequency:
- f1 = f0 (fundamental)
- f2 = 2×f0 (second harmonic)
- f3 = 3×f0 (third harmonic)
- etc.

### Basic Workflow

1. **Click on spectrogram** to mark fundamental (f0)
2. **Adjust harmonic multipliers** in right panel
3. **View predicted harmonics** overlaid on spectrogram
4. **Drag bounding boxes** to adjust time/frequency bounds

### Draggable Bounding Boxes

Each harmonic shown as draggable rectangle:
- Drag to reposition
- Resize to adjust temporal/spectral extent
- Visual feedback for harmonic structure

### Controls

- **Harmonic multiplier sliders**: Adjust which harmonics to display
- **Remove annotation**: Click near existing marker
- **Clear all**: Remove all harmonics for current file

---

## Spectrogram Controls

All annotators share spectrogram computation parameters:

### NFFT (FFT Size)
- **Range**: 256 - 8192 samples
- **Effect**: Frequency resolution
- Higher = better frequency resolution, worse time resolution
- Lower = better time resolution, worse frequency resolution
- **Default**: 512

### Overlap Percentage
- **Range**: 0% - 99%
- **Effect**: Temporal smoothness
- Higher = smoother time representation, more computation
- **Default**: 50%

### Window Function
- **Options**: hann, hamming, blackman, bartlett
- **Effect**: Spectral leakage reduction
- **Default**: hann
- **hann**: Good general purpose, smooth
- **hamming**: Slightly better frequency resolution
- **blackman**: Best sidelobe suppression, widest mainlobe
- **bartlett**: Linear taper, good for transients

### Frequency Limits
- **Min Frequency**: Lower display bound (Hz)
- **Max Frequency**: Upper display bound (Hz)
- Set to focus on frequency band of interest

### Resolution Comparison

Adjust parameters to compare different spectrogram views:
1. Annotate with one setting
2. Change NFFT/overlap
3. Verify annotations still accurate
4. Useful for checking annotation precision

---

## Spectrogram Navigation

### Mouse Controls

**Pan (drag)**
- Click and drag on spectrogram
- Moves viewport
- Does NOT add annotation points (drag distance threshold)

**Zoom**
- Mouse wheel: Zoom in/out at cursor position
- Modifier keys change zoom behavior:
  - **No modifier**: Zoom both axes
  - **Shift + wheel**: Zoom time axis only
  - **Ctrl + wheel**: Zoom frequency axis only

**Click**
- Short click (< drag threshold): Add annotation point
- Works when not dragging

### Zoom Buttons

**Zoom In / Zoom Out**
- Buttons in right panel
- Zoom centered on current view

**Reset View**
- Restore original time/frequency limits
- Shows entire audio file

---

## Audio Playback

### Play Buttons

**Play Full**
- Play entire audio file
- Useful for context

**Play Visible**
- Play only visible time range
- Useful for zoomed regions
- Plays from current view's time limits

### Playback Behavior

- Audio plays in real-time
- Cannot pause (stop by closing window or waiting)
- Uses pysoniq backend for cross-platform compatibility

---

## Keyboard Shortcuts

### General
- **Ctrl + Click**: (Changepoint) Mark onset/offset retroactively
- **Shift + Wheel**: Zoom time axis only
- **Ctrl + Wheel**: Zoom frequency axis only
- **Wheel**: Zoom both axes

### Navigation
- **Left Arrow**: Previous file (may require focus)
- **Right Arrow**: Next file (may require focus)

*Note: Keyboard shortcuts are under development—mouse controls more reliable*

---

## Annotation File Format

Annotations saved as JSON files in annotation directory.

### File Structure

**Filename**: `[audio_filename]_annotations.json`

**Changepoint Annotator**:
```json
{
  "audio_file": "example.wav",
  "sample_rate": 44100,
  "duration": 2.5,
  "contours": [
    [
      {"time": 0.123, "freq": 3500.0},
      {"time": 0.145, "freq": 4200.0},
      {"time": 0.167, "freq": 3800.0}
    ],
    [
      {"time": 0.234, "freq": 2500.0},
      {"time": 0.256, "freq": 2800.0}
    ]
  ],
  "syllables": [],
  "unusable": false,
  "nfft": 512,
  "overlap": 0.5,
  "window": "hann",
  "annotation_version": "1.0"
}
```

**Peak Annotator**:
```json
{
  "audio_file": "example.wav",
  "peaks": [
    {"time": 0.123, "freq": 3500.0},
    {"time": 0.456, "freq": 4200.0}
  ],
  "unusable": false
}
```

**Harmonic Annotator**:
```json
{
  "audio_file": "example.wav",
  "harmonics": [
    {
      "fundamental": {"time": 0.123, "freq": 440.0},
      "multipliers": [1, 2, 3, 4],
      "bounding_boxes": []
    }
  ],
  "unusable": false
}
```

### Backward Compatibility

Changepoint annotator saves both `contours` and `syllables` keys. Older versions used `syllables`; current versions use `contours`.

---

## Troubleshooting

### Audio files don't load
- Click "Load Audio Directory" manually (auto-load buggy)
- Verify files are WAV format
- Check file permissions

### Annotations don't save
- Ensure write permissions in save directory
- Check disk space
- Verify JSON isn't corrupted (validate JSON syntax)

### Spectrogram looks wrong
- Adjust NFFT for frequency resolution
- Adjust overlap for time smoothness
- Try different window functions
- Check frequency limits (min/max Hz)

### Can't click to annotate
- Ensure not dragging (keep mouse still)
- Check if in correct mode (Contour vs Sequence)
- Verify click is within spectrogram bounds

### Clicking near point adds new point instead of removing
**Problem**: Removal threshold too small

**Solution**: In `yaaat/changepoint_annotator.py`, line ~982:

Change:
```python
def remove_nearby_annotation(self, x, y, threshold=0.05):
```

To:
```python
def remove_nearby_annotation(self, x, y, threshold=0.15):
```

Adjust value: 0.10 (moderate), 0.15 (comfortable), 0.20 (large)

### Points appear in wrong location
- Check current zoom level
- Verify spectrogram parameters match intended resolution
- Use Resolution Comparison feature

### Ctrl+Click not working
- Ensure points exist in current_contour first
- Click near (within threshold) of existing point
- Check console output for feedback messages
- Platform-specific: may require focus on window

### Previous/Next navigation issues
- Files must be loaded first
- Check file counter shows correct total
- Annotations auto-save before navigation

---

## Tips & Best Practices

### Annotation Strategy

1. **Start with low resolution** (small NFFT, low overlap)
   - Fast spectrogram computation
   - Good for initial pass

2. **Increase resolution for precision** (large NFFT, high overlap)
   - Fine-tune annotations
   - Verify accuracy

3. **Use sequence mode** to review completed contours
   - Check bounding boxes make sense
   - Verify temporal ordering

### Workflow Efficiency

- **Finish contours regularly**: Don't accumulate many points
- **Use Ctrl+Click**: Retroactively segment forgotten contours
- **Mark unusable early**: Skip bad files immediately
- **Save directory organization**: Use timestamped directories for different annotation sessions

### Quality Control

- **Resolution comparison**: Annotate at multiple NFFT settings
- **Listen while annotating**: Use Play Visible for zoomed regions
- **Check sequence display**: Verify bounding boxes are reasonable
- **Review neighboring files**: Ensure consistency across similar vocalizations

---

### Editing Onset/Offset (Forgot to Click Finish Contour)

If you forget to click "Finish Contour" and continue adding points across multiple syllables, use **Ctrl+Click region extraction** to retroactively segment your points.

#### Problem Scenario
```
You clicked 20 points continuously across 3 separate syllables
Should have been: 6 points → Finish → 7 points → Finish → 7 points
Actually did: 20 points without finishing
```

#### Solution: Ctrl+Click Region Extraction

**Step 1: Mark First Region Corner**
- Hold down **Ctrl key**
- Click at one corner of the region you want to extract (e.g., bottom-left)
- Console prints: `✓ Marked ONSET at t=X.XXXs, f=XXXXHz`
- Keep holding Ctrl!

**Step 2: Mark Second Region Corner**
- Still holding **Ctrl**
- Click at opposite corner of the region (e.g., top-right)
- Defines rectangular bounding box

**What Happens Automatically:**
- All points within the rectangle are extracted
- Extracted points → new finished contour
- Remaining points → stay in current_contour
- Console shows: `✓ Created contour with N points` and `→ M points remaining`

#### Visual Example
```
Current unfinished contour (should be 2 syllables):

Time (s):  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
Freq (Hz): 2k   3k   4k   3k   2k   3k   5k   4k   3k   2k
Points:    •    •    •    •    •    •    •    •    •    •
           └─── Syllable 1 ────┘    └─── Syllable 2 ────┘

Extract Syllable 1:
1. Hold Ctrl, click at (0.1s, 1500Hz) - below/before first syllable
2. Hold Ctrl, click at (0.5s, 5000Hz) - above/after first syllable
   → Extracts points at 0.1, 0.2, 0.3, 0.4, 0.5 (5 points)
   → Remaining: 0.6, 0.7, 0.8, 0.9, 1.0 (5 points)

3. Click "Finish Contour" for remaining points (Syllable 2)
```

#### How Region Selection Works

**Bounding Box Definition:**
- Click order doesn't matter (can click top-right then bottom-left)
- System calculates:
  - Time range: [min(t₁, t₂), max(t₁, t₂)]
  - Frequency range: [min(f₁, f₂), max(f₁, f₂)]
- Extracts all points where: `t_min ≤ time ≤ t_max AND f_min ≤ freq ≤ f_max`

**Requirements:**
- Must have at least **2 points** inside region
- If < 2 points found: operation cancelled, error printed with region bounds

#### Common Use Cases

**1. Split Long Sequence into Multiple Syllables**
```
Have: 20 points across 3 syllables (forgot to finish after each)
Solution:
  - Ctrl+Click around syllable 1 → extracts 6 points
  - Ctrl+Click around syllable 2 → extracts 7 points
  - "Finish Contour" → remaining 7 points = syllable 3
```

**2. Remove Noisy Middle Section**
```
Have: 12 points, middle 4 are noise/artifact
Solution:
  - Ctrl+Click around clean beginning → extract first 4 points
  - Ctrl+Click around clean ending → extract last 4 points
  - "Clear All" → discard remaining noisy points
```

**3. Separate Frequency Bands**
```
Have: Points spanning 1000-8000 Hz, want to separate low/high components
Solution:
  - Ctrl+Click region for 1000-4000 Hz → extracts low-frequency contour
  - Ctrl+Click region for 4000-8000 Hz → extracts high-frequency contour
```

**4. Extract Multiple Short Calls**
```
Have: 30 points across 5 brief calls
Solution:
  - Repeatedly Ctrl+Click around each call's time/frequency region
  - Each extraction creates separate finished contour
```

#### Tips for Effective Region Extraction

1. **Zoom in first**: Easier to see point distribution and define precise regions
2. **Be generous with region size**: Better to include extra space than miss points
3. **Check console output**: Tells you exactly how many points extracted/remaining
4. **Use sequence mode after**: Verify extracted contours have correct bounding boxes
5. **Multiple extractions allowed**: Can Ctrl+Click multiple times to extract different regions
6. **Don't need precision**: Bounding box is forgiving - approximate corners work fine

#### Troubleshooting Region Extraction

**Error: "Only found 0-1 point(s) in region"**
- Problem: Clicks didn't capture enough points
- Solution: Click wider region, or zoom in to see actual point locations
- Tip: Console shows exact region bounds - verify they make sense

**Nothing happens on Ctrl+Click**
- Check: Do you have points in current_contour? (must be unfinished)
- Check: Are you clicking on spectrogram (not axes labels)?
- Check: Console should print "Marked ONSET" after first Ctrl+Click
- Try: Click directly on or very near existing points

**Extracted wrong points**
- Cause: Rectangular region includes all points within time AND frequency bounds
- Solution: Be more precise with corner placement, or remove unwanted points manually after extraction
- Prevention: Zoom in before extraction to see point distribution clearly

**Want to undo extraction**
- Option 1: Click "Clear Previous" to restore last finished contour back to current_contour
- Option 2: Manually remove points using normal click-to-remove, then re-extract
- Note: Multiple undo levels not currently supported

#### Workflow Comparison

**Without Region Extraction (must restart):**
```
1. Click 20 points
2. Realize should have finished after point 6
3. Click "Clear All" → lose all work
4. Start over, being careful to finish after each syllable
```

**With Region Extraction (retroactive fix):**
```
1. Click 20 points
2. Realize should have finished after point 6
3. Ctrl+Click around first 6 points → extract syllable 1
4. Ctrl+Click around next 7 points → extract syllable 2
5. "Finish Contour" → remaining points = syllable 3
6. Done! No work lost.
```

#### Technical Details

**What gets extracted:**
- Only points from `current_contour` (unfinished points)
- Completed contours are not affected
- Points checked with inclusive boundaries: `<=` and `>=`

**What happens to extracted points:**
- Removed from `current_contour`
- Added as new entry to `self.contours` (finished)
- Immediately get onset/offset/changepoint labels based on time sorting

**Cancelling operation:**
- Release Ctrl before second click (if key release handler active)
- Or simply click elsewhere and ignore the marked onset
- Or click "Clear Previous" if extraction already completed