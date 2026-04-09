"""
tabs/base_annotator.py

Base viewer tab for YAAAT — the simplest possible annotator tab.
Inherits all functionality from BaseLayer with no additional annotation logic.

Provides:
    - Spectrogram and waveform viewing
    - File navigation
    - Playback controls
    - Zoom/pan interaction
    - Shared point annotation API (add_annotation_point)

This tab serves as:
    1. A standalone spectrogram viewer with no annotation overhead
    2. A reference implementation for subclassing BaseLayer
    3. A fallback tab when other annotators are not needed

No annotation data is saved or loaded — save_custom_data and load_custom_data
are intentionally left as no-ops (inherited from BaseLayer).
"""

import logging
import tkinter as tk
from tkinter import ttk

from yaaat.core.base_layer import BaseLayer

logger = logging.getLogger(__name__)


##    <(''<)  <( ' ' )>  (>'')>

class BaseAnnotator(BaseLayer):
    """Minimal viewer tab — BaseLayer with no annotation logic.

    All annotation hooks (save_custom_data, load_custom_data,
    process_audio, draw_custom_overlays) are inherited as no-ops.
    setup_custom_controls adds only a viewer label.
    """

    def __init__(self, root):
        """Initialize BaseAnnotator."""
        super().__init__(root)

        if isinstance(root, tk.Tk):
            self.root.title("Base Viewer - YAAAT")

    ##    <(''<)  <( ' ' )>  (>'')>
    # CUSTOM CONTROLS
    ##    <(''<)  <( ' ' )>  (>'')>

    def setup_custom_controls(self):
        """Display viewer mode label in the custom controls section."""
        ttk.Label(
            self.control_panel,
            text="Viewer mode — no annotation",
            font=('', 8, 'italic'),
            foreground='gray'
        ).pack(pady=2)

        ttk.Label(
            self.control_panel,
            text="Use shared point API to add\nannotations programmatically.",
            font=('', 7),
            foreground='gray',
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=2)

    ##    <(''<)  <( ' ' )>  (>'')>
    # OVERLAYS
    ##    <(''<)  <( ' ' )>  (>'')>

    def draw_custom_overlays(self):
        """Draw shared point annotations if any exist for the current file."""
        from yaaat.core.visualization import draw_shared_point_annotations
        draw_shared_point_annotations(self)


##    <(''<)  <( ' ' )>  (>'')>

def main():
    """Launch BaseAnnotator as a standalone viewer tab."""
    root = tk.Tk()
    app  = BaseAnnotator(root)
    root.geometry("1400x800")
    root.mainloop()


if __name__ == "__main__":
    main()

# U S A G I
# from yaaat.tabs.base_annotator import BaseAnnotator
# root = tk.Tk(); app = BaseAnnotator(root); root.geometry("1400x800"); root.mainloop()