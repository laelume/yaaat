"""
main.py

YAAATApp — primary application entry point and tab registration.

Builds a 4-tab Tkinter Notebook interface. Each tab is a separate
annotator instance sharing the same root window.

Tab registration:
    Core tabs (always loaded):
        BaseAnnotator      — spectrogram viewer, no annotation
        ChangepointAnnotator — contour/changepoint annotation
        PeakAnnotator      — dual-resolution peak annotation
        HarmonicAnnotator  — F0/harmonic detection and correction
        BatchAnnotator    — grid-based batch dataset labeling

    Optional tabs (loaded via config or CLI flag — not yet implemented):
        Future tabs registered here when added to yaaat.layers

Tab sync:
    File navigation in any tab triggers _syncing_tabs flag on all other
    tabs to suppress auto-save during programmatic file index updates.
    All tabs share the same audio_files list and current_file_idx after
    initial load — sync is one-way from the navigating tab outward.

    TODO: Implement cross-tab file sync so navigating in one tab
    updates the current file in all other tabs simultaneously.

Window geometry:
    Default: 1400x800. Resizable. Minimum size enforced at 900x600.
"""

import logging
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)


# (つ -' _ '- )つ    (つ -' _ '- )つ
# TAB IMPORTS
# All core tabs imported here. Optional/experimental tabs imported
# conditionally when that loading mechanism is implemented.
# (つ -' _ '- )つ    (つ -' _ '- )つ

from yaaat.tabs.base_annotator        import BaseAnnotator
from yaaat.tabs.changepoint_annotator import ChangepointAnnotator
from yaaat.tabs.peak_annotator        import PeakAnnotator
from yaaat.tabs.harmonic_annotator    import HarmonicAnnotator
from yaaat.tabs.batch_annotator      import BatchAnnotator


##    <(''<)  <( ' ' )>  (>'')>

class YAAATApp:
    """Primary YAAAT application — 4-tab annotator interface.

    Builds a ttk.Notebook with one tab per annotator class.
    Each tab is instantiated as a ttk.Frame child of the notebook.
    """

    # (つ -' _ '- )つ    (つ -' _ '- )つ
    # CORE TAB REGISTRY
    # List of (label, class) pairs in display order.
    # To add a tab: append (label, AnnotatorClass) here.
    # To remove a tab: comment out the entry.
    # (つ -' _ '- )つ    (つ -' _ '- )つ

    CORE_TABS = [
        ("Base",         BaseAnnotator),
        ("Changepoint",  ChangepointAnnotator),
        ("Peak",         PeakAnnotator),
        ("Harmonic",     HarmonicAnnotator),
        ("Batch",       BatchAnnotator),
    ]

    def __init__(self, root):
        """Build the notebook interface and instantiate all core tabs.

        Args:
            root: tk.Tk — the root window
        """
        self.root = root
        self.root.title("YAAAT — Yet Another Audio Annotation Tool")
        self.root.geometry("1400x800")
        self.root.minsize(900, 600)

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # NOTEBOOK
        # Each tab is a ttk.Frame passed as root to the annotator class.
        # The annotator packs its UI into that frame via setup_ui().
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Instantiated tab objects — kept for cross-tab sync access
        self.tabs = []

        for label, AnnotatorClass in self.CORE_TABS:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=label)

            try:
                tab = AnnotatorClass(frame)
                self.tabs.append(tab)
                logger.info("Registered tab: %s", label)
            except Exception as e:
                logger.error("Failed to load tab '%s': %s", label, e)
                # (つ -' _ '- )つ    (つ -' _ '- )つ
                # Show error placeholder in failed tab frame
                # so the notebook remains functional for other tabs.
                # (つ -' _ '- )つ    (つ -' _ '- )つ
                ttk.Label(
                    frame,
                    text=f"Failed to load tab '{label}':\n{e}",
                    foreground='red', font=('', 10)
                ).pack(expand=True)
                self.tabs.append(None)

        # Bind tab change event for future cross-tab sync
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, event):
        """Handle tab switch event.

        Currently a stub — cross-tab file sync is deferred.
        When implemented, this method will:
            1. Get the current file_idx from the previously active tab
            2. Set _skip_reload=True on all other tabs
            3. Set current_file_idx on all other tabs
            4. Call load_current_file() on the newly active tab
            5. Clear _skip_reload on all tabs

        TODO: Implement cross-tab file sync.
        """
        pass


##    <(''<)  <( ' ' )>  (>'')>

def main():
    """Launch the full YAAAT 4-tab interface."""
    root = tk.Tk()
    app  = YAAATApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

# U S A G I
# from yaaat.main import main
# main()