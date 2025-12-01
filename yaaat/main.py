"""Main launcher for YAAAT with tabbed interface"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path

try:
    # Package mode: python -m yaaat
    from yaaat.changepoint_annotator import ChangepointAnnotator
    from yaaat.peak_annotator import PeakAnnotator
    from yaaat.harmonic_annotator import HarmonicAnnotator
except ImportError:
    # Script mode: python main.py
    from changepoint_annotator import ChangepointAnnotator
    from peak_annotator import PeakAnnotator
    from harmonic_annotator import HarmonicAnnotator


class YAAATApp:
    """Main YAAAT application with tabbed interface"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("YAAAT! Yet Another Audio Annotation Tool")
        
        # Shared state
        self.audio_dir = None
        self.save_dir = None
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for each tool
        changepoint_frame = ttk.Frame(self.notebook)
        peak_frame = ttk.Frame(self.notebook)
        harmonic_frame = ttk.Frame(self.notebook)
        
        # Add tabs
        self.notebook.add(changepoint_frame, text="Changepoint Annotator")
        self.notebook.add(peak_frame, text="Peak Annotator")
        self.notebook.add(harmonic_frame, text="Harmonic Annotator")
        
        # Initialize tools (pass frames as parent)
        self.changepoint_tool = ChangepointAnnotator(changepoint_frame)
        self.peak_tool = PeakAnnotator(peak_frame)
        self.harmonic_tool = HarmonicAnnotator(harmonic_frame)
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
    
    def on_tab_change(self, event):
        """Handle tab switching"""
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 0:
            print("Switched to Changepoint Annotator")
        elif current_tab ==1:
            print("Switched to Peak Annotator")
        else: 
            print("Switched to Harmonic Annotator")


def main():
    """Entry point for YAAAT"""
    root = tk.Tk()
    app = YAATApp(root)
    root.geometry("1400x900")
    root.mainloop()


if __name__ == "__main__":
    main()
