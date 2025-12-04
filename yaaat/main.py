"""Main launcher for YAAAT with tabbed interface"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path

try:
    # Package mode: python -m yaaat
    from yaaat.changepoint_annotator import ChangepointAnnotator
    from yaaat.peak_annotator import PeakAnnotator
    from yaaat.harmonic_annotator import HarmonicAnnotator
    from yaaat.sequence_annotator import SequenceAnnotator
except ImportError:
    # Script mode: python main.py
    from changepoint_annotator import ChangepointAnnotator
    from peak_annotator import PeakAnnotator
    from harmonic_annotator import HarmonicAnnotator
    from sequence_annotator import SequenceAnnotator


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
        sequence_frame = ttk.Frame(self.notebook)
        
        # Add tabs
        self.notebook.add(changepoint_frame, text="Changepoint Annotator")
        self.notebook.add(peak_frame, text="Peak Annotator")
        self.notebook.add(harmonic_frame, text="Harmonic Annotator")
        self.notebook.add(sequence_frame, text="Sequence Annotator")

        # Initialize tools (pass frames as parent)
        self.changepoint_tool = ChangepointAnnotator(changepoint_frame)
        self.peak_tool = PeakAnnotator(peak_frame)
        self.harmonic_tool = HarmonicAnnotator(harmonic_frame)
        self.sequence_tool = SequenceAnnotator(sequence_frame)
        
        # Share audio files across all tabs
        self.tools = [self.changepoint_tool, self.peak_tool, self.harmonic_tool, self.sequence_tool]
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)


    def sync_audio_state(self, source_tool):
        """Synchronize audio files and current position from source tool to all others"""
        if not source_tool.audio_files:
            return
        
        for tool in self.tools:
            if tool is source_tool:
                continue
            
            # Copy audio state
            tool.audio_files = source_tool.audio_files
            tool.base_audio_dir = source_tool.base_audio_dir
            tool.current_file_idx = source_tool.current_file_idx
            tool.annotation_dir = source_tool.annotation_dir  # For peak/harmonic tools
            
            # Update UI elements that exist
            if hasattr(tool, 'save_dir_button'):
                tool.save_dir_button.config(text=f"📁 {tool.annotation_dir}")
            
            # Load the current file in the tool
            if tool.audio_files:
                tool.load_current_file()


    def on_tab_change(self, event):
        """Handle tab switching - sync audio state"""
        current_tab_idx = self.notebook.index(self.notebook.select())
        current_tool = self.tools[current_tab_idx]
        
        # Find most recently used tool (one with audio files loaded)
        source_tool = None
        for tool in self.tools:
            if tool.audio_files:
                source_tool = tool
                break
        
        if source_tool and source_tool is not current_tool:
            self.sync_audio_state(source_tool)
            print(f"Synced audio from {source_tool.__class__.__name__} to {current_tool.__class__.__name__}")


def main():
    """Entry point for YAAAT"""
    root = tk.Tk()
    app = YAAATApp(root)
    root.geometry("1400x900")
    root.mainloop()


if __name__ == "__main__":
    main()
