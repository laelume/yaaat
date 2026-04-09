"""
core/grid_layer.py

Grid layout parent class for YAAAT annotator tabs that display
multiple spectrograms simultaneously in a paginated grid view.

Inherits from BaseLayer. Adds:
    - Grid figure and canvas (separate from the single-file spectrogram)
    - Spectrogram cache keyed by file index (grid_spectrograms)
    - Grid size selection UI
    - Page navigation
    - Override of change_nfft() and change_hop() to clear grid cache

Subclass GridLayer for any tab that requires a multi-file grid display.
Override:
    - setup_grid_controls()  : add grid-specific controls to control panel
    - process_grid_page()    : populate grid_spectrograms for current page
    - on_grid_click()        : handle click on a grid cell
    - draw_grid_overlays()   : draw overlays onto individual grid axes

Layout:
    The grid panel replaces the single-file spectrogram panel entirely.
    layer.ax and layer.canvas remain available for compatibility but are
    not displayed in grid mode — grid_fig and grid_canvas own the display.
"""

import logging
import traceback

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk

from yaaat.core.base_layer import BaseLayer
from yaaat.core import audio_utils

logger = logging.getLogger(__name__)

from yaaat.config import CONFIG


# (つ -' _ '- )つ    (つ -' _ '- )つ
# GRID DEFAULTS
# (つ -' _ '- )つ    (つ -' _ '- )つ

_DEFAULT_GRID_SIZE = CONFIG["grid_size"]    # files per page
_GRID_SIZE_OPTIONS = [4, 9, 16, 25, 36]
_GRID_REFRESH_DELAY_MS = 50  # ms delay before redraw after grid size change


##    <(''<)  <( ' ' )>  (>'')>

class GridLayer(BaseLayer):
    """Grid layout parent class for multi-file spectrogram display.

    Extends BaseLayer with a paginated grid view, per-file spectrogram
    cache, and grid-specific UI controls.
    """

    def __init__(self, root):
        """Initialize grid state before calling BaseLayer.__init__."""

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # GRID STATE — must exist before super().__init__() calls setup_ui()
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        # Spectrogram cache: {file_idx: S_db array}
        # Cleared when n_fft or hop_length changes
        self.grid_spectrograms = {}

        # Current page index (0-based)
        self.current_page = 0

        # Number of files displayed per page
        self.grid_size = CONFIG["grid_size"]

        # Grid figure and canvas — separate from BaseLayer's single-file canvas
        self.grid_fig = None
        self.grid_canvas = None

        # List of axes objects in the current grid layout
        self.grid_axes = []

        # Index of the currently selected grid cell (file_idx), or None
        self.selected_cell = None

        # Active grid size tk var for button highlight tracking
        self.grid_size_var = tk.IntVar(value=_DEFAULT_GRID_SIZE)

        super().__init__(root)

    ##    <(''<)  <( ' ' )>  (>'')>
    # SUBCLASS HOOKS
    # Override in GridLayer subclasses for grid-specific behavior
    ##    <(''<)  <( ' ' )>  (>'')>

    def setup_grid_controls(self):
        """Add grid-specific controls to the control panel.

        Called from setup_custom_controls(). Override in subclasses to
        inject additional controls below the grid size selector.
        """
        pass

    def process_grid_page(self):
        """Populate grid_spectrograms for all files on the current page.

        Override in subclasses to compute or load spectrograms for
        the files visible in the current grid page.
        """
        pass

    def on_grid_click(self, file_idx, event):
        """Handle a click on grid cell corresponding to file_idx.

        Override in subclasses to implement selection, annotation,
        or navigation on cell click.

        Args:
            file_idx: int — index into layer.audio_files
            event:    matplotlib MouseEvent
        """
        pass

    def draw_grid_overlays(self, ax, file_idx):
        """Draw overlays onto a single grid cell axis.

        Override in subclasses to add annotation markers, selection
        highlights, or contour overlays to individual grid cells.

        Args:
            ax:       matplotlib Axes — the cell axis
            file_idx: int — index into layer.audio_files
        """
        pass

    ##    <(''<)  <( ' ' )>  (>'')>
    # CUSTOM CONTROLS — injects grid controls into base control panel
    ##    <(''<)  <( ' ' )>  (>'')>

    def setup_custom_controls(self):
        """Build grid size selector and page navigation in the control panel.

        Calls setup_grid_controls() for subclass-specific additions.
        """
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # GRID SIZE SELECTOR
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(
            self.control_panel, text="Grid Size:", font=('', 9, 'bold')
        ).pack(anchor=tk.W, pady=(0, 2))

        grid_size_frame = ttk.Frame(self.control_panel)
        grid_size_frame.pack(fill=tk.X, pady=2)

        self.grid_size_buttons = []
        for size in _GRID_SIZE_OPTIONS:
            btn = tk.Button(
                grid_size_frame, text=str(size), width=4,
                command=lambda s=size: self.change_grid_size(s)
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.grid_size_buttons.append((btn, size))

        self._update_grid_size_highlights()

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # PAGE NAVIGATION
        # (つ -' _ '- )つ    (つ -' _ '- )つ

        ttk.Label(
            self.control_panel, text="Page:", font=('', 9, 'bold')
        ).pack(anchor=tk.W, pady=(4, 2))

        page_nav_frame = ttk.Frame(self.control_panel)
        page_nav_frame.pack(fill=tk.X, pady=2)

        ttk.Button(
            page_nav_frame, text="◄ Prev Page", width=12,
            command=self.previous_page
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            page_nav_frame, text="Next Page ►", width=12,
            command=self.next_page
        ).pack(side=tk.LEFT, padx=2)

        # Page info label — updated by _update_page_label()
        self.page_label = ttk.Label(
            self.control_panel, text="Page 1 / 1", font=('', 8)
        )
        self.page_label.pack(anchor=tk.W, pady=2)

        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        # Subclass-specific grid controls
        self.setup_grid_controls()


    ##    <(''<)  <( ' ' )>  (>'')>
    # UI COMPOSITION OVERRIDE
    # Replaces BaseLayer's single-file spectrogram panel with the grid canvas.
    ##    <(''<)  <( ' ' )>  (>'')>

    def setup_ui(self):
        """Override BaseLayer.setup_ui() to replace the spectrogram panel with the grid canvas.

        Calls super().setup_ui() to build the control panel and plot_frame.
        Then destroys all children of self.plot_frame and calls setup_grid_view()
        to inject the grid canvas in their place.

        BaseLayer's matplotlib figure (self.fig, self.ax, self.canvas) is still
        created by super().setup_ui() but is never packed into the display —
        it is replaced by self.grid_fig and self.grid_canvas from setup_grid_view().
        self.ax and self.canvas remain available for compatibility with any
        BaseLayer methods that reference them, but are not visible.
        """
        super().setup_ui()

        # (つ -' _ '- )つ    (つ -' _ '- )つ
        # Remove all widgets BaseLayer packed into self.plot_frame.
        # This includes the nav bar, matplotlib canvas, bottom nav, and zoom label.
        # self.fig, self.ax, self.canvas remain as attributes for compatibility
        # but are detached from the display.
        # (つ -' _ '- )つ    (つ -' _ '- )つ
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Inject grid canvas into the now-empty plot_frame
        self.setup_grid_view(self.plot_frame)




    ##    <(''<)  <( ' ' )>  (>'')>
    # GRID CANVAS SETUP
    # Called after audio is loaded — builds the grid figure in the plot panel
    ##    <(''<)  <( ' ' )>  (>'')>

    def setup_grid_view(self, plot_frame):
        """Build the grid figure and canvas inside plot_frame.

        Creates a scrollable container so large grids remain accessible.
        Binds mousewheel scrolling to the container.

        Args:
            plot_frame: ttk.Frame — the right panel frame from setup_ui()
        """
        # Scrollable container for the grid canvas
        grid_container = ttk.Frame(plot_frame)
        grid_container.pack(fill=tk.BOTH, expand=True)

        grid_canvas_tk = tk.Canvas(grid_container)
        grid_scrollbar = ttk.Scrollbar(
            grid_container, orient="vertical", command=grid_canvas_tk.yview)

        grid_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        grid_canvas_tk.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        grid_canvas_tk.configure(yscrollcommand=grid_scrollbar.set)

        # Bind mousewheel to the grid scroll container
        grid_canvas_tk.bind("<Enter>", lambda e: grid_canvas_tk.bind_all(
            "<MouseWheel>",
            lambda ev: grid_canvas_tk.yview_scroll(
                int(-1 * (ev.delta / 120)), "units")))
        grid_canvas_tk.bind("<Leave>", lambda e: grid_canvas_tk.unbind_all("<MouseWheel>"))

        # Grid matplotlib figure embedded in the scrollable container
        self.grid_fig = Figure(figsize=(12, 10))
        self.grid_canvas = FigureCanvasTkAgg(self.grid_fig, master=grid_canvas_tk)
        self.grid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        grid_canvas_tk.create_window(
            (0, 0), window=self.grid_canvas.get_tk_widget(), anchor="nw")

        self.grid_canvas.get_tk_widget().bind(
            "<Configure>",
            lambda e: grid_canvas_tk.configure(
                scrollregion=grid_canvas_tk.bbox("all")))

    ##    <(''<)  <( ' ' )>  (>'')>
    # GRID DISPLAY
    ##    <(''<)  <( ' ' )>  (>'')>

    def update_grid_display(self):
        """Redraw the grid figure with spectrograms for the current page.

        Computes grid dimensions from grid_size (nearest square root).
        Calls process_grid_page() to populate grid_spectrograms cache.
        Calls draw_grid_overlays() for each cell after rendering.
        Binds on_grid_click() to each cell axis.
        """
        print(f"DEBUG update_grid_display: audio_files={len(self.audio_files)}, grid_fig={self.grid_fig}")

        if not self.audio_files or self.grid_fig is None:
            return

        try:
            self.grid_fig.clf()

            # Compute grid layout
            cols = int(np.ceil(np.sqrt(self.grid_size)))
            rows = int(np.ceil(self.grid_size / cols))

            # Resize figure to match canvas dimensions
            canvas_w = self.grid_canvas.get_tk_widget().winfo_width()
            canvas_h = self.grid_canvas.get_tk_widget().winfo_height()
            fig_w = canvas_w / self.grid_fig.dpi if canvas_w > 1 else 12
            fig_h = canvas_h / self.grid_fig.dpi if canvas_h > 1 else 10
            self.grid_fig.set_size_inches(fig_w, fig_h, forward=True)

            # Compute file indices for this page
            start_idx = self.current_page * self.grid_size
            end_idx   = min(start_idx + self.grid_size, len(self.audio_files))
            page_file_indices = list(range(start_idx, end_idx))

            # Populate spectrogram cache for visible files
            self.process_grid_page()

            self.grid_axes = []

            for cell_pos, file_idx in enumerate(page_file_indices):
                ax = self.grid_fig.add_subplot(rows, cols, cell_pos + 1)
                self.grid_axes.append(ax)

                filepath_key = str(self.audio_files[file_idx])
                if filepath_key in self.grid_spectrograms:
                    spec = self.grid_spectrograms[filepath_key]

                    # Square cells — extent normalised to [0,1] for uniform display
                    extent = [0, 1, 0, 1]
                    ax.imshow(
                        spec, aspect='auto', origin='lower',
                        extent=extent, cmap='magma', interpolation='nearest'
                    )

                    # Cell title: filename stem, truncated for readability
                    stem = self.audio_files[file_idx].stem
                    title = stem[:12] + '…' if len(stem) > 12 else stem
                    ax.set_title(title, fontsize=6, pad=2)

                else:
                    # No spectrogram available for this cell
                    ax.text(0.5, 0.5, '...', ha='center', va='center',
                            fontsize=8, transform=ax.transAxes)

                # Suppress axis ticks for a clean grid appearance
                ax.set_xticks([])
                ax.set_yticks([])

                # Highlight selected cell
                if file_idx == self.selected_cell:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('cyan')
                        spine.set_linewidth(2)

                # Subclass overlay (selection state, annotations, etc.)
                self.draw_grid_overlays(ax, file_idx)

                # Bind click to this cell
                # Capture file_idx in closure via default argument
                self.grid_canvas.mpl_connect(
                    'button_press_event',
                    lambda ev, idx=file_idx, a=ax: self._on_grid_cell_press(ev, idx, a)
                )

            self.grid_fig.tight_layout(pad=0.4)
            self.grid_canvas.draw()

            self._update_page_label()

        except Exception as e:
            logger.error("ERROR in update_grid_display: %s", e)
            logger.debug(traceback.format_exc())

    def _on_grid_cell_press(self, event, file_idx, ax):
        """Internal dispatcher for grid cell clicks.

        Verifies the click landed in the correct cell axis before
        forwarding to on_grid_click().

        Args:
            event:    matplotlib MouseEvent
            file_idx: int — file index for this cell
            ax:       matplotlib Axes — this cell's axis
        """
        if event.inaxes == ax:
            self.selected_cell = file_idx
            self.on_grid_click(file_idx, event)
            self.update_grid_display()

    ##    <(''<)  <( ' ' )>  (>'')>
    # GRID SIZE AND PAGE NAVIGATION
    ##    <(''<)  <( ' ' )>  (>'')>

    def change_grid_size(self, new_size):
        """Change the number of files displayed per page and redraw.

        Resets to page 0 on grid size change to avoid empty pages.
        Uses a short delay to allow the canvas to resize before redrawing.

        Args:
            new_size: int — number of cells per page
        """
        self.grid_size = new_size
        self.grid_size_var.set(new_size)
        self.current_page = 0
        self._update_grid_size_highlights()

        # Brief delay ensures canvas geometry is settled before redraw
        self.grid_fig.clf()
        self.grid_canvas.draw()
        self.root.after(_GRID_REFRESH_DELAY_MS, self.update_grid_display)

    def next_page(self):
        """Advance to the next grid page if one exists."""
        total_pages = self._total_pages()
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.update_grid_display()

    def previous_page(self):
        """Go back to the previous grid page if not already on page 0."""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_grid_display()

    def _total_pages(self):
        """Return total number of pages given current audio_files and grid_size.

        Returns:
            int — always at least 1
        """
        if not self.audio_files:
            return 1
        return max(1, int(np.ceil(len(self.audio_files) / self.grid_size)))

    def _update_page_label(self):
        """Update the page info label with current page and total pages."""
        if hasattr(self, 'page_label') and self.page_label.winfo_exists():
            self.page_label.config(
                text=f"Page {self.current_page + 1} / {self._total_pages()}"
            )

    def _update_grid_size_highlights(self):
        """Highlight the currently active grid size button."""
        if not hasattr(self, 'grid_size_buttons'):
            return
        for btn, size in self.grid_size_buttons:
            if size == self.grid_size:
                btn.config(bg='lightgreen', relief=tk.SUNKEN)
            else:
                btn.config(bg='SystemButtonFace', relief=tk.RAISED)

    ##    <(''<)  <( ' ' )>  (>'')>
    # SPECTROGRAM PARAMETER OVERRIDES
    # Clear grid cache on parameter changes before delegating to BaseLayer
    ##    <(''<)  <( ' ' )>  (>'')>

    def change_nfft(self, new_nfft):
        """Clear grid spectrogram cache and delegate to BaseLayer.change_nfft().

        Cache must be cleared before recompute so all grid cells
        are regenerated with the new parameter.

        Args:
            new_nfft: int — new n_fft value
        """
        self.grid_spectrograms.clear()
        super().change_nfft(new_nfft)

    def change_hop(self, new_hop):
        """Clear grid spectrogram cache and delegate to BaseLayer.change_hop().

        Cache must be cleared before recompute so all grid cells
        are regenerated with the new parameter.

        Args:
            new_hop: int — new hop_length value
        """
        self.grid_spectrograms.clear()
        super().change_hop(new_hop)

    ##    <(''<)  <( ' ' )>  (>'')>
    # ENTRY POINT
    ##    <(''<)  <( ' ' )>  (>'')>

def main():
    """Launch GridLayer as a standalone viewer for development and testing."""
    root = tk.Tk()
    app = GridLayer(root)
    root.geometry("1400x900")
    root.mainloop()


if __name__ == "__main__":
    main()

# U S A G I
# from yaaat.core.grid_layer import GridLayer
# class BinaryAnnotator(GridLayer):
#     def process_grid_page(self): ...
#     def on_grid_click(self, file_idx, event): ...
#     def draw_grid_overlays(self, ax, file_idx): ...