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
import time
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

# === === === === === === === === === === === === === === === === === ===
# NAV DEBUG GATE
# _DEBUG_NAV forces navigation debug logging on regardless of logger level.
# Leave False in normal operation; logger.debug still fires when the logger
# level is set to DEBUG. Two independent toggles: this constant OR log level.
# Used by _dbg_nav() to instrument page turns, redraws, and cell dispatch.
# === === === === === === === === === === === === === === === === === ===

_DEBUG_NAV = True  # VERBOSE_NAV: set True to force-print navigation trace


def _dbg_nav(msg):
    """Emit a navigation debug line when verbose nav tracing is active.

    Routes through logger.debug so output respects the verbosity flag.
    When _DEBUG_NAV is True, also prints unconditionally to stdout so the
    trace is visible without reconfiguring the logger level.

    Args:
        msg: str — preformatted debug message
    """
    # VERBOSE_NAV: dual-path emit — logger for normal use, print for force mode
    logger.debug(msg)
    if _DEBUG_NAV:
        print(f"[NAV] {msg}")

# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ====

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

        # === === === === === === === === === === === === === === === === ===
        # MPL CONNECT ACCUMULATION COUNTER
        # Incremented once per mpl_connect call in update_grid_display.
        # Never reset — running total is the accumulation proxy. A value that
        # climbs by one cell-set per redraw confirms callbacks are never
        # disconnected (direct cause candidate for the dispatch corruption and
        # the frozen-interface symptom on repeated navigation).
        # === === === === === === === === === === === === === === === === ===
        self._mpl_cid_count = 0  # VERBOSE_NAV: cumulative mpl_connect registrations

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
        
        # VERBOSE_NAV: replaced bare debug print with gated _dbg_nav emit
        _dbg_nav(
            f"update_grid_display ENTRY: n_files={len(self.audio_files)} "
            f"grid_fig={'set' if self.grid_fig is not None else 'None'} "
            f"current_page={self.current_page}"
        )

        if not self.audio_files or self.grid_fig is None:
            # VERBOSE_NAV: early return — nothing to draw or canvas not built yet
            _dbg_nav("update_grid_display ABORT: no audio_files or grid_fig is None")
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

            # VERBOSE_NAV: canvas geometry trace — exposes the container-layout
            # hypothesis. If canvas_w/canvas_h stay at 1 across page turns the
            # widget is not receiving geometry, implicating the pack/create_window
            # conflict in setup_grid_view rather than the page-index logic.
            _dbg_nav(
                f"update_grid_display GEOM: canvas_w={canvas_w} canvas_h={canvas_h} "
                f"fig_w={fig_w:.2f} fig_h={fig_h:.2f} rows={rows} cols={cols}"
            )

            # Compute file indices for this page
            start_idx = self.current_page * self.grid_size
            end_idx   = min(start_idx + self.grid_size, len(self.audio_files))
            page_file_indices = list(range(start_idx, end_idx))

            # VERBOSE_NAV: page-bounds trace — confirms the index window shifts
            # on page turn. If start_idx/end_idx change but the display does not,
            # the redraw path is suspect, not the index arithmetic.
            _dbg_nav(
                f"update_grid_display BOUNDS: start_idx={start_idx} "
                f"end_idx={end_idx} n_cells={len(page_file_indices)}"
            )

            # VERBOSE_NAV: time the load+compute phase in isolation. This phase
            # runs synchronously on the Tk main thread; a large delta here is
            # consistent with the interface freeze on page turn.
            _t_load_start = time.perf_counter()

            # Populate spectrogram cache for visible files
            self.process_grid_page()

            _t_load_end = time.perf_counter()
            _dbg_nav(
                f"update_grid_display PHASE load+compute: "
                f"{(_t_load_end - _t_load_start) * 1000:.1f}ms "
                f"n_cells={len(page_file_indices)}"
            )

            # VERBOSE_NAV: mark start of the render phase (cell subplot build,
            # imshow, overlays, and the per-cell mpl_connect registration).
            _t_render_start = time.perf_counter()

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
                # VERBOSE_NAV: increment cumulative registration counter. No
                # disconnect is paired with this connect — the running total is
                # the accumulation evidence, logged once per redraw below.
                self._mpl_cid_count += 1

            self.grid_fig.tight_layout(pad=0.4)

            self.grid_canvas.draw()

            # VERBOSE_NAV: render phase delta and cumulative connection count.
            # Reading: if load+compute dominates, the freeze is I/O+STFT bound
            # (address in process_grid_page — threading/checkpoint). If render
            # grows each turn while _mpl_cid_count climbs, the freeze is
            # accumulation-bound (address by disconnecting stale callbacks).
            _t_render_end = time.perf_counter()
            _dbg_nav(
                f"update_grid_display PHASE render+draw: "
                f"{(_t_render_end - _t_render_start) * 1000:.1f}ms "
                f"mpl_cid_total={self._mpl_cid_count}"
            )


            self._update_page_label()

        except Exception as e:
            logger.error("ERROR in update_grid_display: %s", e)
            logger.debug(traceback.format_exc())

    # def _on_grid_cell_press(self, event, file_idx, ax):
    #     """Internal dispatcher for grid cell clicks.

    #     Verifies the click landed in the correct cell axis before
    #     forwarding to on_grid_click().

    #     Args:
    #         event:    matplotlib MouseEvent
    #         file_idx: int — file index for this cell
    #         ax:       matplotlib Axes — this cell's axis
    #     """
    #     if event.inaxes == ax:
    #         self.selected_cell = file_idx
    #         self.on_grid_click(file_idx, event)
    #         self.update_grid_display()


    def _on_grid_cell_press(self, event, file_idx, ax):
        """Internal dispatcher for grid cell clicks.

        Verifies the click landed in the correct cell axis before
        forwarding to on_grid_click().

        Args:
            event:    matplotlib MouseEvent
            file_idx: int — file index for this cell
            ax:       matplotlib Axes — this cell's axis
        """
        # VERBOSE_NAV: dispatch trace — fires once per registered callback.
        # mpl_connect is called per-cell on every update_grid_display and the
        # connections are never disconnected, so callbacks accumulate across
        # redraws. Diagnostic reading: a SINGLE physical click that emits more
        # than one DISPATCH line confirms callback accumulation. If the count
        # grows by one cell-set per page turn, the accumulation is the cause of
        # corrupted cell dispatch. If instead exactly one DISPATCH fires per
        # click but the page does not change, the layout/redraw path is implicated.
        matched = (event.inaxes == ax)
        _dbg_nav(
            f"_on_grid_cell_press DISPATCH: file_idx={file_idx} "
            f"inaxes_match={matched} current_page={self.current_page}"
        )

        if matched:
            self.selected_cell = file_idx
            # VERBOSE_NAV: matched cell — forwarding to subclass click handler
            _dbg_nav(f"_on_grid_cell_press MATCH: forwarding file_idx={file_idx}")
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
        # VERBOSE_NAV: log size transition and the forced page reset
        _dbg_nav(
            f"change_grid_size ENTRY: old_size={self.grid_size} "
            f"new_size={new_size} current_page={self.current_page} -> 0"
        )

        self.grid_size = new_size
        self.grid_size_var.set(new_size)
        self.current_page = 0
        self._update_grid_size_highlights()

        # Brief delay ensures canvas geometry is settled before redraw
        self.grid_fig.clf()
        self.grid_canvas.draw()

        # VERBOSE_NAV: deferred redraw scheduled — confirms the after() callback
        # is queued. If the trace shows this line but no subsequent
        # update_grid_display ENTRY, the delayed callback is being dropped.
        _dbg_nav(
            f"change_grid_size SCHEDULED: redraw in {_GRID_REFRESH_DELAY_MS}ms"
        )
        self.root.after(_GRID_REFRESH_DELAY_MS, self.update_grid_display)


    def next_page(self):
        """Advance to the next grid page if one exists."""
        total_pages = self._total_pages()

        # VERBOSE_NAV: log entry state before any mutation
        _dbg_nav(
            f"next_page ENTRY: current_page={self.current_page} "
            f"total_pages={total_pages} grid_size={self.grid_size} "
            f"n_files={len(self.audio_files)}"
        )


        if self.current_page < total_pages - 1:
            self.current_page += 1
            # VERBOSE_NAV: confirm increment took effect before redraw
            _dbg_nav(f"next_page ADVANCE: current_page -> {self.current_page}")
            # VERBOSE_NAV: bracket the full synchronous redraw to measure the
            # click-to-complete latency the frozen interface is exhibiting.
            _t_nav_start = time.perf_counter()
            self.update_grid_display()
            _dbg_nav(
                f"next_page TOTAL nav: "
                f"{(time.perf_counter() - _t_nav_start) * 1000:.1f}ms"
            )

        else:
            # VERBOSE_NAV: boundary hit — already on last page, no redraw
            _dbg_nav("next_page BLOCKED: already on last page")

    def previous_page(self):
        """Go back to the previous grid page if not already on page 0."""
        # VERBOSE_NAV: log entry state before any mutation
        _dbg_nav(
            f"previous_page ENTRY: current_page={self.current_page} "
            f"grid_size={self.grid_size} n_files={len(self.audio_files)}"
        )

        if self.current_page > 0:
            self.current_page -= 1
            # VERBOSE_NAV: confirm decrement took effect before redraw
            _dbg_nav(f"previous_page RETREAT: current_page -> {self.current_page}")
            # VERBOSE_NAV: bracket the full synchronous redraw to measure the
            # click-to-complete latency the frozen interface is exhibiting.
            _t_nav_start = time.perf_counter()
            self.update_grid_display()
            _dbg_nav(
                f"previous_page TOTAL nav: "
                f"{(time.perf_counter() - _t_nav_start) * 1000:.1f}ms"
            )
            
        else:
            # VERBOSE_NAV: boundary hit — already on page 0, no redraw
            _dbg_nav("previous_page BLOCKED: already on page 0")

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
# class BatchAnnotator(GridLayer):
#     def process_grid_page(self): ...
#     def on_grid_click(self, file_idx, event): ...
#     def draw_grid_overlays(self, ax, file_idx): ...