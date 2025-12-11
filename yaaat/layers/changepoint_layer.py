"""
Changepoint Layer - Built on BaseLayer
Interactive tool for annotating time-frequency changepoints on spectrograms
"""

from base_layer import BaseLayer
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from pathlib import Path
import json


class ChangepointLayer(BaseLayer):
    """Changepoint annotation tool for syllable segmentation"""
    
    def __init__(self, root):
        # Algorithm-specific data (initialize BEFORE calling super)
        self.current_contour = []
        self.contours = []
        self.annotations = []
        self.changepoints = []
        
        # Annotation mode
        self.annotation_mode = tk.StringVar(value='contour')
        
        # Onset/offset marking
        self.pending_onset_idx = None
        
        # Lasso selection
        self.lasso_mode = False
        self.lasso_points = []
        self.lasso_lines = []
        
        # Guide lines
        self.show_time_guides = tk.BooleanVar(value=False)
        self.show_freq_guides = tk.BooleanVar(value=False)
        self.show_all_guides_var = tk.BooleanVar(value=False)
        self.hide_text = tk.BooleanVar(value=False)
        
        # Bounding box
        self.show_bounding_box = tk.BooleanVar(value=False)
        self.bounding_box_shape = tk.StringVar(value='rectangle')
        
        # Harmonics for bounding boxes
        self.harmonics = [
            {'multiplier': tk.DoubleVar(value=2.0), 'show': tk.BooleanVar(value=False), 
             'label': None, 'color': 'cyan', 'name': '2nd'},
            {'multiplier': tk.DoubleVar(value=3.0), 'show': tk.BooleanVar(value=False), 
             'label': None, 'color': 'orange', 'name': '3rd'}
        ]
        self.harmonic_repeat_ids = {}
        self.dragging_harmonic = None
        
        # Tracking
        self.total_syllables_across_files = 0
        self.total_skipped_files = 0
        self.file_was_annotated = False
        
        # Call parent constructor
        super().__init__(root)
        
        if isinstance(root, tk.Tk):
            self.root.title("Changepoint Annotator - YAAAT")
    
    def setup_custom_controls(self):
        """Add changepoint-specific controls"""
        # Instructions
        ttk.Label(self.control_panel, text="Instructions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        instructions = ttk.Label(self.control_panel, 
                        text="• Click: add point\n"
                             "• Click near point: remove\n"
                             "• Ctrl+Click: lasso selection\n"
                             "• Ctrl+Click+Click: mark endpoints", 
                        wraplength=400, font=('', 8))
        instructions.pack(padx=5, pady=(0, 5))
        
        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # Annotation mode
        ttk.Label(self.control_panel, text="Mode:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        mode_frame = ttk.Frame(self.control_panel)
        mode_frame.pack(fill=tk.X, pady=2)
        
        ttk.Radiobutton(mode_frame, text="Contour", variable=self.annotation_mode, 
                        value='contour', command=self.switch_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Sequence", variable=self.annotation_mode,
                        value='sequence', command=self.switch_mode).pack(side=tk.LEFT, padx=5)
        
        self.mode_instructions = ttk.Label(self.control_panel, 
            text="Contour: Click points → Finish Contour",
            wraplength=400, font=('', 8, 'italic'), foreground='blue')
        self.mode_instructions.pack(pady=2)
        
        # Syllable info
        self.syllable_info = ttk.Label(self.control_panel, 
            text="Unsaved: 0 | Saved: 0 | Total: 0", 
            wraplength=400, font=('', 8))
        self.syllable_info.pack(fill=tk.X, pady=2)
        
        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # Guides
        ttk.Label(self.control_panel, text="Guides:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        guides_grid = ttk.Frame(self.control_panel)
        guides_grid.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(guides_grid, text="Time Lines", variable=self.show_time_guides, 
                       command=self.toggle_guides).grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Freq Lines", variable=self.show_freq_guides, 
                       command=self.toggle_guides).grid(row=0, column=1, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Show All", variable=self.show_all_guides_var, 
                       command=self.toggle_show_all).grid(row=0, column=2, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(guides_grid, text="Hide Text", variable=self.hide_text, 
                       command=self.toggle_guides).grid(row=0, column=3, sticky=tk.W, padx=2, pady=2)
        
        ttk.Checkbutton(guides_grid, text="Bounding Box", variable=self.show_bounding_box, 
                       command=self.toggle_guides).grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        
        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # Actions
        ttk.Label(self.control_panel, text="Contour Actions:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        button_grid = ttk.Frame(self.control_panel)
        button_grid.pack(pady=2)
        
        buttons = [
            ("Finish Contour", self.finish_contour),
            ("Clear Last", self.clear_last),
            ("Clear All", self.clear_all),
            ("Skip File", self.skip_file),
            ("Save Anno", self.save_custom_data),
        ]
        
        for i, (text, command) in enumerate(buttons):
            ttk.Button(button_grid, text=text, command=command, width=12).grid(
                row=i//3, column=i%3, padx=2, pady=2, sticky='ew')
        
        ttk.Separator(self.control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)
        
        # Statistics
        ttk.Label(self.control_panel, text="Statistics:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.stats_label = ttk.Label(self.control_panel, text="No annotations", 
                                     justify=tk.LEFT, font=('', 8))
        self.stats_label.pack(fill=tk.X, pady=2)
        
        # Annotations table
        ttk.Label(self.control_panel, text="Contours:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        
        # Table header
        header_frame = ttk.Frame(self.control_panel)
        header_frame.pack(fill=tk.X, pady=2)
        ttk.Label(header_frame, text="#", font=('', 8, 'bold'), width=3).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="t_onset", font=('', 8, 'bold'), width=6).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="t_offset", font=('', 8, 'bold'), width=6).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="f_min", font=('', 8, 'bold'), width=6).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="f_max", font=('', 8, 'bold'), width=6).pack(side=tk.LEFT)
        
        # Scrollable table
        table_frame = ttk.Frame(self.control_panel, height=150)
        table_frame.pack(fill=tk.X, pady=2)
        table_frame.pack_propagate(False)
        
        ann_canvas = tk.Canvas(table_frame, highlightthickness=0)
        ann_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=ann_canvas.yview)
        self.annotations_inner_frame = ttk.Frame(ann_canvas)
        
        ann_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ann_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ann_canvas.configure(yscrollcommand=ann_scrollbar.set)
        
        ann_canvas.create_window((0, 0), window=self.annotations_inner_frame, anchor="nw")
        self.annotations_inner_frame.bind("<Configure>", 
            lambda e: ann_canvas.configure(scrollregion=ann_canvas.bbox("all")))
    
    def process_audio(self):
        """Process audio (no automatic detection for changepoints)"""
        pass
    
    def switch_mode(self):
        """Switch annotation mode"""
        mode = self.annotation_mode.get()
        if mode == 'sequence':
            self.mode_instructions.config(text="Sequence: Mark syllable boundaries")
        else:
            self.mode_instructions.config(text="Contour: Click points → Finish Contour")
    
    def on_custom_press(self, event):
        """Handle changepoint-specific press events"""
        # Check for Ctrl+Click (lasso or endpoint marking)
        import sys
        is_ctrl = False
        if sys.platform == 'win32':
            import ctypes
            is_ctrl = bool(ctypes.windll.user32.GetKeyState(0x11) & 0x8000)
        else:
            is_ctrl = hasattr(event, 'key') and event.key in ['control', 'ctrl']
        
        if is_ctrl and (self.current_contour or self.contours):
            # Check if near existing point
            time_threshold = 0.02
            freq_threshold = 100
            
            clicked_near = self.find_nearest_point(event.xdata, event.ydata, time_threshold, freq_threshold)
            
            if clicked_near:
                # Endpoint marking mode
                self.drag_start = (event.xdata, event.ydata)
                return True
            else:
                # Start lasso
                self.lasso_mode = True
                self.lasso_points = [(event.xdata, event.ydata)]
                self.drag_start = (event.xdata, event.ydata)
                print("✓ Lasso mode started")
                return True
        
        return False
    
    def on_custom_motion(self, event):
        """Handle lasso drawing"""
        if self.lasso_mode and self.drag_start is not None:
            self.lasso_points.append((event.xdata, event.ydata))
            self.draw_lasso_preview()
            return True
        
        return False
    
    def on_custom_release(self, event):
        """Handle point addition or lasso finish"""
        if self.lasso_mode:
            if len(self.lasso_points) >= 2:
                self.finish_lasso_selection()
            else:
                self.cancel_lasso()
            return True
        
        # Check for point addition/removal
        if event.xdata is None or event.ydata is None:
            return False
        
        drag_dist = 0
        if self.drag_start:
            x0, y0 = self.drag_start
            drag_dist = np.sqrt((event.xdata - x0)**2 + (event.ydata - y0)**2)
        
        if drag_dist < 0.05:
            # Click - add or remove point
            if self.remove_nearby_annotation(event.xdata, event.ydata):
                print("Removed point")
            else:
                self.current_contour.append({
                    'time': float(event.xdata),
                    'freq': float(event.ydata)
                })
                self.changes_made = True
                self.rebuild_annotations()
                self.update_display(recompute_spec=False)
                print(f"+ Point {len(self.current_contour)}: t={event.xdata:.3f}s, f={event.ydata:.0f}Hz")
            return True
        
        return False
    
    def find_nearest_point(self, x, y, time_threshold, freq_threshold):
        """Find if click is near existing point"""
        for point in self.current_contour:
            if abs(point['time'] - x) < time_threshold and abs(point['freq'] - y) < freq_threshold:
                return point
        
        for contour in self.contours:
            points = contour['points'] if isinstance(contour, dict) else contour
            for point in points:
                if abs(point['time'] - x) < time_threshold and abs(point['freq'] - y) < freq_threshold:
                    return point
        
        return None
    
    def draw_lasso_preview(self):
        """Draw lasso path"""
        for line in self.lasso_lines:
            try:
                line.remove()
            except:
                pass
        self.lasso_lines = []
        
        if len(self.lasso_points) < 2:
            return
        
        xs = [p[0] for p in self.lasso_points]
        ys = [p[1] for p in self.lasso_points]
        
        line = self.ax.plot(xs, ys, 'y-', linewidth=2, alpha=0.7)[0]
        self.lasso_lines.append(line)
        
        self.canvas.draw_idle()
    
    def finish_lasso_selection(self):
        """Extract points within lasso"""
        print(f"✓ Lasso selection with {len(self.lasso_points)} vertices")
        # Simplified - would need full point_in_polygon logic
        self.cancel_lasso()
    
    def cancel_lasso(self):
        """Cancel lasso"""
        self.lasso_mode = False
        self.lasso_points = []
        for line in self.lasso_lines:
            try:
                line.remove()
            except:
                pass
        self.lasso_lines = []
        self.canvas.draw_idle()
    
    def finish_contour(self):
        """Finish current contour"""
        if len(self.current_contour) < 2:
            messagebox.showwarning("Need More Points", "Need at least 2 points")
            return
        
        sorted_points = sorted(self.current_contour, key=lambda x: x['time'])
        self.contours.append({
            'points': sorted_points,
            'onset_idx': 0,
            'offset_idx': len(sorted_points) - 1
        })
        
        self.current_contour = []
        self.changes_made = True
        self.rebuild_annotations()
        self.update_display(recompute_spec=False)
        print(f"✓ Contour complete ({len(self.contours)} total)")
    
    def clear_last(self):
        """Remove last point"""
        if self.current_contour:
            self.current_contour.pop()
            self.changes_made = True
            self.rebuild_annotations()
            self.update_display(recompute_spec=False)
        elif self.contours:
            self.current_contour = self.contours.pop()['points']
            self.changes_made = True
            self.rebuild_annotations()
            self.update_display(recompute_spec=False)
    
    def clear_all(self):
        """Clear all annotations"""
        if (self.annotations or self.current_contour) and messagebox.askyesno("Clear", "Remove all?"):
            self.annotations = []
            self.current_contour = []
            self.contours = []
            self.changes_made = True
            self.update_display(recompute_spec=False)
    
    def skip_file(self):
        """Skip current file"""
        # Simplified - would show dialog for reason
        print("✓ Skipped file")
        self.next_file()
    
    def rebuild_annotations(self):
        """Rebuild annotation list"""
        self.annotations = []
        
        for contour in self.contours:
            points = contour['points'] if isinstance(contour, dict) else contour
            onset_idx = contour.get('onset_idx', 0) if isinstance(contour, dict) else 0
            offset_idx = contour.get('offset_idx', len(points) - 1) if isinstance(contour, dict) else len(points) - 1
            
            for i, point in enumerate(points):
                if i == onset_idx:
                    label = 'onset'
                elif i == offset_idx:
                    label = 'offset'
                else:
                    label = 'changepoint'
                
                self.annotations.append({
                    'time': point['time'],
                    'freq': point['freq'],
                    'label': label
                })
        
        for point in self.current_contour:
            self.annotations.append({
                'time': point['time'],
                'freq': point['freq'],
                'label': 'changepoint'
            })
        
        unsaved = len(self.current_contour)
        saved = sum(len(c['points']) if isinstance(c, dict) else len(c) for c in self.contours)
        self.syllable_info.config(text=f"Unsaved: {unsaved} | Saved: {saved} | Total: {self.total_syllables_across_files}")
        
        self.update_stats()
        self.update_annotations_table()
    
    def update_annotations_table(self):
        """Update table display"""
        for widget in self.annotations_inner_frame.winfo_children():
            widget.destroy()
        
        for idx, contour in enumerate(self.contours):
            points = contour['points'] if isinstance(contour, dict) else contour
            if len(points) < 1:
                continue
            
            sorted_points = sorted(points, key=lambda x: x['time'])
            t_onset = sorted_points[0]['time']
            t_offset = sorted_points[-1]['time']
            freqs = [p['freq'] for p in points]
            f_min = min(freqs)
            f_max = max(freqs)
            
            row = ttk.Frame(self.annotations_inner_frame)
            row.pack(fill=tk.X, pady=1)
            
            ttk.Label(row, text=f"{idx+1}", font=('', 8), width=3).pack(side=tk.LEFT)
            ttk.Label(row, text=f"{t_onset:.4f}", font=('', 8), width=7).pack(side=tk.LEFT)
            ttk.Label(row, text=f"{t_offset:.4f}", font=('', 8), width=7).pack(side=tk.LEFT)
            ttk.Label(row, text=f"{f_min:.0f}", font=('', 8), width=7).pack(side=tk.LEFT)
            ttk.Label(row, text=f"{f_max:.0f}", font=('', 8), width=7).pack(side=tk.LEFT)
    
    def remove_nearby_annotation(self, x, y):
        """Remove annotation near click"""
        time_threshold = 0.01
        freq_threshold = 10
        
        for i, point in enumerate(self.current_contour):
            if abs(point['time'] - x) < time_threshold and abs(point['freq'] - y) < freq_threshold:
                self.current_contour.pop(i)
                self.changes_made = True
                self.rebuild_annotations()
                return True
        
        return False
    
    def toggle_guides(self):
        """Toggle guide visibility"""
        if self.y is not None:
            self.update_display(recompute_spec=False)
    
    def toggle_show_all(self):
        """Toggle all guides"""
        if self.show_all_guides_var.get():
            self.show_time_guides.set(True)
            self.show_freq_guides.set(True)
        else:
            self.show_time_guides.set(False)
            self.show_freq_guides.set(False)
        self.toggle_guides()
    
    def update_stats(self):
        """Update statistics"""
        if not self.annotations:
            self.stats_label.config(text="No annotations")
            return
        
        onset = next((a for a in self.annotations if a['label'] == 'onset'), None)
        offset = next((a for a in self.annotations if a['label'] == 'offset'), None)
        
        if onset and offset:
            duration = offset['time'] - onset['time']
            freqs = [a['freq'] for a in self.annotations]
            delta_freq = max(freqs) - min(freqs)
            
            self.stats_label.config(
                text=f"Duration: {duration:.3f}s | ΔFreq: {delta_freq:.1f} Hz")
        else:
            self.stats_label.config(text="Incomplete annotation")
    
    def draw_custom_overlays(self):
        """Draw annotation points and guides"""
        colors = {'onset': 'green', 'offset': 'magenta', 'changepoint': 'cyan'}
        
        for ann in self.annotations:
            self.ax.scatter(ann['time'], ann['freq'], 
                          c=colors[ann['label']], 
                          marker='.',
                          s=100, 
                          linewidths=1,
                          zorder=10)
        
        # Guide lines
        if self.annotations and (self.show_time_guides.get() or self.show_freq_guides.get()):
            times = [ann['time'] for ann in self.annotations]
            freqs = [ann['freq'] for ann in self.annotations]
            
            if self.show_time_guides.get():
                for t in times:
                    self.ax.axvline(x=t, color='lime', linestyle='--', linewidth=1.5, alpha=0.5)
            
            if self.show_freq_guides.get():
                for f in freqs:
                    self.ax.axhline(y=f, color='yellow', linestyle='--', linewidth=1.5, alpha=0.5)
    
    def save_custom_data(self):
        """Save changepoint annotations"""
        if not self.audio_files or self.annotation_dir is None:
            return
        
        audio_file = self.audio_files[self.current_file_idx]
        relative_path = audio_file.relative_to(self.base_audio_dir).parent
        filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
        
        if filename_prefix and filename_prefix != '.':
            annotation_file = self.annotation_dir / f"{filename_prefix}_{audio_file.stem}_changepoint_annotations.json"
        else:
            annotation_file = self.annotation_dir / f"{audio_file.stem}_changepoint_annotations.json"
        
        # Convert contours to serializable format
        serializable_contours = []
        for contour in self.contours:
            if isinstance(contour, dict):
                serializable_contours.append(contour['points'])
            else:
                serializable_contours.append(contour)
        
        data = {
            'audio_file': str(audio_file),
            'annotations': self.annotations,
            'syllables': serializable_contours,
            'spec_params': {
                'n_fft': self.n_fft.get(),
                'hop_length': self.hop_length.get()
            }
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.changes_made = False
        self.update_display(recompute_spec=False)
        print(f"✓ Saved to {annotation_file.name}")
    
    def load_custom_data(self):
        """Load changepoint annotations"""
        if not self.audio_files or self.annotation_dir is None:
            return
        
        audio_file = self.audio_files[self.current_file_idx]
        relative_path = audio_file.relative_to(self.base_audio_dir).parent
        filename_prefix = str(relative_path).replace('/', '_').replace('\\', '_')
        
        if filename_prefix and filename_prefix != '.':
            annotation_file = self.annotation_dir / f"{filename_prefix}_{audio_file.stem}_changepoint_annotations.json"
        else:
            annotation_file = self.annotation_dir / f"{audio_file.stem}_changepoint_annotations.json"
        
        # Clear state
        self.annotations = []
        self.current_contour = []
        self.contours = []
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                
                syllables = data.get('syllables', [])
                for syll in syllables:
                    if isinstance(syll, list) and len(syll) > 0:
                        self.contours.append({
                            'points': syll,
                            'onset_idx': 0,
                            'offset_idx': len(syll) - 1
                        })
                
                if self.contours:
                    self.rebuild_annotations()
                    print(f"✓ Loaded {len(self.contours)} contours")


def main():
    """Entry point"""
    root = tk.Tk()
    app = ChangepointAnnotator(root)
    root.geometry("1400x800")
    root.mainloop()


if __name__ == "__main__":
    main()