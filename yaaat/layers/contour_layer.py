# """
# Contour Annotator Layer for YAAAT
# Interactive contour extraction with toggleable processing steps
# """

# import numpy as np
# import tkinter as tk
# from tkinter import ttk
# import json
# from pathlib import Path
# import sys

# # from layers.base_layer import BaseLayer
# from .base_layer import BaseLayer

# from ..contours.processing import process_audio_file
# from ..contours.config import DEFAULT_CONFIG


# class ContourLayer(BaseLayer):
#     """
#     Contour extraction annotator with toggleable processing pipeline.
    
#     Processing pipeline stages:
#     1. Impulse denoising (toggleable)
#     2. Normalization + Cubing (always on)
#     3. Rolling ball background subtraction (toggleable)
#     4. CLAHE enhancement (toggleable)
#     5. Ridge detection method (selectable: hessian/frangi/threshold/edge/adaptive)
#     6. Frangi fallback (toggleable, only for hessian)
#     7. Component selection (always on)
#     """
    
#     def __init__(self, root):
#         # Processing state
#         self.config = DEFAULT_CONFIG
#         self.result = None
        
#         # Toggle states for processing steps
#         self.enable_denoise = tk.BooleanVar(value=True)
#         self.enable_rolling_ball = tk.BooleanVar(value=True)
#         self.enable_clahe = tk.BooleanVar(value=True)
#         self.enable_frangi_fallback = tk.BooleanVar(value=False)
        
#         # Ridge detection method
#         self.ridge_method = tk.StringVar(value='hessian')
        
#         # Parameter controls
#         self.impulse_freq_thresh = tk.DoubleVar(value=0.4)
#         self.ridge_threshold = tk.DoubleVar(value=0.3)
#         self.hessian_sigma_min = tk.IntVar(value=1)
#         self.hessian_sigma_max = tk.IntVar(value=5)
        
#         # Component selection (d-train mode)
#         self.dtrain_mode = tk.BooleanVar(value=False)
        
#         # Display options
#         self.show_clahe = tk.BooleanVar(value=False)
#         self.show_component_mask = tk.BooleanVar(value=True)
#         self.show_energy_peak = tk.BooleanVar(value=True)

#         # Live updating grid window with customized parameters
#         self.grid_window = None
#         self.grid_canvas = None
#         self.grid_fig = None
        
#         super().__init__(root)
    
#     def setup_custom_controls(self):
#         """Add contour-specific controls"""
        
#         # ===== PROCESSING PIPELINE TOGGLES =====
#         pipeline_frame = ttk.LabelFrame(self.control_panel, text="Processing Pipeline", padding=5)
#         pipeline_frame.pack(fill=tk.X, pady=5)
        
#         ttk.Checkbutton(
#             pipeline_frame,
#             text="1. Impulse Denoising",
#             variable=self.enable_denoise,
#             command=self.reprocess_on_change
#         ).pack(anchor=tk.W, pady=2)
        
#         # Denoise parameter
#         denoise_param_frame = ttk.Frame(pipeline_frame)
#         denoise_param_frame.pack(fill=tk.X, padx=20, pady=2)
#         ttk.Label(denoise_param_frame, text="Freq thresh:", font=('', 8)).pack(side=tk.LEFT)
        
#         denoise_slider = ttk.Scale(
#             denoise_param_frame,
#             from_=0.1,
#             to=0.9,
#             orient=tk.HORIZONTAL,
#             variable=self.impulse_freq_thresh,
#             length=100
#         )
#         denoise_slider.pack(side=tk.LEFT, padx=5)
#         denoise_slider.bind('<ButtonRelease-1>', lambda e: self.reprocess_on_change())  # ADD THIS

#         ttk.Label(denoise_param_frame, textvariable=self.impulse_freq_thresh, 
#                  font=('', 7), width=4).pack(side=tk.LEFT)
        
#         ttk.Label(pipeline_frame, text="2. Normalize + Cube (always on)", 
#                  font=('', 8), foreground='gray').pack(anchor=tk.W, pady=2, padx=20)
        
#         ttk.Checkbutton(
#             pipeline_frame,
#             text="3. Rolling Ball Background",
#             variable=self.enable_rolling_ball,
#             command=self.reprocess_on_change
#         ).pack(anchor=tk.W, pady=2)
        
#         ttk.Checkbutton(
#             pipeline_frame,
#             text="4. CLAHE Enhancement",
#             variable=self.enable_clahe,
#             command=self.reprocess_on_change
#         ).pack(anchor=tk.W, pady=2)
        
#         # ===== RIDGE DETECTION METHOD =====
#         ridge_frame = ttk.LabelFrame(self.control_panel, text="5. Ridge Detection", padding=5)
#         ridge_frame.pack(fill=tk.X, pady=5)
        
#         methods = [
#             ('Hessian', 'hessian'),
#             ('Frangi', 'frangi'),
#             ('Threshold', 'threshold'),
#             ('Edge (Sobel)', 'edge'),
#             ('Adaptive', 'adaptive')
#         ]
        
#         for label, value in methods:
#             ttk.Radiobutton(
#                 ridge_frame,
#                 text=label,
#                 variable=self.ridge_method,
#                 value=value,
#                 command=self.reprocess_on_change
#             ).pack(anchor=tk.W, pady=1)
        
#         # Ridge threshold
#         thresh_frame = ttk.Frame(ridge_frame)
#         thresh_frame.pack(fill=tk.X, pady=5)
#         ttk.Label(thresh_frame, text="Threshold:", font=('', 8)).pack(side=tk.LEFT)
        
        
#         ridge_slider = ttk.Scale(
#             thresh_frame,
#             from_=0.05,
#             to=0.5,
#             orient=tk.HORIZONTAL,
#             variable=self.ridge_threshold,
#             length=100
#         )
#         ridge_slider.pack(side=tk.LEFT, padx=5)
#         ridge_slider.bind('<ButtonRelease-1>', lambda e: self.reprocess_on_change())  # ADD THIS


#         ttk.Label(thresh_frame, textvariable=self.ridge_threshold, 
#                  font=('', 7), width=4).pack(side=tk.LEFT)
        
#         # Hessian sigma range (only visible when hessian selected)
#         self.hessian_params_frame = ttk.Frame(ridge_frame)
#         self.hessian_params_frame.pack(fill=tk.X, pady=2)
        
#         sigma_frame = ttk.Frame(self.hessian_params_frame)
#         sigma_frame.pack(fill=tk.X)
#         ttk.Label(sigma_frame, text="Sigma range:", font=('', 8)).pack(side=tk.LEFT)
#         ttk.Entry(sigma_frame, textvariable=self.hessian_sigma_min, width=3).pack(side=tk.LEFT, padx=2)
#         ttk.Label(sigma_frame, text="-", font=('', 8)).pack(side=tk.LEFT)
#         ttk.Entry(sigma_frame, textvariable=self.hessian_sigma_max, width=3).pack(side=tk.LEFT, padx=2)
#         ttk.Button(sigma_frame, text="↻", width=2, command=self.reprocess_on_change).pack(side=tk.LEFT, padx=2)
        
#         # Frangi fallback (only for hessian)
#         self.fallback_check = ttk.Checkbutton(
#             ridge_frame,
#             text="6. Frangi Fallback (mid-syllable detection)",
#             variable=self.enable_frangi_fallback,
#             command=self.reprocess_on_change
#         )
#         self.fallback_check.pack(anchor=tk.W, pady=2)
        
#         ttk.Label(ridge_frame, text="7. Component Selection (always on)", 
#                  font=('', 8), foreground='gray').pack(anchor=tk.W, pady=2)
        
#         # ===== COMPONENT SELECTION MODE =====
#         component_frame = ttk.LabelFrame(self.control_panel, text="Component Selection", padding=5)
#         component_frame.pack(fill=tk.X, pady=5)
        
#         ttk.Checkbutton(
#             component_frame,
#             text="D-train mode (aggressive temporal bridging)",
#             variable=self.dtrain_mode,
#             command=self.reprocess_on_change
#         ).pack(anchor=tk.W, pady=2)
        
#         ttk.Label(component_frame, text="For sharp transitions/discontinuities", 
#                  font=('', 7), foreground='gray').pack(anchor=tk.W, padx=20)
        
#         # ===== DISPLAY OPTIONS =====
#         display_frame = ttk.LabelFrame(self.control_panel, text="Display", padding=5)
#         display_frame.pack(fill=tk.X, pady=5)
        
#         ttk.Checkbutton(
#             display_frame,
#             text="Show CLAHE image (grayscale)",
#             variable=self.show_clahe,
#             command=self.update_display
#         ).pack(anchor=tk.W, pady=2)
        
#         ttk.Checkbutton(
#             display_frame,
#             text="Show component mask",
#             variable=self.show_component_mask,
#             command=self.update_display
#         ).pack(anchor=tk.W, pady=2)
        
#         ttk.Checkbutton(
#             display_frame,
#             text="Show energy peak marker",
#             variable=self.show_energy_peak,
#             command=self.update_display
#         ).pack(anchor=tk.W, pady=2)


#         # Grid comparison 
#         ttk.Separator(display_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
#         ttk.Button(
#             display_frame,
#             text="Open Grid Comparison View",
#             command=self.open_grid_comparison
#         ).pack(fill=tk.X, pady=2)


#         # ===== PROCESSING INFO =====
#         self.info_frame = ttk.LabelFrame(self.control_panel, text="Processing Info", padding=5)
#         self.info_frame.pack(fill=tk.X, pady=5)
        
#         self.info_label = ttk.Label(self.info_frame, text="No processing yet", 
#                                     font=('', 8), wraplength=350)
#         self.info_label.pack(fill=tk.X)
        
#         # Update visibility based on initial method
#         self.update_method_visibility()
    
#     def update_method_visibility(self):
#         """Show/hide controls based on selected ridge method"""
#         method = self.ridge_method.get()
        
#         if method == 'hessian':
#             self.hessian_params_frame.pack(fill=tk.X, pady=2)
#             self.fallback_check.pack(anchor=tk.W, pady=2)
#         else:
#             self.hessian_params_frame.pack_forget()
#             self.fallback_check.pack_forget()
    
#     def reprocess_on_change(self):
#         """Reprocess audio when parameters change"""
#         self.update_method_visibility()
#         if self.y is not None:
#             self.process_audio()
#             self.update_display(recompute_spec=False)

#             # Update grid if window is open
#             if self.grid_window is not None and self.grid_window.winfo_exists():
#                 self.refresh_grid()
    
#     def process_audio(self):
#         """Run contour extraction with current parameters"""
#         if self.y is None:
#             return
        
#         print("Processing contours...")
        
#         # Build parameters from UI state
#         params = {
#             # Spectrogram
#             'fmin': self.fmin_calc.get(),
#             'fmax': self.fmax_calc.get(),
#             'n_fft': self.n_fft.get(),
#             'hop_length': self.hop_length.get(),
            
#             # Impulse denoising
#             'denoise_impulses': self.enable_denoise.get(),
#             'impulse_freq_threshold': self.impulse_freq_thresh.get(),
#             'impulse_intensity_threshold_db': 10.0,
#             'impulse_max_gap': 3,
#             'impulse_expand': 2,
            
#             # Rolling ball
#             'low_freq_boundary_hz': 1000,
#             'mid_freq_boundary_hz': 3000,
#             'kernel_low_freq': (7, 20) if self.enable_rolling_ball.get() else (1, 1),
#             'kernel_low_flatness': 0.1,
#             'kernel_mid_freq': (5, 20) if self.enable_rolling_ball.get() else (1, 1),
#             'kernel_mid_flatness': 0.1,
#             'kernel_high_freq': (3, 20) if self.enable_rolling_ball.get() else (1, 1),
#             'kernel_high_flatness': 0.1,
            
#             # CLAHE
#             'clahe_clip_limit': 2.0 if self.enable_clahe.get() else 0.0,
#             'clahe_tile_grid_size': (8, 8),
            
#             # Ridge detection
#             'ridge_filter': self.ridge_method.get(),
#             'ridge_threshold': self.ridge_threshold.get(),
#             'hessian_sigmas': (self.hessian_sigma_min.get(), self.hessian_sigma_max.get()),
#             'frangi_sigmas': (1, 5),
#             'frangi_scale_step': 0.5,
            
#             # Fallback
#             'use_frangi_fallback': self.enable_frangi_fallback.get() and self.ridge_method.get() == 'hessian',
#             'fallback_edge_threshold': 100,
            
#             # Component selection (d-train mode)
#             'time_dist_divisor': 20.0 if self.dtrain_mode.get() else 50.0,
#             'freq_dist_divisor': 100.0,
#             'size_score_divisor': 5.0,
#             'energy_score_divisor': 10.0,
#             'time_weight': 10.0 if self.dtrain_mode.get() else 5.0,
#             'freq_weight': 1.0,
#             'size_weight': 1.0,
#             'energy_weight': 1.0,
#         }
        
#         # Process with current audio file path
#         audio_path = str(self.audio_files[self.current_file_idx])
#         self.result = process_audio_file(audio_path, y=self.y, sr=self.sr, **params)
        
#         # Update info display
#         info_text = (
#             f"Ridge filter: {self.result['ridge_filter']}\n"
#             f"Components detected: {self.result['num_components']}\n"
#             f"Selected component pixels: {self.result['n_pixels']}\n"
#         )
        
#         if self.result.get('used_fallback'):
#             info_text += "⚠ Used Frangi fallback\n"
        
#         self.info_label.config(text=info_text)
        
#         print(f"✓ Detected {self.result['num_components']} components, "
#               f"selected {self.result['n_pixels']} pixels")
    
#     def draw_custom_overlays(self):
#         """Draw contour extraction results"""
#         if self.result is None:
#             return
        
#         # Show CLAHE image if requested
#         if self.show_clahe.get() and 'clahe' in self.result:
#             clahe_img = self.result['clahe']
#             extent = [
#                 self.times[0],
#                 self.times[-1],
#                 self.freqs[0],
#                 self.freqs[-1]
#             ]
#             self.ax.imshow(
#                 clahe_img,
#                 aspect='auto',
#                 origin='lower',
#                 extent=extent,
#                 cmap='gray',
#                 alpha=0.5,
#                 interpolation='bilinear'
#             )
        
#         # Show component mask
#         if self.show_component_mask.get() and 'component' in self.result:
#             component = self.result['component']
#             if component.sum() > 0:
#                 extent = [
#                     self.times[0],
#                     self.times[-1],
#                     self.freqs[0],
#                     self.freqs[-1]
#                 ]
#                 self.ax.imshow(
#                     component,
#                     aspect='auto',
#                     origin='lower',
#                     extent=extent,
#                     cmap='hot',
#                     alpha=0.7,
#                     interpolation='nearest'
#                 )
        
#         # Show energy peak marker
#         if self.show_energy_peak.get() and 'max_energy_idx' in self.result:
#             freq_idx, time_idx = self.result['max_energy_idx']
#             time_s = self.times[time_idx]
#             freq_hz = self.freqs[freq_idx]
            
#             # Draw crosshair
#             self.ax.plot([time_s - 0.01, time_s + 0.01], 
#                         [freq_hz, freq_hz], color='lime', lw=2)
#             self.ax.plot([time_s, time_s], 
#                         [freq_hz - 100, freq_hz + 100], color='lime', lw=2)
    
#     def save_custom_data(self):
#         """Save contour annotations"""
#         if not self.annotation_dir or self.result is None:
#             return
        
#         filename = self.audio_files[self.current_file_idx].stem
#         save_path = self.annotation_dir / f"{filename}_contours.json"
        
#         # Save processing parameters and results
#         data = {
#             'filename': self.audio_files[self.current_file_idx].name,
#             'parameters': {
#                 'denoise_enabled': self.enable_denoise.get(),
#                 'impulse_freq_threshold': self.impulse_freq_thresh.get(),
#                 'rolling_ball_enabled': self.enable_rolling_ball.get(),
#                 'clahe_enabled': self.enable_clahe.get(),
#                 'ridge_method': self.ridge_method.get(),
#                 'ridge_threshold': self.ridge_threshold.get(),
#                 'hessian_sigmas': (self.hessian_sigma_min.get(), self.hessian_sigma_max.get()),
#                 'frangi_fallback_enabled': self.enable_frangi_fallback.get(),
#                 'dtrain_mode': self.dtrain_mode.get(),
#             },
#             'results': {
#                 'n_pixels': int(self.result['n_pixels']),
#                 'num_components': int(self.result['num_components']),
#                 'used_fallback': bool(self.result.get('used_fallback', False)),
#                 'ridge_filter_used': self.result['ridge_filter'],
#                 'max_energy_idx': [int(x) for x in self.result['max_energy_idx']],
#             }
#         }
        
#         with open(save_path, 'w') as f:
#             json.dump(data, f, indent=2)
        
#         self.changes_made = False
#         print(f"✓ Saved contours to {save_path}")
    
#     def load_custom_data(self):
#         """Load contour annotations"""
#         if not self.annotation_dir:
#             return
        
#         filename = self.audio_files[self.current_file_idx].stem
#         load_path = self.annotation_dir / f"{filename}_contours.json"
        
#         if not load_path.exists():
#             return
        
#         with open(load_path, 'r') as f:
#             data = json.load(f)
        
#         # Restore parameters
#         params = data.get('parameters', {})
#         self.enable_denoise.set(params.get('denoise_enabled', True))
#         self.impulse_freq_thresh.set(params.get('impulse_freq_threshold', 0.4))
#         self.enable_rolling_ball.set(params.get('rolling_ball_enabled', True))
#         self.enable_clahe.set(params.get('clahe_enabled', True))
#         self.ridge_method.set(params.get('ridge_method', 'hessian'))
#         self.ridge_threshold.set(params.get('ridge_threshold', 0.3))
        
#         if 'hessian_sigmas' in params:
#             self.hessian_sigma_min.set(params['hessian_sigmas'][0])
#             self.hessian_sigma_max.set(params['hessian_sigmas'][1])
        
#         self.enable_frangi_fallback.set(params.get('frangi_fallback_enabled', False))
#         self.dtrain_mode.set(params.get('dtrain_mode', False))
        
#         print(f"✓ Loaded contour parameters from {load_path}")

#     def open_grid_comparison(self):
#         """Open/refresh a grid comparison window"""
#         if self.y is None:
#             return
        
#         # If window doesn't exist, create it
#         if self.grid_window is None or not self.grid_window.winfo_exists():
#             self.grid_window = tk.Toplevel(self.root)
#             self.grid_window.title("Contour Method Comparison Grid - Live Update")
#             self.grid_window.geometry("1600x1200")
            
#             # Create matplotlib figure
#             from matplotlib.figure import Figure
#             from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
#             self.grid_fig = Figure(figsize=(16, 12))
#             self.grid_fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.2)
            
#             self.grid_canvas = FigureCanvasTkAgg(self.grid_fig, master=self.grid_window)
#             self.grid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
#             # Add refresh button
#             btn_frame = ttk.Frame(self.grid_window)
#             btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
#             ttk.Button(btn_frame, text="Refresh Grid", command=self.refresh_grid).pack()
        
#         # Generate/refresh the grid
#         self.refresh_grid()
    
#     def refresh_grid(self):
#         """Refresh the grid comparison with current audio"""
#         if self.grid_fig is None or self.y is None:
#             return
        
#         self.grid_fig.clear()
        
#         # Define 16 different configurations to compare
#         configs = [
#             # Row 1: Hessian variations
#             ("Hessian\nNo Denoise", {'denoise_impulses': False, 'ridge_filter': 'hessian', 'ridge_threshold': 0.3}),
#             ("Hessian\n+Denoise", {'denoise_impulses': True, 'ridge_filter': 'hessian', 'ridge_threshold': 0.3}),
#             ("Hessian\nthresh=0.2", {'denoise_impulses': True, 'ridge_filter': 'hessian', 'ridge_threshold': 0.2}),
#             ("Hessian\n+Fallback", {'denoise_impulses': True, 'ridge_filter': 'hessian', 'ridge_threshold': 0.3, 'use_frangi_fallback': True}),
            
#             # Row 2: Frangi variations
#             ("Frangi\nthresh=0.1", {'denoise_impulses': True, 'ridge_filter': 'frangi', 'ridge_threshold': 0.1}),
#             ("Frangi\nthresh=0.15", {'denoise_impulses': True, 'ridge_filter': 'frangi', 'ridge_threshold': 0.15}),
#             ("Frangi\nthresh=0.2", {'denoise_impulses': True, 'ridge_filter': 'frangi', 'ridge_threshold': 0.2}),
#             ("Hessian\nσ:1-10", {'denoise_impulses': True, 'ridge_filter': 'hessian', 'ridge_threshold': 0.2, 'hessian_sigmas': (1, 10)}),
            
#             # Row 3: Other methods
#             ("Threshold\n0.3", {'denoise_impulses': True, 'ridge_filter': 'threshold', 'ridge_threshold': 0.3}),
#             ("Edge\n0.3", {'denoise_impulses': True, 'ridge_filter': 'edge', 'ridge_threshold': 0.3}),
#             ("Adaptive\n0.3", {'denoise_impulses': True, 'ridge_filter': 'adaptive', 'ridge_threshold': 0.3}),
#             ("No Rolling\nBall", {'denoise_impulses': True, 'ridge_filter': 'hessian', 'ridge_threshold': 0.3, 'kernel_low_freq': (1,1), 'kernel_mid_freq': (1,1), 'kernel_high_freq': (1,1)}),
            
#             # Row 4: D-train and variations
#             ("D-train\nMode", {'denoise_impulses': True, 'ridge_filter': 'hessian', 'ridge_threshold': 0.3, 'time_dist_divisor': 20.0, 'time_weight': 10.0}),
#             ("No CLAHE", {'denoise_impulses': True, 'ridge_filter': 'hessian', 'ridge_threshold': 0.3, 'clahe_clip_limit': 0.0}),
#             ("Impulse\nthresh=0.3", {'denoise_impulses': True, 'impulse_freq_threshold': 0.3, 'ridge_filter': 'hessian', 'ridge_threshold': 0.3}),
#             ("Current\nSettings", {}),  # Use current UI settings
#         ]
        
#         # Process each configuration
#         print("Refreshing grid comparison...")
#         audio_path = str(self.audio_files[self.current_file_idx])
        
#         for idx, (label, config_overrides) in enumerate(configs):
#             ax = self.grid_fig.add_subplot(4, 4, idx + 1)
            
#             # Build parameters
#             if label == "Current\nSettings":
#                 params = self._get_current_params()
#             else:
#                 params = {
#                     'fmin': self.fmin_calc.get(),
#                     'fmax': self.fmax_calc.get(),
#                     'n_fft': self.n_fft.get(),
#                     'hop_length': self.hop_length.get(),
#                     'denoise_impulses': True,
#                     'impulse_freq_threshold': 0.4,
#                     'impulse_intensity_threshold_db': 10.0,
#                     'impulse_max_gap': 3,
#                     'impulse_expand': 2,
#                     'low_freq_boundary_hz': 1000,
#                     'mid_freq_boundary_hz': 3000,
#                     'kernel_low_freq': (7, 20),
#                     'kernel_low_flatness': 0.1,
#                     'kernel_mid_freq': (5, 20),
#                     'kernel_mid_flatness': 0.1,
#                     'kernel_high_freq': (3, 20),
#                     'kernel_high_flatness': 0.1,
#                     'clahe_clip_limit': 2.0,
#                     'clahe_tile_grid_size': (8, 8),
#                     'ridge_filter': 'hessian',
#                     'ridge_threshold': 0.3,
#                     'hessian_sigmas': (1, 5),
#                     'frangi_sigmas': (1, 5),
#                     'frangi_scale_step': 0.5,
#                     'use_frangi_fallback': False,
#                     'fallback_edge_threshold': 100,
#                     'time_dist_divisor': 50.0,
#                     'freq_dist_divisor': 100.0,
#                     'size_score_divisor': 5.0,
#                     'energy_score_divisor': 10.0,
#                     'time_weight': 5.0,
#                     'freq_weight': 1.0,
#                     'size_weight': 1.0,
#                     'energy_weight': 1.0,
#                 }
#                 params.update(config_overrides)
            
#             # Process with pre-loaded audio
#             result = process_audio_file(audio_path, y=self.y, sr=self.sr, **params)
            
#             # Plot
#             extent = [self.times[0], self.times[-1], self.freqs[0], self.freqs[-1]]
#             ax.imshow(self.S_db, aspect='auto', origin='lower', extent=extent, 
#                      cmap='magma', alpha=0.5, interpolation='bilinear')
            
#             if result['component'].sum() > 0:
#                 ax.imshow(result['component'], aspect='auto', origin='lower', extent=extent,
#                          cmap='hot', alpha=0.7, interpolation='nearest')
            
#             freq_idx, time_idx = result['max_energy_idx']
#             time_s = self.times[time_idx]
#             freq_hz = self.freqs[freq_idx]
#             ax.plot([time_s - 0.01, time_s + 0.01], [freq_hz, freq_hz], color='lime', lw=1.5)
#             ax.plot([time_s, time_s], [freq_hz - 100, freq_hz + 100], color='lime', lw=1.5)
            
#             fallback_marker = "*" if result.get('used_fallback') else ""
#             ax.set_title(f"{label}\n{result['n_pixels']}px{fallback_marker}", fontsize=8)
            
#             ymin, ymax = self._convert_ylim_to_scale(self.fmin_display.get(), self.fmax_display.get())
#             ax.set_ylim(ymin, ymax)
#             ax.tick_params(labelsize=6)
#             ax.set_xlabel('Time (s)', fontsize=7)
#             ax.set_ylabel('Freq (Hz)', fontsize=7)
        
#         self.grid_canvas.draw()
#         print("✓ Grid refreshed")


#     def _get_current_params(self):
#         """Get current parameters from UI"""
#         return {
#             'fmin': self.fmin_calc.get(),
#             'fmax': self.fmax_calc.get(),
#             'n_fft': self.n_fft.get(),
#             'hop_length': self.hop_length.get(),
#             'denoise_impulses': self.enable_denoise.get(),
#             'impulse_freq_threshold': self.impulse_freq_thresh.get(),
#             'impulse_intensity_threshold_db': 10.0,
#             'impulse_max_gap': 3,
#             'impulse_expand': 2,
#             'low_freq_boundary_hz': 1000,
#             'mid_freq_boundary_hz': 3000,
#             'kernel_low_freq': (7, 20) if self.enable_rolling_ball.get() else (1, 1),
#             'kernel_low_flatness': 0.1,
#             'kernel_mid_freq': (5, 20) if self.enable_rolling_ball.get() else (1, 1),
#             'kernel_mid_flatness': 0.1,
#             'kernel_high_freq': (3, 20) if self.enable_rolling_ball.get() else (1, 1),
#             'kernel_high_flatness': 0.1,
#             'clahe_clip_limit': 2.0 if self.enable_clahe.get() else 0.0,
#             'clahe_tile_grid_size': (8, 8),
#             'ridge_filter': self.ridge_method.get(),
#             'ridge_threshold': self.ridge_threshold.get(),
#             'hessian_sigmas': (self.hessian_sigma_min.get(), self.hessian_sigma_max.get()),
#             'frangi_sigmas': (1, 5),
#             'frangi_scale_step': 0.5,
#             'use_frangi_fallback': self.enable_frangi_fallback.get() and self.ridge_method.get() == 'hessian',
#             'fallback_edge_threshold': 100,
#             'time_dist_divisor': 20.0 if self.dtrain_mode.get() else 50.0,
#             'freq_dist_divisor': 100.0,
#             'size_score_divisor': 5.0,
#             'energy_score_divisor': 10.0,
#             'time_weight': 10.0 if self.dtrain_mode.get() else 5.0,
#             'freq_weight': 1.0,
#             'size_weight': 1.0,
#             'energy_weight': 1.0,
#             }


# def main():
#     """Entry point for contour layer"""
#     root = tk.Tk()
#     app = ContourLayer(root)
#     root.geometry("1600x900")
#     root.mainloop()

# if __name__ == "__main__":
#     main()
