#!/usr/bin/env python3
"""
Search functionality for CLIP Image Search.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import torch
import numpy as np
from PIL import Image
from utils import ToolTip
from utils.model_utils import load_embeddings
import sv_ttk

class SearchTab:
    def __init__(self, app, notebook):
        self.app = app
        
        # Create the search tab
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="Search")
        
        # Initialize search variables
        self.query_image = tk.StringVar()
        self.query_text = tk.StringVar()
        self.cached_results = []
        self.result_paths = []
        self.current_page = 0
        self.results_per_page = 30
        
        # Create a list of suggested models for the dropdown
        self.suggested_models = [
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",  # Default high quality model
            "openai/clip-vit-large-patch14",          # OpenAI's large model
            "openai/clip-vit-base-patch32",           # OpenAI's base model
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",  # Smaller, faster LAION model
            "OFA-Sys/chinese-clip-vit-base-patch16",  # Chinese language model
        ]
        
        # Create the search UI
        self._create_search_ui()
        
        # Bind events
        self.query_text_entry.bind("<Return>", self._on_text_search)
        
    def _create_search_ui(self):
        """Create the search tab UI"""
        # Main container with padding
        main_container = ttk.Frame(self.tab, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Top section (configuration and search controls) - flex layout
        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, pady=(0, 15))
        top_frame.columnconfigure(0, weight=1)  # Settings panel
        top_frame.columnconfigure(1, weight=1)  # Search panel

        # Left side: Settings
        self._create_settings_panel(top_frame)
        
        # Right side: Search controls
        self._create_search_panel(top_frame)
        
        # Results section
        self._create_results_panel(main_container)
        
        # Pagination frame
        self._create_pagination(main_container)

    def _create_settings_panel(self, parent):
        """Create settings panel"""
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="10")
        settings_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 5))

        # Two-column grid layout with better spacing
        settings_frame.columnconfigure(1, weight=1)
        
        # Embeddings file row
        ttk.Label(settings_frame, text="Embeddings:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=8)
        ttk.Entry(settings_frame, textvariable=self.app.embeddings_file, width=30).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=8)
        browse_button = ttk.Button(settings_frame, text="Browse...", command=self.app._browse_embeddings)
        browse_button.grid(row=0, column=2, padx=5, pady=8)

        # Model selection row
        ttk.Label(settings_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=8)
        
        # Create a combobox for model selection
        model_combo = ttk.Combobox(settings_frame, textvariable=self.app.model_name, width=30)
        model_combo.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=8)
        
        # Configure the combobox
        model_combo['values'] = self.suggested_models
        model_combo.configure(state="normal")  # Allow custom entries
        
        # Create button frame for model controls
        model_buttons = ttk.Frame(settings_frame)
        model_buttons.grid(row=1, column=2, padx=5, pady=8)
        
        # Add an info button for model description
        info_button = ttk.Button(model_buttons, text="ℹ", width=2, style="Info.TButton")
        info_button.pack(side=tk.LEFT, padx=(0, 3))
        
        # Create tooltip for the info button
        self.model_tooltip = ToolTip(info_button, "Select a model to see its description", delay=100, wraplength=350)
        
        # Configure the style for info buttons
        style = ttk.Style()
        style.configure("Info.TButton", font=("", 10, "bold"))
        
        ttk.Button(model_buttons, text="Load", command=self.app._load_model).pack(side=tk.LEFT)
        
        # Update tooltip when model changes
        def update_model_tooltip(*args):
            model = self.app.model_name.get()
            if model in self.app.model_descriptions:
                self.model_tooltip.update_text(self.app.model_descriptions[model])
            else:
                self.model_tooltip.update_text("Custom model")
        
        self.app.model_name.trace_add("write", update_model_tooltip)
        
        # Call once to set initial tooltip
        update_model_tooltip()

        # Auto-load options row
        auto_load_frame = ttk.Frame(settings_frame)
        auto_load_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=8)
        
        ttk.Label(auto_load_frame, text="Auto-load on Startup:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Create checkboxes for auto-load options
        ttk.Checkbutton(auto_load_frame, text="Embeddings", 
                       variable=self.app.auto_load_embeddings,
                       command=self.app._save_config).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Checkbutton(auto_load_frame, text="Model", 
                       variable=self.app.auto_load_model,
                       command=self.app._save_config).pack(side=tk.LEFT)

        # Action button row - span all columns
        action_frame = ttk.Frame(settings_frame)
        action_frame.grid(row=3, column=0, columnspan=3, pady=8, sticky=tk.EW)
        
        # Load embeddings button - now starts a background thread
        load_btn = ttk.Button(action_frame, text="Load Embeddings", 
                             command=self._start_load_embeddings)
        load_btn.pack(fill=tk.X)

    def _start_load_embeddings(self):
        """Start loading embeddings in a background thread"""
        if not os.path.isfile(self.app.embeddings_file.get()):
            messagebox.showerror("Error", "Please select a valid embeddings file")
            return

        # Disable the load button while loading
        for widget in self.tab.winfo_children():
            if isinstance(widget, ttk.Button) and widget['text'] == "Load Embeddings":
                widget.configure(state="disabled")

        # Start loading in background thread
        threading.Thread(target=self._load_embeddings_thread, daemon=True).start()

    def _load_embeddings_thread(self):
        """Background thread for loading embeddings"""
        try:
            self.app.status_text.set("Loading embeddings...")
            self.app.progress_var.set(10)
            self.app.embeddings = load_embeddings(self.app.embeddings_file.get())
            self.app.progress_var.set(100)
            self.app.status_text.set(f"Loaded {len(self.app.embeddings)} embeddings")
        except Exception as e:
            error_msg = str(e)
            self.app.status_text.set(f"Error loading embeddings: {error_msg}")
            # Show error dialog in main thread, passing the error message directly
            self.app.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to load embeddings: {msg}"))
        finally:
            # Re-enable the load button in main thread
            self.app.root.after(0, self._enable_load_button)

    def _enable_load_button(self):
        """Re-enable the load embeddings button"""
        for widget in self.tab.winfo_children():
            if isinstance(widget, ttk.Button) and widget['text'] == "Load Embeddings":
                widget.configure(state="normal")

    def _create_search_panel(self, parent):
        """Create search controls panel"""
        search_frame = ttk.LabelFrame(parent, text="Search", padding="10")
        search_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(5, 0))

        # Configure grid
        search_frame.columnconfigure(1, weight=1)
        
        # Text search row
        ttk.Label(search_frame, text="Text Query:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=8)
        self.query_text_entry = ttk.Entry(search_frame, textvariable=self.query_text, width=30)
        self.query_text_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=8)
        ttk.Button(search_frame, text="Search", command=self._on_text_search).grid(row=0, column=2, padx=5, pady=8)

        # Image search row
        ttk.Label(search_frame, text="Image Query:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=8)
        ttk.Entry(search_frame, textvariable=self.query_image, width=30).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=8)
        ttk.Button(search_frame, text="Browse...", command=self._browse_query_image).grid(row=1, column=2, padx=5, pady=8)

        # Threshold controls row
        ttk.Label(search_frame, text="Score Threshold:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=8)
        threshold_frame = ttk.Frame(search_frame)
        threshold_frame.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=8)
        
        ttk.Checkbutton(threshold_frame, text="Auto", variable=self.app.auto_threshold).pack(side=tk.LEFT)
        self.threshold_spinbox = ttk.Spinbox(threshold_frame, from_=0.0, to=1.0, increment=0.05,
                                           textvariable=self.app.manual_threshold, width=5)
        self.threshold_spinbox.pack(side=tk.LEFT, padx=10)
        
        # Action button row
        action_frame = ttk.Frame(search_frame)
        action_frame.grid(row=3, column=0, columnspan=3, pady=8, sticky=tk.EW)
        
        # Clear results button
        ttk.Button(action_frame, text="Clear Results", command=self._clear_results).pack(fill=tk.X)

    def _create_results_panel(self, parent):
        """Create the results display panel"""
        results_frame = ttk.Frame(parent)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create canvas with vertical scrollbar only
        self.canvas = tk.Canvas(results_frame, bd=0, highlightthickness=0)
        
        # Set the canvas background based on current theme
        theme = sv_ttk.get_theme()
        bg_color = "#1c1c1c" if theme == "dark" else "#f0f0f0"
        self.canvas.configure(bg=bg_color)
        
        self.scrollbar_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set)
        
        # Pack scrollbar and canvas
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for results
        self.results_container = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.results_container, anchor=tk.NW)
        
        # Bind events for scrolling
        self.results_container.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind to theme changes via a special tag
        self.canvas.bindtags((self.canvas, "ThemeAwareCanvas", "Canvas", "all"))
        
        # Bind window resize event
        self.app.root.bind("<Configure>", self._on_window_resize)

    def _create_pagination(self, parent):
        """Create pagination controls"""
        pagination_frame = ttk.Frame(parent)
        pagination_frame.pack(fill=tk.X, pady=(10, 0))

        # Center the pagination controls
        spacer1 = ttk.Frame(pagination_frame)
        spacer1.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        controls_frame = ttk.Frame(pagination_frame)
        controls_frame.pack(side=tk.LEFT)
        
        ttk.Button(controls_frame, text="Previous", command=self._prev_page).pack(side=tk.LEFT, padx=5)
        self.page_label = ttk.Label(controls_frame, text="Page 1")
        self.page_label.pack(side=tk.LEFT, padx=15)
        ttk.Button(controls_frame, text="Next", command=self._next_page).pack(side=tk.LEFT, padx=5)
        
        spacer2 = ttk.Frame(pagination_frame)
        spacer2.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _browse_query_image(self):
        """Browse for an image to use as search query"""
        filename = filedialog.askopenfilename(
            title="Select Query Image",
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"),
                ("All files", "*.*")
            )
        )
        if filename:
            self.query_image.set(filename)
            self._image_search()

    def _on_text_search(self, event=None):
        """Handle text search event"""
        self._text_search()
        
    def _text_search(self):
        """Perform text-based search"""
        if not self.query_text.get().strip():
            messagebox.showinfo("Info", "Please enter a text query")
            return

        if not self.app.model or not self.app.tokenizer:
            messagebox.showinfo("Info", "Please load a model first")
            return

        if not self.app.embeddings:
            messagebox.showinfo("Info", "Please load embeddings first")
            return

        def search_task():
            try:
                query = self.query_text.get().strip()
                self.app.status_text.set(f"Searching for '{query}'...")
                self.app.progress_var.set(10)

                # Generate text embedding
                inputs = self.app.tokenizer([query], padding=True, return_tensors="pt")

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    text_features = self.app.model.get_text_features(**inputs)
                    text_embedding = text_features.cpu().numpy()[0]
                    text_embedding = text_embedding / np.linalg.norm(text_embedding)

                self.app.progress_var.set(50)

                # Search for similar images
                similarities = []
                for path, data in self.app.embeddings.items():
                    if 'embedding' not in data:
                        continue

                    image_embedding = np.array(data['embedding'])
                    image_embedding = image_embedding / np.linalg.norm(image_embedding)
                    similarity = np.dot(text_embedding, image_embedding)
                    similarities.append((path, similarity))

                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)

                self.app.progress_var.set(90)

                # Store all results in cache
                self.cached_results = similarities
                
                # Update display with current page and threshold
                self._update_displayed_results()
                
                self.app.progress_var.set(100)
                
            except Exception as e:
                self.app.status_text.set(f"Error during search: {str(e)}")
                messagebox.showerror("Error", f"Search failed: {str(e)}")

        threading.Thread(target=search_task, daemon=True).start()

    def _image_search(self):
        """Perform image-based search"""
        if not os.path.isfile(self.query_image.get()):
            messagebox.showerror("Error", "Please select a valid image file")
            return

        if not self.app.model or not self.app.processor:
            messagebox.showinfo("Info", "Please load a model first")
            return

        if not self.app.embeddings:
            messagebox.showinfo("Info", "Please load embeddings first")
            return

        def search_task():
            try:
                query_path = self.query_image.get()
                self.app.status_text.set(f"Searching for similar images to {os.path.basename(query_path)}...")
                self.app.progress_var.set(10)

                # Load and process query image
                query_image = Image.open(query_path).convert('RGB')
                inputs = self.app.processor(images=query_image, return_tensors="pt")

                # Move to GPU if available
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items() if k != 'text'}

                with torch.no_grad():
                    image_features = self.app.model.get_image_features(**inputs)
                    query_embedding = image_features.cpu().numpy()[0]
                    # Normalize
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)

                self.app.progress_var.set(50)

                # Search for similar images
                similarities = []
                for path, data in self.app.embeddings.items():
                    if 'embedding' not in data:
                        continue

                    image_embedding = np.array(data['embedding'])
                    # In case the stored embedding is not normalized
                    image_embedding = image_embedding / np.linalg.norm(image_embedding)

                    similarity = np.dot(query_embedding, image_embedding)
                    similarities.append((path, similarity))

                # Sort by similarity (highest first)
                similarities.sort(key=lambda x: x[1], reverse=True)

                self.app.progress_var.set(90)

                # Store all results in cache
                self.cached_results = similarities
                
                # Update display with threshold
                self._update_displayed_results()
                
                self.app.progress_var.set(100)
            except Exception as e:
                self.app.status_text.set(f"Error during search: {str(e)}")
                messagebox.showerror("Error", f"Search failed: {str(e)}")

        threading.Thread(target=search_task, daemon=True).start()

    def _update_displayed_results(self):
        """Update the result_paths based on threshold and sorting"""
        if not self.cached_results:
            return
        
        # Get threshold value
        if self.app.auto_threshold.get():
            # Calculate dynamic threshold based on score distribution
            scores = [score for _, score in self.cached_results]
            if scores:
                # Use aggressive thresholding
                mean_score = sum(scores) / len(scores)
                std_score = np.std(scores)
                
                # Options for stricter thresholding:
                # 1. Use mean + percentage of range
                score_range = max(scores) - min(scores)
                threshold_from_range = mean_score + (score_range * 0.2)
                
                # 2. Use higher percentile
                percentile_95 = np.percentile(scores, 95)
                
                # 3. Use mean + std with larger multiplier
                threshold_from_std = mean_score + (1.0 * std_score)
                
                # Take the highest value of our three approaches
                self.app.min_score = max(threshold_from_range, percentile_95, threshold_from_std)
                
                # Ensure minimum threshold
                self.app.min_score = max(0.3, self.app.min_score)
                
                # Update manual threshold for display
                self.app.manual_threshold.set(round(self.app.min_score, 2))
        else:
            self.app.min_score = self.app.manual_threshold.get()
        
        # Filter results by threshold
        filtered_results = [(path, score) for path, score in self.cached_results 
                           if score >= self.app.min_score]
        
        # Update status with threshold info
        result_count = len(filtered_results)
        total_count = len(self.cached_results)
        threshold_pct = self.app.min_score * 100
        
        if result_count == 0:
            self.app.status_text.set(f"No results above {threshold_pct:.1f}% similarity")
        elif result_count < total_count:
            percentage = (result_count / total_count) * 100
            query = self.query_text.get().strip() or os.path.basename(self.query_image.get())
            self.app.status_text.set(
                f"Found {result_count} matches for '{query}' with {threshold_pct:.1f}%+ similarity ({percentage:.1f}% of dataset)")
        else:
            self.app.status_text.set(f"All {result_count} results above {threshold_pct:.1f}% similarity")
        
        # Store all filtered results
        self.result_paths = filtered_results
        
        # Reset to first page
        self.current_page = 0
        self._update_results_page()

    def _update_results_page(self):
        """Update the displayed results for current page"""
        # Get the thumbnail manager to handle this
        self.app.thumbnail_manager.update_results_page(
            self.results_container, 
            self.canvas,
            self.result_paths,
            self.current_page,
            self.results_per_page
        )
        
        total_pages = (len(self.result_paths) + self.results_per_page - 1) // self.results_per_page
        self.page_label.config(text=f"Page {self.current_page + 1} of {total_pages}")

    def _clear_results(self):
        """Clear all search results"""
        self.result_paths = []
        self.cached_results = []
        self.current_page = 0

        # Clear display
        for widget in self.results_container.winfo_children():
            widget.destroy()

        self.page_label.config(text="Page 1")
        self.app.status_text.set("Results cleared")
        
        # Clean up thumbnail cache
        self.app.thumbnail_manager.cleanup_cache()

    def _prev_page(self):
        """Go to previous page of results"""
        if self.current_page > 0:
            self.current_page -= 1
            self._update_results_page()

    def _next_page(self):
        """Go to next page of results"""
        max_page = (len(self.result_paths) + self.results_per_page - 1) // self.results_per_page - 1
        if self.current_page < max_page:
            self.current_page += 1
            self._update_results_page()

    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """When canvas is resized, resize the inner frame to match"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
    
    def _on_window_resize(self, event):
        """Handle window resize event"""
        # Only respond to root window resizing, not child widgets
        if event.widget == self.app.root:
            # Ignore resize events during thumbnail loading
            if self.app.thumbnail_manager.is_loading_thumbnails:
                return
            
            # Ignore small resize changes
            if hasattr(self, '_last_width'):
                width_change = abs(event.width - self._last_width)
                if width_change < 50:  # Ignore small width changes
                    return
                
            self._last_width = event.width
            
            # Cancel any pending resize timer
            if hasattr(self, "_resize_timer"):
                self.app.root.after_cancel(self._resize_timer)
            
            # Schedule a single resize update
            self._resize_timer = self.app.root.after(250, self._delayed_resize_update)

    def _delayed_resize_update(self):
        """Handle the actual resize update after delay"""
        if self.result_paths:
            self._update_results_page() 

    def update_canvas_theme(self, theme):
        """Update canvas background based on theme"""
        if hasattr(self, 'canvas'):
            bg_color = "#1c1c1c" if theme == "dark" else "#f0f0f0"
            self.canvas.configure(bg=bg_color) 