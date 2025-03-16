#!/usr/bin/env python3
"""
Embedding generation functionality for CLIP Image Search.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import utils
from utils.model_utils import process_images_batch
import sv_ttk

class GenerateTab:
    def __init__(self, app, notebook):
        self.app = app
        
        # Create the generate tab
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="Generate Embeddings")
        
        # Initialize variables
        self._initialize_variables()
        
        # Create UI elements
        self._create_ui()
    
    def _initialize_variables(self):
        """Initialize variables for embedding generation"""
        self.gen_directory = tk.StringVar()
        self.gen_output = tk.StringVar()
        self.gen_model = tk.StringVar(value=self.app.model_name.get())
        self.gen_batch_size = tk.IntVar(value=16)
        self.gen_fp16 = tk.BooleanVar(value=False)
        self.gen_progress = tk.DoubleVar(value=0)
        self.gen_status = tk.StringVar(value="Ready")
        self.gen_stop_flag = False
        
        self.suggested_models = [
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",  # Default high quality model
            "openai/clip-vit-large-patch14",          # OpenAI's large model
            "openai/clip-vit-base-patch32",           # OpenAI's base model
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",  # Smaller, faster LAION model
            "OFA-Sys/chinese-clip-vit-base-patch16",  # Chinese language model
        ]
    
    def _create_ui(self):
        """Create the UI for embedding generation tab"""
        # Main frame
        frame = ttk.Frame(self.tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Input settings
        settings_frame = ttk.LabelFrame(frame, text="Generation Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Image directory
        ttk.Label(settings_frame, text="Images Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.gen_directory, width=40).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(settings_frame, text="Browse...", command=self._browse_gen_directory).grid(row=0, column=2, padx=5, pady=5)
        
        # Output file
        ttk.Label(settings_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.gen_output, width=40).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(settings_frame, text="Browse...", command=self._browse_gen_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Use current embeddings button
        ttk.Button(settings_frame, text="Use Current Embeddings File", 
                   command=lambda: self.gen_output.set(self.app.embeddings_file.get())).grid(
            row=1, column=3, padx=5, pady=5)
        
        # Model selection with dropdown
        ttk.Label(settings_frame, text="Model:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Create a combobox for model selection
        model_combo = ttk.Combobox(settings_frame, textvariable=self.gen_model, width=38)
        model_combo.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Configure the combobox
        model_combo['values'] = self.suggested_models
        model_combo.configure(state="normal")  # Allow custom entries
        
        # Add an info button for model description
        info_button = ttk.Button(settings_frame, text="ℹ", width=2, style="Info.TButton")
        info_button.grid(row=2, column=2, padx=(0,5), pady=5)
        
        # Create tooltip for the info button
        self.model_tooltip = utils.ToolTip(info_button, "Select a model to see its description", delay=100, wraplength=350)
        
        # Configure the style for info buttons
        style = ttk.Style()
        current_theme = sv_ttk.get_theme()
        if current_theme == "dark":
            style.configure("Info.TButton", font=("", 10, "bold"))
        else:
            style.configure("Info.TButton", font=("", 10, "bold"))
        
        ttk.Button(settings_frame, text="Use Search Model", command=lambda: self.gen_model.set(self.app.model_name.get())).grid(
            row=2, column=3, padx=5, pady=5)
        
        # Update tooltip when model changes
        def update_model_tooltip(*args):
            model = self.gen_model.get()
            if model in self.app.model_descriptions:
                self.model_tooltip.update_text(self.app.model_descriptions[model])
            else:
                self.model_tooltip.update_text("Custom model")
        
        self.gen_model.trace_add("write", update_model_tooltip)
        
        # Call once to set initial tooltip
        update_model_tooltip()
        
        # Batch size
        ttk.Label(settings_frame, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(settings_frame, from_=1, to=64, textvariable=self.gen_batch_size, width=5).grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # FP16 checkbox
        ttk.Checkbutton(settings_frame, text="Use FP16 (half precision, requires GPU)",
                       variable=self.gen_fp16).grid(row=3, column=2, padx=5, pady=5, sticky=tk.W)
        
        settings_frame.columnconfigure(1, weight=1)
        
        # Action buttons
        actions_frame = ttk.Frame(frame)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(actions_frame, text="Generate Embeddings", command=self._generate_embeddings).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Stop", command=self._stop_generation).pack(side=tk.LEFT, padx=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.gen_progress_bar = ttk.Progressbar(progress_frame, variable=self.gen_progress, length=200, mode="determinate")
        self.gen_progress_bar.pack(fill=tk.X, pady=5)
        
        ttk.Label(progress_frame, textvariable=self.gen_status).pack(anchor=tk.W)
        
        # Log frame
        log_frame = ttk.LabelFrame(frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, height=10, yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
    
    def _browse_gen_directory(self):
        """Browse for a directory containing images to embed"""
        directory = filedialog.askdirectory(title="Select Images Directory")
        if directory:
            self.gen_directory.set(directory)
    
    def _browse_gen_output(self):
        """Browse for output file location"""
        filename = filedialog.asksaveasfilename(
            title="Save Embeddings As",
            filetypes=(
                ("Compressed JSON", "*.json.zst"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ),
            defaultextension=".json.zst"
        )
        if filename:
            self.gen_output.set(filename)
    
    def _log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def _stop_generation(self):
        """Stop the current generation process"""
        self.gen_stop_flag = True
        self.gen_status.set("Stopping...")
        self._log("Stopping generation at next batch...")
    
    def _generate_embeddings(self):
        """Generate embeddings for all images in the directory"""
        # Validate inputs
        if not self.gen_directory.get() or not os.path.isdir(self.gen_directory.get()):
            messagebox.showerror("Error", "Please select a valid directory containing images")
            return
            
        if not self.gen_output.get():
            messagebox.showerror("Error", "Please specify an output file")
            return
            
        if not self.gen_model.get():
            messagebox.showerror("Error", "Please specify a model")
            return
            
        # Update status
        self.gen_progress.set(0)
        self.gen_status.set("Starting...")
        self.gen_stop_flag = False
        self._log(f"Starting embedding generation for {self.gen_directory.get()}")
        self._log(f"Output file: {self.gen_output.get()}")
        self._log(f"Model: {self.gen_model.get()}")
        self._log(f"Batch size: {self.gen_batch_size.get()}")
        self._log(f"FP16: {self.gen_fp16.get()}")
        
        def generation_task():
            try:
                # Update embeddings file path to use .zst extension if not specified
                output_path = self.gen_output.get()
                if not output_path.endswith('.zst') and not output_path.endswith('.json'):
                    output_path = output_path + '.json.zst'
                    self.gen_output.set(output_path)
                
                # Get image paths
                self._log("Finding images in directory...")
                self.gen_status.set("Finding images...")
                image_files, skipped_files = utils.get_image_files(self.gen_directory.get())
                
                # Report skipped files
                if skipped_files:
                    self._log(f"\nSkipped {len(skipped_files)} invalid or non-image files")
                    for path, reason in skipped_files[:10]:  # Show first 10
                        self._log(f"  - {os.path.basename(path)}: {reason}")
                    if len(skipped_files) > 10:
                        self._log(f"  ... and {len(skipped_files) - 10} more")
                
                self._log(f"Found {len(image_files)} valid images")
                
                # Load existing embeddings
                self.gen_progress.set(5)
                self.gen_status.set("Loading existing embeddings...")
                existing_embeddings = {}
                if os.path.exists(output_path):
                    self._log("Loading existing embeddings file...")
                    try:
                        existing_embeddings = utils.load_embeddings(output_path)
                        self._log(f"Loaded {len(existing_embeddings)} existing embeddings")
                    except Exception as e:
                        self._log(f"Error loading existing embeddings: {str(e)}")
                        self._log("Starting with empty embeddings")
                
                # Find images that need processing
                self.gen_progress.set(10)
                to_process = utils.find_images_to_process(image_files, existing_embeddings)
                
                self._log(f"{len(to_process)} images need processing")
                if not to_process:
                    self.gen_status.set("No new images to process")
                    return
                
                # Generate embeddings
                self.gen_status.set(f"Generating embeddings ({len(to_process)} images)...")
                new_count = 0
                
                # Initialize model
                self._log(f"Loading model {self.gen_model.get()}...")
                batch_size = self.gen_batch_size.get()
                
                # Process images in batches
                self._log("Processing images in batches...")
                new_embeddings, failed_paths = process_images_batch(
                    to_process,
                    self.gen_model.get(),
                    batch_size=batch_size,
                    use_fp16=self.gen_fp16.get(),
                    progress_callback=lambda i, total: self._update_progress(i, total, 20, 90),
                    stop_flag=lambda: self.gen_stop_flag
                )
                
                # Report failures
                if failed_paths:
                    self._log(f"\nFailed to process {len(failed_paths)} images")
                    for path, reason in failed_paths[:10]:
                        self._log(f"  - {os.path.basename(path)}: {reason}")
                    if len(failed_paths) > 10:
                        self._log(f"  ... and {len(failed_paths) - 10} more")
                
                # Update progress
                self.gen_progress.set(95)
                
                # Merge with existing embeddings
                self._log("Merging with existing embeddings...")
                new_count = len(new_embeddings)
                existing_embeddings.update(new_embeddings)
                
                # Save embeddings
                self._log(f"Saving {len(existing_embeddings)} embeddings to {output_path}...")
                self.gen_status.set("Saving embeddings...")
                utils.save_embeddings(existing_embeddings, output_path)
                self.gen_progress.set(100)
                self.gen_status.set(f"Complete - {new_count} new embeddings generated")
                self._log(f"Completed processing {len(to_process)} images")
                self._log(f"New/updated embeddings: {new_count}")
                self._log(f"Total embeddings in file: {len(existing_embeddings)}")
                self._log(f"Embeddings saved to: {output_path}")
                
                # Offer to load the embeddings for search
                if messagebox.askyesno("Generation Complete", 
                                    f"Generated {new_count} new embeddings. Load them for search?"):
                    self.app.embeddings_file.set(output_path)
                    self.app.notebook.select(0)  # Switch to search tab
                    self.app._load_embeddings()
            
            except Exception as e:
                self._log(f"Error during embedding generation: {str(e)}")
                self.gen_status.set(f"Error: {str(e)}")
                import traceback
                self._log(traceback.format_exc())
        
        # Start generation thread
        threading.Thread(target=generation_task, daemon=True).start()

    def _update_progress(self, current, total, start_pct, end_pct):
        """Update progress bar with scaled values"""
        if total == 0:
            progress = 0
        else:
            progress = current / total
        
        # Scale to the specified range
        scaled_progress = start_pct + progress * (end_pct - start_pct)
        self.gen_progress.set(scaled_progress)
        
        # Update status message
        self.gen_status.set(f"Processing images: {current}/{total} ({int(progress * 100)}%)") 