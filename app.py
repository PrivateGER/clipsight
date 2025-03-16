#!/usr/bin/env python3
"""
CLIP Search Application

Main application class for the CLIP Image Search tool.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import torch
import numpy as np
from PIL import Image
import sv_ttk
import darkdetect
import sys

from search import SearchTab
from thumbnails import ThumbnailManager
from generate_tab import GenerateTab
from utils.config import ConfigManager
from utils.model_utils import load_model, load_embeddings

class CLIPSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CLIPSight")
        self.root.geometry("1500x1000")
        self.root.minsize(800, 600)

        # Set app icon if available
        try:
            self.root.iconbitmap("search_icon.ico")
        except:
            pass

        # Initialize config manager first
        self.config_manager = ConfigManager()
        
        # Initialize variables - only init theme once here
        self._initialize_variables()
        
        # Apply theme from config before creating any UI
        self._apply_config_theme()
        
        # Create UI elements with theme already applied
        self._create_menu()
        self._create_layout()
        
        # Initialize other components
        self.thumbnail_manager = ThumbnailManager(self)
        
        # Load the rest of the config
        self._load_remaining_config()

    def _initialize_variables(self):
        """Initialize all the application variables"""
        # Basic variables
        self.embeddings_file = tk.StringVar()
        self.model_name = tk.StringVar(value="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.status_text = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0)
        
        # Auto-load settings
        self.auto_load_embeddings = tk.BooleanVar(value=False)
        self.auto_load_model = tk.BooleanVar(value=False)
        
        # Initialize theme variable only once
        self.theme_var = tk.StringVar(value="system")
        
        # Model descriptions for UI display
        self.model_descriptions = {
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": "High quality LAION model with excellent general-purpose performance",
            "openai/clip-vit-large-patch14": "OpenAI's large CLIP model - good english language alignment",
            "openai/clip-vit-base-patch32": "OpenAI's smaller, faster model - good for basic needs",
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": "Faster, smaller LAION model with good performance",
            "OFA-Sys/chinese-clip-vit-base-patch16": "Specialized model for Chinese language text and images"
        }
        
        # Model and embeddings
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.embeddings = {}
        
        # Threshold variables
        self.min_score = 0.0
        self.auto_threshold = tk.BooleanVar(value=True)
        self.manual_threshold = tk.DoubleVar(value=0.25)

    def _apply_config_theme(self):
        """Apply theme from config"""
        config = self.config_manager.load_config() or {}
        theme = config.get("theme", "system")
        
        # Set theme variable
        self.theme_var.set(theme)
        
        # Determine actual theme to use
        actual_theme = theme
        if theme == "system":
            actual_theme = darkdetect.theme().lower()
        
        # Apply theme
        print(f"Applying theme: {theme} (actual: {actual_theme})")
        sv_ttk.set_theme(actual_theme)
        
        # Apply Windows titlebar theme
        if sys.platform == 'win32':
            self._update_titlebar_theme(actual_theme)
        
        # Apply custom styles
        self._update_custom_styles(actual_theme)

    def _create_menu(self):
        """Create the application menu"""
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Embeddings File", command=self._browse_embeddings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Theme menu - use the already initialized theme_var
        theme_menu = tk.Menu(menubar, tearoff=0)
        theme_menu.add_radiobutton(label="Light", variable=self.theme_var, value="light", command=self._toggle_theme)
        theme_menu.add_radiobutton(label="Dark", variable=self.theme_var, value="dark", command=self._toggle_theme)
        theme_menu.add_radiobutton(label="System", variable=self.theme_var, value="system", command=self._toggle_theme)
        menubar.add_cascade(label="Theme", menu=theme_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _create_layout(self):
        """Create the main application layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create the search tab
        self.search_tab = SearchTab(self, self.notebook)
        
        # Create the generate tab
        self.generate_tab = GenerateTab(self, self.notebook)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, length=200, mode="determinate")
        self.progress_bar.pack(side=tk.RIGHT, padx=10, pady=5)

        status_label = ttk.Label(status_frame, textvariable=self.status_text)
        status_label.pack(side=tk.LEFT, padx=10, pady=5)

    def _browse_embeddings(self):
        """Open file dialog to select embeddings file"""
        filename = filedialog.askopenfilename(
            title="Select Embeddings File",
            filetypes=(
                ("Compressed JSON", "*.json.zst"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            )
        )
        if filename:
            self.embeddings_file.set(filename)
            self._save_config()

    def _load_model(self):
        """Load the CLIP model"""
        def load_task():
            try:
                model_name = self.model_name.get()
                self.status_text.set(f"Loading model {model_name}...")
                self.model, self.processor, self.tokenizer = load_model(model_name)
                
                device = "GPU" if torch.cuda.is_available() else "CPU"
                self.status_text.set(f"Model loaded on {device}")
            except Exception as e:
                self.status_text.set(f"Error loading model: {str(e)}")
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

        threading.Thread(target=load_task, daemon=True).start()

    def _load_embeddings(self):
        """Load embeddings from file"""
        if not os.path.isfile(self.embeddings_file.get()):
            messagebox.showerror("Error", "Please select a valid embeddings file")
            return

        def load_task():
            try:
                self.status_text.set("Loading embeddings...")
                self.progress_var.set(10)
                self.embeddings = load_embeddings(self.embeddings_file.get())
                self.progress_var.set(100)
                self.status_text.set(f"Loaded {len(self.embeddings)} embeddings")
            except Exception as e:
                self.status_text.set(f"Error loading embeddings: {str(e)}")
                messagebox.showerror("Error", f"Failed to load embeddings: {str(e)}")

        threading.Thread(target=load_task, daemon=True).start()

    def _save_config(self):
        """Save current configuration"""
        config = {
            "embeddings_file": self.embeddings_file.get(),
            "model_name": self.model_name.get(),
            "theme": self.theme_var.get(),  # Save the theme preference
            "auto_load_embeddings": self.auto_load_embeddings.get(),
            "auto_load_model": self.auto_load_model.get()
        }
        self.config_manager.save_config(config)

    def _load_remaining_config(self):
        """Load non-theme config settings"""
        config = self.config_manager.load_config() or {}
        
        if "embeddings_file" in config:
            self.embeddings_file.set(config["embeddings_file"])
        
        if "model_name" in config:
            self.model_name.set(config["model_name"])
        
        # Load auto-load preferences
        if "auto_load_embeddings" in config:
            self.auto_load_embeddings.set(config["auto_load_embeddings"])
        
        if "auto_load_model" in config:
            self.auto_load_model.set(config["auto_load_model"])
        
        # Auto-load resources if enabled
        self._auto_load_resources()

    def _auto_load_resources(self):
        """Automatically load resources based on preferences"""
        # Check if should auto-load model
        if self.auto_load_model.get() and self.model_name.get():
            self.status_text.set(f"Auto-loading model...")
            self._load_model()
        
        # Check if should auto-load embeddings
        if self.auto_load_embeddings.get() and self.embeddings_file.get() and os.path.isfile(self.embeddings_file.get()):
            self.status_text.set(f"Auto-loading embeddings...")
            self._load_embeddings()

    def _show_about(self):
        """Show about dialog"""
        about_text = """CLIPsight

A simple GUI application for searching images using CLIP embeddings.
Supports both text-based and image-based queries.

- Use text search to find images that match a description
- Use image search to find visually similar images
- Click on results to open images
"""
        messagebox.showinfo("About CLIPSight", about_text)

    def _toggle_theme(self):
        """Toggle between light and dark themes"""
        requested_theme = self.theme_var.get()
        
        # Handle "system" theme option
        actual_theme = requested_theme
        if requested_theme == "system":
            actual_theme = darkdetect.theme().lower()
        
        # Print debug info
        print(f"Toggling theme to: {requested_theme} (actual: {actual_theme})")
        
        # Set the theme
        sv_ttk.set_theme(actual_theme)
        
        # Update Windows-specific title bar if applicable
        if sys.platform == 'win32':
            self._update_titlebar_theme(actual_theme)
        
        # Update all custom styled elements
        self._update_custom_styles(actual_theme)
        
        # Save the theme preference to config
        self._save_config()

    def _update_titlebar_theme(self, theme):
        """Update Windows titlebar theme"""
        try:
            import pywinstyles
            version = sys.getwindowsversion()

            if version.major == 10 and version.build >= 22000:
                # Set the title bar color to the background color on Windows 11
                header_color = "#1c1c1c" if theme == "dark" else "#fafafa"
                pywinstyles.change_header_color(self.root, header_color)
            elif version.major == 10:
                pywinstyles.apply_style(self.root, "dark" if theme == "dark" else "normal")

                # A hacky way to update the title bar's color on Windows 10
                self.root.wm_attributes("-alpha", 0.99)
                self.root.wm_attributes("-alpha", 1)
        except (ImportError, Exception) as e:
            print(f"Error updating titlebar theme: {e}")

    def _update_custom_styles(self, theme):
        """Update all custom styles to match the current theme"""
        style = ttk.Style()
        
        # Info buttons
        style.configure("Info.TButton", font=("", 10, "bold"))
        
        # Update canvas backgrounds
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Canvas):
                bg_color = "#1c1c1c" if theme == "dark" else "#f0f0f0"
                widget.configure(bg=bg_color)
        
        # Update search tab canvas
        if hasattr(self, 'search_tab') and hasattr(self.search_tab, 'update_canvas_theme'):
            self.search_tab.update_canvas_theme(theme) 