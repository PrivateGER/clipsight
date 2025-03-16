#!/usr/bin/env python3
"""
Thumbnail handling for CLIP Image Search.
"""

import os
import sys
import hashlib
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

import utils


class ThumbnailManager:
    def __init__(self, app):
        self.app = app
        self.thumbnail_cache = {}
        self.thumbnail_load_queue = []
        self.is_loading_thumbnails = False
        
        # Create thumbnail directory
        self.thumbnail_dir = os.path.join(os.path.expanduser("~"), ".clip_search", "thumbnails")
        os.makedirs(self.thumbnail_dir, exist_ok=True)
    
    def update_results_page(self, container, canvas, result_paths, current_page, results_per_page):
        """Update the displayed results for current page"""
        # Cancel any pending thumbnail loads
        self.thumbnail_load_queue = []
        self.is_loading_thumbnails = False
        
        # Clear current page
        for widget in container.winfo_children():
            widget.destroy()

        # Calculate start and end indices
        start_idx = current_page * results_per_page
        end_idx = min(start_idx + results_per_page, len(result_paths))
        
        if start_idx >= end_idx:
            return

        # Calculate layout
        container_width = canvas.winfo_width()
        if container_width <= 1:  # Canvas not yet properly sized
            self.app.root.update_idletasks()
            container_width = canvas.winfo_width()
        
        scrollbar_width = 20
        available_width = max(container_width - scrollbar_width, 200)  # Minimum width
        thumbnail_width = 200
        padding = 10
        min_spacing = 5
        
        columns = max(1, (available_width + min_spacing) // (thumbnail_width + padding + min_spacing))
        remaining_width = available_width - (columns * (thumbnail_width + padding))
        extra_spacing = remaining_width // (columns + 1) if columns > 1 else 0

        # Reset grid configuration
        for i in range(columns):
            container.grid_columnconfigure(i, weight=1, minsize=thumbnail_width)

        # Create all frames and placeholders first
        frames = []
        for i, (path, score) in enumerate(result_paths[start_idx:end_idx]):
            row = i // columns
            col = i % columns

            # Create frame
            img_frame = ttk.Frame(container)
            img_frame.grid(row=row, column=col, padx=(extra_spacing + 5), pady=5, sticky=tk.NW)
            frames.append((img_frame, path, score))

            # Create placeholder
            placeholder = ttk.Label(img_frame, text="Loading...", width=20, anchor="center")
            placeholder.pack(pady=80)

        # Start loading thumbnails in batches
        self.is_loading_thumbnails = True
        self._load_thumbnail_batch(frames)
    
    def _load_thumbnail_batch(self, frames, batch_size=5):
        """Load thumbnails in small batches to prevent UI freezing"""
        if not frames or not self.is_loading_thumbnails:
            self.is_loading_thumbnails = False
            return
        
        # Process a batch
        batch = frames[:batch_size]
        remaining = frames[batch_size:]
        
        def process_batch():
            for img_frame, path, score in batch:
                try:
                    if not img_frame.winfo_exists():
                        continue
                    
                    # Clear placeholder
                    for widget in img_frame.winfo_children():
                        widget.destroy()

                    # Load thumbnail
                    thumb_path = self._get_thumbnail_path(path)
                    if path in self.thumbnail_cache:
                        photo = self.thumbnail_cache[path]
                    else:
                        try:
                            if os.path.exists(thumb_path):
                                # Fast load for existing thumbnails
                                with Image.open(thumb_path) as img:
                                    photo = ImageTk.PhotoImage(img)
                            else:
                                # Optimized thumbnail generation
                                with Image.open(path) as img:
                                    # Convert to RGB only if needed
                                    if img.mode not in ('RGB', 'L'):
                                        img = img.convert('RGB')
                                    
                                    # Calculate target size maintaining aspect ratio
                                    target_size = (200, 200)
                                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                                    
                                    # Use optimize flag and higher compression
                                    img.save(thumb_path, 'WEBP', 
                                           quality=80,  # Slightly lower quality
                                           method=4,    # Faster compression
                                           optimize=True)
                                    photo = ImageTk.PhotoImage(img)
                        
                            self.thumbnail_cache[path] = photo
                        except Exception as e:
                            print(f"Error creating thumbnail for {path}: {e}")
                            # Create error placeholder
                            error_img = Image.new('RGB', (200, 200), color='gray')
                            photo = ImageTk.PhotoImage(error_img)

                    # Create and pack image label
                    img_label = ttk.Label(img_frame, image=photo)
                    img_label.pack()

                    # Add filename and score labels
                    name_label = ttk.Label(img_frame, text=os.path.basename(path), wraplength=180)
                    name_label.pack()
                    # Display score as percentage
                    score_pct = score * 100
                    score_label = ttk.Label(img_frame, text=f"Score: {score_pct:.1f}%")
                    score_label.pack()

                    # Bind events
                    img_label.bind("<Button-1>", lambda e, p=path: self._open_image(p))
                    
                    # Add context menu
                    img_context_menu = tk.Menu(img_label, tearoff=0)
                    img_context_menu.add_command(label="Open Image", 
                                               command=lambda p=path: self._open_image(p))
                    img_context_menu.add_command(label="Search Similar Images", 
                                               command=lambda p=path: self._search_by_result(p))
                    img_context_menu.add_separator()
                    img_context_menu.add_command(label="Delete from Index", 
                                               command=lambda p=path: self._delete_from_index(p))
                    
                    def show_context_menu(event, menu=img_context_menu):
                        menu.tk_popup(event.x_root, event.y_root)
                        
                    img_label.bind("<Button-3>", show_context_menu)
                    if sys.platform == 'darwin':
                        img_label.bind("<Button-2>", show_context_menu)

                except Exception as e:
                    print(f"Error loading thumbnail for {path}: {e}")

            # Schedule next batch
            if remaining:
                self.app.root.after(50, lambda: self._load_thumbnail_batch(remaining))
            else:
                self.is_loading_thumbnails = False

        # Run batch processing in a thread
        threading.Thread(target=process_batch, daemon=True).start()
    
    def _open_image(self, image_path):
        """Open the image in the default image viewer"""
        if sys.platform == 'win32':
            os.startfile(image_path)
        elif sys.platform == 'darwin':  # macOS
            os.system(f'open "{image_path}"')
        else:  # Linux
            os.system(f'xdg-open "{image_path}"')
    
    def _search_by_result(self, image_path):
        """Use a result image as a new search query"""
        search_tab = self.app.search_tab
        search_tab.query_image.set(image_path)
        search_tab.query_text.set("")  # Clear any text query
        search_tab._image_search()
    
    def _get_thumbnail_path(self, image_path):
        """Get path for cached thumbnail"""
        # Create hash of original path to use as filename
        path_hash = hashlib.md5(image_path.encode()).hexdigest()
        return os.path.join(self.thumbnail_dir, f"{path_hash}.webp")
    
    def cleanup_cache(self):
        """Clean up thumbnail cache"""
        # Clear memory cache
        self.thumbnail_cache.clear()
        
        # Clean up disk cache
        try:
            # Get all thumbnail files and their modification times
            thumb_files = []
            for f in os.listdir(self.thumbnail_dir):
                path = os.path.join(self.thumbnail_dir, f)
                mtime = os.path.getmtime(path)
                size = os.path.getsize(path)
                thumb_files.append((path, mtime, size))
            
            # Sort by modification time (oldest first)
            thumb_files.sort(key=lambda x: x[1])
            
            # Calculate total size
            total_size = sum(size for _, _, size in thumb_files)
            max_cache_size = 500 * 1024 * 1024  # 500MB
            
            # Remove old files until we're under the limit
            while total_size > max_cache_size and thumb_files:
                path, _, size = thumb_files.pop(0)  # Remove oldest
                try:
                    os.remove(path)
                    total_size -= size
                except OSError:
                    pass
                
        except Exception as e:
            print(f"Error cleaning thumbnail cache: {e}")
    
    def _delete_from_index(self, image_path):
        """Delete an image from the embeddings index with confirmation"""
        filename = os.path.basename(image_path)
        if not messagebox.askyesno("Confirm Deletion", 
                                 f"Are you sure you want to delete {filename} from the index?\n\n"
                                 f"This won't delete the actual file, only its entry in the embeddings database."):
            return
            
        # Find absolute path in case it's not already absolute
        abs_path = os.path.abspath(image_path)
        
        # Check if path exists in embeddings
        if abs_path in self.app.embeddings:
            # Remove from embeddings
            del self.app.embeddings[abs_path]
            
            # Save updated embeddings
            embeddings_file = self.app.embeddings_file.get()
            try:
                utils.save_embeddings(self.app.embeddings, embeddings_file)
                self.app.status_text.set(f"Removed {filename} from index")
                
                # Update search results if this was part of the results
                search_tab = self.app.search_tab
                if search_tab.cached_results:
                    # Remove from cached results
                    search_tab.cached_results = [(p, s) for p, s in search_tab.cached_results if p != abs_path]
                    # Remove from current results
                    search_tab.result_paths = [(p, s) for p, s in search_tab.result_paths if p != abs_path]
                    # Update the display
                    search_tab._update_results_page()
                
            except Exception as e:
                self.app.status_text.set(f"Error saving updated embeddings: {str(e)}")
                messagebox.showerror("Error", f"Failed to save updated embeddings: {str(e)}")
        else:
            # Try to find by relative path comparison
            found = False
            for key in list(self.app.embeddings.keys()):
                if os.path.basename(key) == filename:
                    del self.app.embeddings[key]
                    found = True
                    break
                    
            if found:
                try:
                    utils.save_embeddings(self.app.embeddings, self.app.embeddings_file.get())
                    self.app.status_text.set(f"Removed {filename} from index")
                    
                    # Update search results
                    search_tab = self.app.search_tab
                    if search_tab.cached_results:
                        # Remove from results by filename
                        search_tab.cached_results = [(p, s) for p, s in search_tab.cached_results 
                                                   if os.path.basename(p) != filename]
                        search_tab.result_paths = [(p, s) for p, s in search_tab.result_paths 
                                                 if os.path.basename(p) != filename]
                        search_tab._update_results_page()

                    self.app.embeddings = utils.load_embeddings(self.app.embeddings_file.get())
                    self.app.status_text.set(f"Removed {filename} from index")
                        
                except Exception as e:
                    self.app.status_text.set(f"Error saving updated embeddings: {str(e)}")
                    messagebox.showerror("Error", f"Failed to save updated embeddings: {str(e)}")
            else:
                messagebox.showinfo("Not Found", f"Could not find {filename} in the index") 