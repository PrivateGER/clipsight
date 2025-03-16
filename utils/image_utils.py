#!/usr/bin/env python3
"""
Image utility functions for CLIP Image Search.
"""

import os
from PIL import Image

def get_image_files(directory):
    """
    Get a list of all valid image files in the directory.
    Checks both file extension and attempts to verify file is a valid image.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    image_files = []
    skipped_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file.lower())[1]
            
            # Skip files without image extensions
            if ext not in image_extensions:
                skipped_files.append((file_path, "Not an image file extension"))
                continue
                
            # Verify file is a valid image
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify it's actually an image
                image_files.append(file_path)
            except Exception as e:
                skipped_files.append((file_path, f"Invalid image file: {str(e)}"))

    return sorted(image_files), skipped_files 