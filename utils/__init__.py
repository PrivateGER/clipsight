#!/usr/bin/env python3
"""
Utility functions for CLIP Image Search.
"""

from utils.config import ConfigManager
from utils.image_utils import get_image_files
from utils.model_utils import (
    load_model, 
    load_embeddings, 
    save_embeddings, 
    find_images_to_process,
    process_images_batch
)
from utils.tooltip import ToolTip 

def get_current_datetime_str():
    """Get current datetime as a string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S") 