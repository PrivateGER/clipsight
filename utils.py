#!/usr/bin/env python3
"""
Shared utilities for CLIP image search and embedding generation.
"""

import os
import json
import torch
import zstandard as zstd
from typing import Dict, List, Any
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, ViTModel, ViTImageProcessor


def get_image_files(directory: str) -> List[str]:
    """Get a list of all image files in the directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif']
    image_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    return sorted(image_files)


def load_embeddings(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load existing embeddings from a file."""
    if not os.path.exists(file_path):
        # Try with .zst extension
        zst_path = file_path + '.zst'
        if os.path.exists(zst_path):
            file_path = zst_path
        else:
            return {}

    try:
        if file_path.endswith('.zst'):
            with open(file_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                json_str = dctx.decompress(f.read()).decode('utf-8')
                return json.loads(json_str)
        else:
            # Legacy JSON support
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load embeddings from {file_path}. Starting fresh. Error: {e}")
        return {}


def save_embeddings(embeddings: Dict[str, Dict[str, Any]], file_path: str) -> None:
    """Save embeddings to a compressed file."""
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Ensure the file has .zst extension
    if not file_path.endswith('.zst'):
        file_path = file_path + '.zst'

    # Convert to JSON string
    json_str = json.dumps(embeddings, indent=None)

    # Compress and save
    cctx = zstd.ZstdCompressor(level=10)  # Higher compression level
    compressed = cctx.compress(json_str.encode('utf-8'))
    
    with open(file_path, 'wb') as f:
        f.write(compressed)


def is_clip_model(model_name: str) -> bool:
    """Check if the model is a CLIP model based on its name."""
    return 'clip' in model_name.lower()


def load_model(model_name: str, use_fp16: bool = False, force_cpu: bool = False):
    """Load a CLIP or ViT model with appropriate processor."""
    is_clip = is_clip_model(model_name)
    
    # Set device
    device = 'cpu'
    if torch.cuda.is_available() and not force_cpu:
        device = 'cuda'
    
    # Load processor
    if is_clip:
        processor = CLIPProcessor.from_pretrained(model_name)
        model_class = CLIPModel
    else:
        processor = ViTImageProcessor.from_pretrained(model_name)
        model_class = ViTModel
    
    # Load model with appropriate precision
    if device == 'cuda' and use_fp16:
        model = model_class.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    else:
        model = model_class.from_pretrained(model_name).to(device)
        
    return model, processor, device, is_clip


def find_images_to_process(image_files: List[str], existing_embeddings: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Find images that need to be processed based on path comparison.
    Returns a list of image paths that need processing.
    """
    to_process = []
    existing_paths = {entry['path'] for entry in existing_embeddings.values()}

    for image_path in image_files:
        abs_path = os.path.abspath(image_path)
        if abs_path not in existing_paths:
            to_process.append(abs_path)

    return to_process 