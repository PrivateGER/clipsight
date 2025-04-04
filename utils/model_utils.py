#!/usr/bin/env python3
"""
Model and embedding utilities for CLIP Image Search.
"""

import os
import json
import zstandard as zstd
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, ViTModel, ViTImageProcessor

def is_clip_model(model_name):
    """Check if the model is a CLIP model based on its name."""
    return 'clip' in model_name.lower()

def load_model(model_name):
    """Load a CLIP model with appropriate processor and tokenizer."""
    is_clip = is_clip_model(model_name)
    
    # Set device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # Load model and processor
    if is_clip:
        processor = CLIPProcessor.from_pretrained(model_name)
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
    else:
        processor = ViTImageProcessor.from_pretrained(model_name)
        tokenizer = None
        model = ViTModel.from_pretrained(model_name).to(device)
    
    return model, processor, tokenizer

def load_embeddings(file_path):
    """Load embeddings from a .json or .json.zst file."""
    if file_path.endswith('.zst'):
        with open(file_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text = reader.read().decode('utf-8')
                return json.loads(text)
    else:
        with open(file_path, 'r') as f:
            return json.load(f)

def save_embeddings(embeddings, file_path):
    """Save embeddings to a .json or .json.zst file."""
    if file_path.endswith('.zst'):
        json_str = json.dumps(embeddings)
        with open(file_path, 'wb') as fh:
            cctx = zstd.ZstdCompressor(level=10)
            compressed = cctx.compress(json_str.encode('utf-8'))
            fh.write(compressed)
    else:
        with open(file_path, 'w') as f:
            json.dump(embeddings, f)

def find_images_to_process(image_files, existing_embeddings):
    """
    Find images that need processing by comparing with existing embeddings.
    
    Args:
        image_files: List of image file paths
        existing_embeddings: Dictionary of existing embeddings
        
    Returns:
        List of image paths that need processing
    """
    # Filter out special entries like '_model_info' that aren't actual embeddings
    filtered_embeddings = {k: v for k, v in existing_embeddings.items() if not k.startswith('_')}
    
    # Extract paths from existing embeddings, handling different possible structures
    existing_paths = set()
    for key, entry in filtered_embeddings.items():
        # If the entry itself is the path (key is the filename, value contains the embedding)
        if isinstance(entry, dict) and 'path' in entry:
            existing_paths.add(entry['path'])
        # If the key is the path
        else:
            existing_paths.add(key)
    
    # Find images that don't have embeddings yet
    to_process = [path for path in image_files if path not in existing_paths]
    
    return to_process

def process_images_batch(image_paths, model_name, batch_size=16, use_fp16=False, progress_callback=None, stop_flag=None):
    """
    Process images in batches to generate embeddings.
    
    Args:
        image_paths: List of image paths to process
        model_name: Model name to use
        batch_size: Batch size for processing
        use_fp16: Whether to use FP16 precision
        progress_callback: Callback to update progress (current, total)
        stop_flag: Function that returns True if processing should stop
        
    Returns:
        Tuple of (embeddings_dict, failed_paths)
    """
    is_clip = is_clip_model(model_name)
    
    # Set device
    device = 'cpu'
    if torch.cuda.is_available():
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
    
    # Process images in batches
    results = {}
    failed_paths = []
    
    # Create batches
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        # Check if we should stop
        if stop_flag and stop_flag():
            break
        
        # Update progress
        if progress_callback:
            progress_callback(batch_idx * batch_size, len(image_paths))
        
        # Process batch
        batch_results, batch_failures = process_batch(batch, model, processor, device, is_clip)
        
        # Store results
        results.update(batch_results)
        failed_paths.extend(batch_failures)
    
    # Final progress update
    if progress_callback and not (stop_flag and stop_flag()):
        progress_callback(len(image_paths), len(image_paths))
    
    return results, failed_paths

def process_batch(batch, model, processor, device, is_clip):
    """Process a batch of images with the model."""
    results = {}
    failed_paths = []
    
    if is_clip:
        # Process for CLIP models
        try:
            # Load and process all images
            batch_images = []
            valid_paths = []
            
            for img_path in batch:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                    valid_paths.append(img_path)
                except Exception as e:
                    failed_paths.append((img_path, str(e)))
            
            if not batch_images:
                return results, failed_paths
            
            # Process batch
            inputs = processor(images=batch_images, return_tensors="pt")
            
            # Move to device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                embeddings = outputs.cpu().numpy()
                
                # Normalize embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            
            # Store results
            for i, img_path in enumerate(valid_paths):
                abs_path = os.path.abspath(img_path)
                results[abs_path] = {
                    'path': abs_path,
                    'embedding': embeddings[i].tolist()
                }
        
        except Exception as e:
            # If batch processing fails, try processing images individually
            for img_path in batch:
                try:
                    img = Image.open(img_path).convert('RGB')
                    inputs = processor(images=img, return_tensors="pt")
                    
                    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.get_image_features(**inputs)
                        embedding = outputs.cpu().numpy()[0]
                        embedding = embedding / np.linalg.norm(embedding)
                    
                    abs_path = os.path.abspath(img_path)
                    results[abs_path] = {
                        'path': abs_path,
                        'embedding': embedding.tolist()
                    }
                except Exception as inner_e:
                    failed_paths.append((img_path, str(inner_e)))
    
    else:
        # Process for ViT models
        try:
            # Load and process all images
            batch_images = []
            valid_paths = []
            
            for img_path in batch:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                    valid_paths.append(img_path)
                except Exception as e:
                    failed_paths.append((img_path, str(e)))
            
            if not batch_images:
                return results, failed_paths
            
            # Process batch
            inputs = processor(images=batch_images, return_tensors="pt")
            
            # Move to device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.pooler_output.cpu().numpy()
                
                # Normalize embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            
            # Store results
            for i, img_path in enumerate(valid_paths):
                abs_path = os.path.abspath(img_path)
                results[abs_path] = {
                    'path': abs_path,
                    'embedding': embeddings[i].tolist()
                }
        
        except Exception as e:
            # If batch processing fails, try processing images individually
            for img_path in batch:
                try:
                    img = Image.open(img_path).convert('RGB')
                    inputs = processor(images=img, return_tensors="pt")
                    
                    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        embedding = outputs.pooler_output.cpu().numpy()[0]
                        embedding = embedding / np.linalg.norm(embedding)
                    
                    abs_path = os.path.abspath(img_path)
                    results[abs_path] = {
                        'path': abs_path,
                        'embedding': embedding.tolist()
                    }
                except Exception as inner_e:
                    failed_paths.append((img_path, str(inner_e)))
    
    return results, failed_paths 