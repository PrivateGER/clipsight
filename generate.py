#!/usr/bin/env python3
"""
Batched ViT Image Embedding Generator

This script generates Vision Transformer (ViT) embeddings for images in a specified directory,
saving them to a file and only processing new images not already embedded.
Implements proper batch processing for significantly faster embedding generation.
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Any, Set
from pathlib import Path
import hashlib
import zstandard as zstd

import numpy as np
from tqdm import tqdm

import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor, ViTModel, ViTImageProcessor


def calculate_file_hash(file_path: str) -> str:
    """Calculate the SHA-256 hash of a file to uniquely identify it."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


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

    print(f"Found {len(to_process)} images to process out of {len(image_files)} total images")
    return to_process


def process_batch_vit(batch_paths: List[str], model: ViTModel,
                      processor: ViTImageProcessor, device: str) -> Dict[str, Dict[str, Any]]:
    """Process a batch of images using a ViT model."""
    results = {}
    batch_images = []
    valid_paths = []

    # Load images
    for img_path in batch_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            batch_images.append(img)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")

    if not batch_images:
        return results

    try:
        # Process images as a batch
        inputs = processor(images=batch_images, return_tensors="pt")

        # Move to device
        if device == 'cuda':
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Store results
        for idx, img_path in enumerate(valid_paths):
            abs_path = os.path.abspath(img_path)
            results[abs_path] = {
                'path': abs_path,
                'embedding': batch_embeddings[idx].tolist()
            }

    except Exception as e:
        print(f"Error processing batch: {e}")
        # Fall back to individual processing if batch fails
        for img_path in valid_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                inputs = processor(images=img, return_tensors="pt")

                if device == 'cuda':
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

                abs_path = os.path.abspath(img_path)
                results[abs_path] = {
                    'path': abs_path,
                    'embedding': embedding.tolist()
                }
            except Exception as inner_e:
                print(f"Error processing individual image {img_path}: {inner_e}")

    return results


def process_batch_clip(batch_paths: List[str], model: CLIPModel,
                       processor: CLIPProcessor, device: str) -> Dict[str, Dict[str, Any]]:
    """Process a batch of images using a CLIP model."""
    results = {}
    batch_images = []
    valid_paths = []

    # Load images
    for img_path in batch_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            batch_images.append(img)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")

    if not batch_images:
        return results

    try:
        # Process images as a batch
        inputs = processor(
            text=None,
            images=batch_images,
            return_tensors="pt",
            padding=True
        )

        # Remove text key and move to device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items() if k != 'text'}

        # Generate embeddings
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            batch_embeddings = outputs.cpu().numpy()

            # Normalize embeddings
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            batch_embeddings = batch_embeddings / norms

        # Store results
        for idx, img_path in enumerate(valid_paths):
            abs_path = os.path.abspath(img_path)
            results[abs_path] = {
                'path': abs_path,
                'embedding': batch_embeddings[idx].tolist()
            }

    except Exception as e:
        print(f"Error processing batch: {e}")
        # Fall back to individual processing if batch fails
        for img_path in valid_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                inputs = processor(
                    text=None,
                    images=img,
                    return_tensors="pt"
                )

                inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items() if k != 'text'}

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
                print(f"Error processing individual image {img_path}: {inner_e}")

    return results


def main(args: argparse.Namespace) -> None:
    # Update embeddings file path to use .zst extension if not specified
    if not args.output.endswith('.zst'):
        args.output = args.output + '.zst'
    
    # Load existing embeddings
    embeddings_file = args.output
    existing_embeddings = load_embeddings(embeddings_file)
    print(f"Loaded {len(existing_embeddings)} existing embeddings")

    # Get all image files in the directory
    image_files = get_image_files(args.directory)
    print(f"Found {len(image_files)} images in {args.directory}")

    # Check if CLIP model
    is_clip = is_clip_model(args.model)
    model_type = "CLIP" if is_clip else "ViT"

    # Load model and processor
    print(f"Loading {model_type} model: {args.model}")

    if is_clip:
        processor = CLIPProcessor.from_pretrained(args.model)
        model_class = CLIPModel
    else:
        processor = ViTImageProcessor.from_pretrained(args.model)
        model_class = ViTModel

    # Set up device and precision
    if torch.cuda.is_available() and not args.cpu:
        device = 'cuda'
        # Check available VRAM
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        print(f"GPU detected with {vram_gb:.2f} GB VRAM")

        # Load model with appropriate precision
        if args.fp16:
            print(f"Loading {model_type} model with half precision (FP16)")
            model = model_class.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
        else:
            print(f"Loading {model_type} model with full precision (FP32)")
            model = model_class.from_pretrained(args.model).to(device)

        print("Using GPU for inference")
    else:
        device = 'cpu'
        model = model_class.from_pretrained(args.model)
        print("Using CPU for inference")

    # Find images that need processing
    to_process = find_images_to_process(image_files, existing_embeddings)

    # Set batch size
    batch_size = args.batch_size
    print(f"Using batch size of {batch_size}")

    # Process in batches
    new_count = 0
    total_batches = (len(to_process) + batch_size - 1) // batch_size

    with tqdm(total=len(to_process), desc="Processing images") as pbar:
        for i in range(0, len(to_process), batch_size):
            batch_paths = to_process[i:i + batch_size]

            # Process batch
            if is_clip:
                batch_results = process_batch_clip(batch_paths, model, processor, device)
            else:
                batch_results = process_batch_vit(batch_paths, model, processor, device)

            # Update embeddings
            existing_embeddings.update(batch_results)
            new_count += len(batch_results)
            pbar.update(len(batch_paths))

            # Save periodically
            if (i // batch_size + 1) % args.save_interval == 0 or i + batch_size >= len(to_process):
                save_embeddings(existing_embeddings, embeddings_file)
                pbar.set_postfix({"New": new_count, "Total": len(existing_embeddings)})

    # Final save
    save_embeddings(existing_embeddings, embeddings_file)

    print(f"\nCompleted processing {len(to_process)} images")
    print(f"New/updated embeddings: {new_count}")
    print(f"Total embeddings in file: {len(existing_embeddings)}")
    print(f"Embeddings saved to: {embeddings_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ViT/CLIP embeddings for images")
    parser.add_argument("--directory", "-d", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--output", "-o", type=str, default="embeddings.json.zst",
                        help="Output file for embeddings")
    parser.add_argument("--model", "-m", type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                        help="ViT/CLIP model to use")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference even if GPU is available")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision to save VRAM")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save embeddings every N batches")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for processing images (requires more VRAM)")

    args = parser.parse_args()
    main(args)