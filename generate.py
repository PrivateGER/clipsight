#!/usr/bin/env python3
"""
Batched ViT Image Embedding Generator

This script generates Vision Transformer (ViT) embeddings for images in a specified directory,
saving them to a file and only processing new images not already embedded.
Implements proper batch processing for significantly faster embedding generation.
"""

import os
import argparse
from typing import Dict, List, Any
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import utils

def main(args: argparse.Namespace) -> None:
    # Update embeddings file path to use .zst extension if not specified
    if not args.output.endswith('.zst'):
        args.output = args.output + '.zst'
    
    # Load existing embeddings
    embeddings_file = args.output
    existing_embeddings = utils.load_embeddings(embeddings_file)
    print(f"Loaded {len(existing_embeddings)} existing embeddings")

    # Get all image files in the directory
    image_files, skipped_files = utils.get_image_files(args.directory)
    
    # Report skipped files
    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} invalid or non-image files:")
        for path, reason in skipped_files[:10]:  # Show first 10
            print(f"  - {os.path.basename(path)}: {reason}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")
        print()
    
    print(f"Found {len(image_files)} valid images in {args.directory}")

    # Check if CLIP model
    is_clip = utils.is_clip_model(args.model)
    model_type = "CLIP" if is_clip else "ViT"

    # Load model
    print(f"Loading {model_type} model: {args.model}")
    model, processor, device, is_clip = utils.load_model(
        args.model, 
        use_fp16=args.fp16, 
        force_cpu=args.cpu
    )

    # Log device and precision information
    if device == 'cuda':
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        precision = "FP16" if args.fp16 else "FP32"
        print(f"GPU detected with {vram_gb:.2f} GB VRAM, using {precision}")
    else:
        print("Using CPU for inference")

    # Find images that need processing
    to_process = utils.find_images_to_process(image_files, existing_embeddings)

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
            results, failed = process_batch(batch_paths, model, processor, device, is_clip)

            # Update embeddings
            existing_embeddings.update(results)
            new_count += len(results)
            pbar.update(len(batch_paths))

            # Save periodically
            if (i // batch_size + 1) % args.save_interval == 0 or i + batch_size >= len(to_process):
                utils.save_embeddings(existing_embeddings, embeddings_file)
                pbar.set_postfix({"New": new_count, "Total": len(existing_embeddings)})

    # Final save
    utils.save_embeddings(existing_embeddings, embeddings_file)

    print(f"\nCompleted processing {len(to_process)} images")
    print(f"New/updated embeddings: {new_count}")
    print(f"Total embeddings in file: {len(existing_embeddings)}")
    print(f"Embeddings saved to: {embeddings_file}")

def process_batch(batch_paths: List[str], model, processor, device: str, is_clip: bool) -> Dict[str, Dict[str, Any]]:
    """Process a batch of images using the appropriate model."""
    if is_clip:
        results, failed = process_batch_clip(batch_paths, model, processor, device)
    else:
        results, failed = process_batch_vit(batch_paths, model, processor, device)
    
    return results, failed

def process_batch_vit(batch_paths: List[str], model, processor, device: str) -> Dict[str, Dict[str, Any]]:
    """Process a batch of images using a ViT model."""
    results = {}
    batch_images = []
    valid_paths = []
    failed_paths = []  # Track failed images

    # Load images
    for img_path in batch_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            batch_images.append(img)
            valid_paths.append(img_path)
        except Exception as e:
            failed_paths.append((img_path, str(e)))
            print(f"Error opening {img_path}: {e}")

    if not batch_images:
        return results, failed_paths

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
                failed_paths.append((img_path, str(inner_e)))
                print(f"Error processing individual image {img_path}: {inner_e}")

    return results, failed_paths

def process_batch_clip(batch_paths: List[str], model, processor, device: str) -> Dict[str, Dict[str, Any]]:
    """Process a batch of images using a CLIP model."""
    results = {}
    batch_images = []
    valid_paths = []
    failed_paths = []  # Track failed images

    # Load images
    for img_path in batch_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            batch_images.append(img)
            valid_paths.append(img_path)
        except Exception as e:
            failed_paths.append((img_path, str(e)))
            print(f"Error opening {img_path}: {e}")

    if not batch_images:
        return results, failed_paths

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
                failed_paths.append((img_path, str(inner_e)))
                print(f"Error processing individual image {img_path}: {inner_e}")

    return results, failed_paths

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