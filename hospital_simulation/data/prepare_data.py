#!/usr/bin/env python3
"""
Script to preprocess patient data and download example images.
Run this script before starting the hospital simulation.
"""

import os
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import random
from hospital_simulation.data.preprocess_data import DataPreprocessor
from hospital_simulation.utils.env_loader import load_environment

def download_example_images():
    """Download example patient photos and X-rays."""
    example_images_dir = Path(__file__).parent / "example_images"
    example_images_dir.mkdir(exist_ok=True)

    # Example image sources for X-rays from the dataset
    print("\nDownloading example X-ray images...")
    try:
        from datasets import load_dataset
        xray_dataset = load_dataset("keremberke/chest-xray-classification", name="full")
        
        # Save some example X-rays
        for condition in ["NORMAL", "PNEUMONIA"]:
            condition_images = [img for img in xray_dataset["train"] if img["label"] == condition]
            for i, img_data in enumerate(random.sample(condition_images, 3)):
                img = img_data["image"]
                img_path = example_images_dir / f"xray_{condition.lower()}_{i+1}.jpg"
                img.save(str(img_path))
                print(f"Saved {img_path.name}")
    except Exception as e:
        print(f"Error downloading X-ray images: {e}")

def main():
    """Run data preparation."""
    print("Starting data preparation...")
    
    # Load environment variables first
    load_environment()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    try:
        # Run preprocessing pipeline
        preprocessor.preprocess_and_save()
        
        # Download example images
        print("\nDownloading example images...")
        download_example_images()
        
        print("\nData preparation completed successfully!")
        
    except Exception as e:
        print(f"\nError during data preparation: {e}")
        raise

if __name__ == "__main__":
    main() 