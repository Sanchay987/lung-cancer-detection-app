#!/usr/bin/env python3
"""
Extract Sample CT Scan Images (Corrected Version)

This script extracts and saves sample images in both:
1. Preprocessed NPY format (for accurate model predictions)
2. PNG format (for visualization only)
3. Metadata JSON (with HU ranges for proper preprocessing)

Usage:
    python extract_samples_corrected.py
"""

import h5py
import numpy as np
from PIL import Image
import os
import json
import tensorflow as tf
from pathlib import Path


def extract_corrected_samples(
    hdf5_path='all_patches.hdf5',
    output_dir='sample_images_corrected',
    num_malignant=5,
    num_benign=5,
    seed=42
):
    """Extract sample CT scan images with proper HU preservation."""

    print("=" * 80)
    print("CT Scan Sample Extractor (Corrected - HU Preserving)")
    print("=" * 80)
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_path.absolute()}")
    print()

    if not os.path.exists(hdf5_path):
        print(f"❌ Error: File '{hdf5_path}' not found!")
        return

    print(f"📂 Loading data from: {hdf5_path}")

    # Load data
    with h5py.File(hdf5_path, 'r') as f:
        images = f['ct_slices'][:]
        labels = f['slice_class'][:]

    labels = labels.reshape(-1)

    print(f"✅ Loaded {len(images)} images")
    print(f"   - Original shape: {images.shape}")
    print(f"   - HU range: [{images.min():.0f}, {images.max():.0f}]")
    print()

    # Count classes
    num_class_0 = np.sum(labels == 0)
    num_class_1 = np.sum(labels == 1)

    print(f"📊 Dataset Statistics:")
    print(f"   - Benign (Class 0): {num_class_0} images")
    print(f"   - Malignant (Class 1): {num_class_1} images")
    print()

    # Set random seed
    np.random.seed(seed)

    # Get indices for each class
    benign_indices = np.where(labels == 0)[0]
    malignant_indices = np.where(labels == 1)[0]

    # Sample indices
    num_benign = min(num_benign, len(benign_indices))
    num_malignant = min(num_malignant, len(malignant_indices))

    selected_benign = np.random.choice(benign_indices, num_benign, replace=False)
    selected_malignant = np.random.choice(malignant_indices, num_malignant, replace=False)

    print("🎯 Extracting and preprocessing images:")
    print(f"   - {num_benign} benign samples")
    print(f"   - {num_malignant} malignant samples")
    print()

    metadata = {}
    target_size = (96, 96)

    # Process benign images
    print("💾 Saving benign images...")
    for i, idx in enumerate(selected_benign, 1):
        img_hu = images[idx]

        # Apply EXACT training preprocessing
        img_3channel = np.stack([img_hu] * 3, axis=-1)
        img_resized = tf.image.resize(img_3channel, target_size)
        img_preprocessed = img_resized.numpy().astype('float32') / 255.0

        # Save preprocessed NPY (for prediction)
        npy_filename = output_path / f"benign_{i:03d}.npy"
        np.save(npy_filename, img_preprocessed)

        # Save visualization PNG (min-max normalized)
        img_vis = ((img_hu - img_hu.min()) /
                   (img_hu.max() - img_hu.min()) * 255).astype(np.uint8)
        img_vis_resized = np.array(Image.fromarray(img_vis).resize(target_size))
        png_filename = output_path / f"benign_{i:03d}.png"
        Image.fromarray(img_vis_resized).save(png_filename)

        # Store metadata
        metadata[f"benign_{i:03d}"] = {
            "label": 0,
            "label_name": "benign",
            "hu_min": float(img_hu.min()),
            "hu_max": float(img_hu.max()),
            "hu_mean": float(img_hu.mean()),
            "original_index": int(idx)
        }

        print(f"   ✅ benign_{i:03d}: HU[{img_hu.min():.0f}, {img_hu.max():.0f}]")

    # Process malignant images
    print()
    print("💾 Saving malignant images...")
    for i, idx in enumerate(selected_malignant, 1):
        img_hu = images[idx]

        # Apply EXACT training preprocessing
        img_3channel = np.stack([img_hu] * 3, axis=-1)
        img_resized = tf.image.resize(img_3channel, target_size)
        img_preprocessed = img_resized.numpy().astype('float32') / 255.0

        # Save preprocessed NPY (for prediction)
        npy_filename = output_path / f"malignant_{i:03d}.npy"
        np.save(npy_filename, img_preprocessed)

        # Save visualization PNG (min-max normalized)
        img_vis = ((img_hu - img_hu.min()) /
                   (img_hu.max() - img_hu.min()) * 255).astype(np.uint8)
        img_vis_resized = np.array(Image.fromarray(img_vis).resize(target_size))
        png_filename = output_path / f"malignant_{i:03d}.png"
        Image.fromarray(img_vis_resized).save(png_filename)

        # Store metadata
        metadata[f"malignant_{i:03d}"] = {
            "label": 1,
            "label_name": "malignant",
            "hu_min": float(img_hu.min()),
            "hu_max": float(img_hu.max()),
            "hu_mean": float(img_hu.mean()),
            "original_index": int(idx)
        }

        print(f"   ✅ malignant_{i:03d}: HU[{img_hu.min():.0f}, {img_hu.max():.0f}]")

    # Save metadata JSON
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 80)
    print("✨ Extraction Complete!")
    print("=" * 80)
    print()
    print(f"📁 Saved {num_benign + num_malignant} image pairs to: {output_path.absolute()}")
    print()
    print("📦 Files created:")
    print("   - *.npy: Preprocessed images ready for model prediction")
    print("   - *.png: Visualization images (for display only)")
    print("   - metadata.json: HU ranges and labels")
    print()
    print("🚀 Usage in app:")
    print("   - Load NPY files for direct prediction (already preprocessed)")
    print("   - Use PNG files for visualization")
    print("   - Check metadata.json for ground truth labels")
    print()

    # Create README
    readme_path = output_path / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("CT Scan Sample Images (Corrected - HU Preserving)\n")
        f.write("=" * 80 + "\n\n")
        f.write("This folder contains lung nodule CT scan samples with proper HU preservation.\n\n")
        f.write("Files:\n")
        f.write("-" * 80 + "\n")
        f.write("*.npy files: Preprocessed images (96x96x3) ready for model.predict()\n")
        f.write("             These are already preprocessed identically to training data.\n\n")
        f.write("*.png files: Visualization images (for human viewing only)\n")
        f.write("             These use min-max normalization for better contrast.\n")
        f.write("             DO NOT use these for model predictions!\n\n")
        f.write("metadata.json: Contains HU ranges and ground truth labels\n\n")
        f.write("Expected Predictions:\n")
        f.write("-" * 80 + "\n")
        f.write(f"benign_*.npy:     Label 0 → Model should predict < 0.5 (NO CANCER)\n")
        f.write(f"malignant_*.npy:  Label 1 → Model should predict > 0.5 (CANCER)\n\n")
        f.write("Usage Example (Python):\n")
        f.write("-" * 80 + "\n")
        f.write("import numpy as np\n")
        f.write("from keras.models import load_model\n\n")
        f.write("# Load model\n")
        f.write("model = load_model('final_novel_attention_model.keras')\n\n")
        f.write("# Load preprocessed image\n")
        f.write("img = np.load('sample_images_corrected/malignant_001.npy')\n\n")
        f.write("# Add batch dimension\n")
        f.write("img_batch = np.expand_dims(img, axis=0)\n\n")
        f.write("# Predict\n")
        f.write("prediction = model.predict(img_batch)\n")
        f.write("print(f'Prediction: {prediction[0][0]:.4f}')\n\n")

    print(f"📄 Created README.txt in {output_dir}/")
    print()


if __name__ == "__main__":
    extract_corrected_samples()
