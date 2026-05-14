#!/usr/bin/env python3
"""
Extract Sample CT Scan Images from Training Dataset

This script extracts sample images from the all_patches.hdf5 file
and saves them as PNG files for testing the Streamlit app.

Usage:
    python extract_samples.py
    python extract_samples.py --num-samples 10 --output-dir test_images
"""

import h5py
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path


def extract_sample_images(
    hdf5_path='all_patches.hdf5',
    output_dir='sample_images',
    num_malignant=5,
    num_benign=5,
    seed=42
):
    """
    Extract sample CT scan images from HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file containing CT patches
        output_dir: Directory to save extracted images
        num_malignant: Number of malignant (cancer) samples to extract
        num_benign: Number of benign (non-cancer) samples to extract
        seed: Random seed for reproducibility
    """

    print("=" * 60)
    print("CT Scan Sample Image Extractor")
    print("=" * 60)
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_path.absolute()}")
    print()

    # Check if HDF5 file exists
    if not os.path.exists(hdf5_path):
        print(f"❌ Error: File '{hdf5_path}' not found!")
        print("   Please ensure the HDF5 file is in the current directory.")
        return

    print(f"📂 Loading data from: {hdf5_path}")

    try:
        # Load data
        with h5py.File(hdf5_path, 'r') as f:
            images = f['ct_slices'][:]
            labels = f['slice_class'][:]

        labels = labels.reshape(-1)

        print(f"✅ Loaded {len(images)} images")
        print(f"   - Shape: {images.shape}")
        print(f"   - Labels shape: {labels.shape}")
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

        # Ensure we don't request more samples than available
        num_benign = min(num_benign, len(benign_indices))
        num_malignant = min(num_malignant, len(malignant_indices))

        # Randomly sample indices
        selected_benign = np.random.choice(benign_indices, num_benign, replace=False)
        selected_malignant = np.random.choice(malignant_indices, num_malignant, replace=False)

        print("🎯 Extracting Images:")
        print(f"   - {num_benign} benign samples")
        print(f"   - {num_malignant} malignant samples")
        print()

        # Extract and save benign images
        print("💾 Saving benign images...")
        for i, idx in enumerate(selected_benign, 1):
            img_array = images[idx]

            # Normalize to 0-255
            img_normalized = ((img_array - img_array.min()) /
                            (img_array.max() - img_array.min()) * 255).astype(np.uint8)

            # Save as PNG
            filename = output_path / f"benign_{i:03d}.png"
            Image.fromarray(img_normalized).save(filename)
            print(f"   ✅ {filename.name}")

        # Extract and save malignant images
        print()
        print("💾 Saving malignant images...")
        for i, idx in enumerate(selected_malignant, 1):
            img_array = images[idx]

            # Normalize to 0-255
            img_normalized = ((img_array - img_array.min()) /
                            (img_array.max() - img_array.min()) * 255).astype(np.uint8)

            # Save as PNG
            filename = output_path / f"malignant_{i:03d}.png"
            Image.fromarray(img_normalized).save(filename)
            print(f"   ✅ {filename.name}")

        print()
        print("=" * 60)
        print("✨ Extraction Complete!")
        print("=" * 60)
        print()
        print(f"📁 Saved {num_benign + num_malignant} images to: {output_path.absolute()}")
        print()
        print("🚀 Next Steps:")
        print("   1. Start the Streamlit app: ./run_app.sh")
        print("   2. Upload images from the sample_images/ folder")
        print("   3. Compare predictions with known labels")
        print()
        print("📝 Image Labels:")
        print("   - benign_*.png   → Should predict: NO CANCER (low score)")
        print("   - malignant_*.png → Should predict: CANCER (high score)")
        print()

        # Create a README in the output directory
        readme_path = output_path / "README.txt"
        with open(readme_path, 'w') as f:
            f.write("CT Scan Sample Images\n")
            f.write("=" * 60 + "\n\n")
            f.write("This folder contains sample lung nodule CT scan images\n")
            f.write("extracted from the training dataset.\n\n")
            f.write("Image Labels:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Benign Images (Class 0):   {num_benign} files\n")
            f.write(f"Malignant Images (Class 1): {num_malignant} files\n\n")
            f.write("Expected Predictions:\n")
            f.write("-" * 60 + "\n")
            f.write("benign_*.png:\n")
            f.write("  - Prediction: NO CANCER DETECTED\n")
            f.write("  - Score: < 0.5 (lower is better)\n")
            f.write("  - Confidence: High for 'No Cancer'\n\n")
            f.write("malignant_*.png:\n")
            f.write("  - Prediction: CANCER DETECTED\n")
            f.write("  - Score: > 0.5 (higher indicates cancer)\n")
            f.write("  - Confidence: High for 'Cancer'\n\n")
            f.write("How to Use:\n")
            f.write("-" * 60 + "\n")
            f.write("1. Start the Streamlit app: ./run_app.sh\n")
            f.write("2. Upload any image from this folder\n")
            f.write("3. Click 'Analyze CT Scan'\n")
            f.write("4. Review the prediction and Grad-CAM visualization\n")
            f.write("5. Compare with the known label (filename)\n\n")
            f.write("Note: These images are from the training/test dataset,\n")
            f.write("so the model should perform well on them.\n\n")
            f.write(f"Generated: {np.datetime64('now')}\n")

        print(f"📄 Created README.txt in {output_dir}/")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Extract sample CT scan images from HDF5 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 5 benign and 5 malignant images (default)
  python extract_samples.py

  # Extract 10 of each type
  python extract_samples.py --num-samples 10

  # Custom output directory
  python extract_samples.py --output-dir test_images

  # Extract different numbers for each class
  python extract_samples.py --benign 3 --malignant 7
        """
    )

    parser.add_argument(
        '--hdf5-path',
        default='all_patches.hdf5',
        help='Path to HDF5 file (default: all_patches.hdf5)'
    )

    parser.add_argument(
        '--output-dir',
        default='sample_images',
        help='Output directory for extracted images (default: sample_images)'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        help='Number of samples per class (overrides --benign and --malignant)'
    )

    parser.add_argument(
        '--benign',
        type=int,
        default=5,
        help='Number of benign samples (default: 5)'
    )

    parser.add_argument(
        '--malignant',
        type=int,
        default=5,
        help='Number of malignant samples (default: 5)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # If --num-samples is provided, use it for both classes
    num_benign = args.num_samples if args.num_samples else args.benign
    num_malignant = args.num_samples if args.num_samples else args.malignant

    extract_sample_images(
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        num_malignant=num_malignant,
        num_benign=num_benign,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
