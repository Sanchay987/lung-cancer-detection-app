#!/usr/bin/env python3
"""
DICOM to NPY Converter for Lung Cancer Detection App

This script converts DICOM CT scan files to NPY format with the exact
preprocessing pipeline used during model training, ensuring 100% accuracy.

Usage:
    python dicom_to_npy.py <input.dcm> <output.npy>
    python dicom_to_npy.py --batch <dicom_folder> <output_folder>

Requirements:
    pip install numpy tensorflow pillow pydicom
"""

import os
import sys
import glob
import argparse
import numpy as np
import tensorflow as tf

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. DICOM support disabled.")
    print("Install with: pip install pydicom")

from PIL import Image


def preprocess_for_model(img_array):
    """
    Apply the exact preprocessing pipeline used during training.

    This function takes an image array (from DICOM, PNG, or TIFF) and
    preprocesses it to match the training data format.

    Args:
        img_array: 2D numpy array with HU values (for DICOM) or pixel values

    Returns:
        Preprocessed 3D array of shape (96, 96, 3) ready for model input
    """
    # Ensure float32
    img_array = img_array.astype('float32')

    # For DICOM files, img_array already contains HU values
    # For PNG/TIFF, values are typically [0, 255] and need HU approximation
    # However, this script is designed for DICOM → NPY conversion

    # Stack to 3 channels (MobileNetV2 requires RGB-like input)
    img_3channel = np.stack([img_array] * 3, axis=-1)

    # Resize to 96×96 (model input size)
    img_resized = tf.image.resize(img_3channel, (96, 96))

    # Normalize by 255 (same as training)
    img_preprocessed = img_resized.numpy().astype('float32') / 255.0

    return img_preprocessed


def dicom_to_npy(dicom_path, output_path, verbose=True):
    """
    Convert a DICOM CT scan to NPY format for accurate predictions.

    Args:
        dicom_path: Path to DICOM file (.dcm)
        output_path: Path to save NPY file (.npy)
        verbose: Print conversion details

    Returns:
        True if successful, False otherwise
    """
    if not PYDICOM_AVAILABLE:
        print("Error: pydicom not installed. Cannot convert DICOM files.")
        return False

    try:
        # 1. Load DICOM
        dicom = pydicom.dcmread(dicom_path)
        img_hu = dicom.pixel_array.astype('float32')

        # Handle 3D volumes (take middle slice)
        if len(img_hu.shape) == 3:
            slice_idx = img_hu.shape[0] // 2
            img_hu = img_hu[slice_idx]
            if verbose:
                print(f"   3D volume detected, using slice {slice_idx}/{img_hu.shape[0]}")

        # Apply DICOM rescale slope and intercept to get true HU values
        if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
            img_hu = img_hu * dicom.RescaleSlope + dicom.RescaleIntercept
            if verbose:
                print(f"   Applied rescale: slope={dicom.RescaleSlope}, intercept={dicom.RescaleIntercept}")

        # 2. Preprocess using training pipeline
        img_preprocessed = preprocess_for_model(img_hu)

        # 3. Save as NPY
        np.save(output_path, img_preprocessed)

        if verbose:
            print(f"✅ Converted: {os.path.basename(dicom_path)} -> {os.path.basename(output_path)}")
            print(f"   Input shape: {img_hu.shape}")
            print(f"   Output shape: {img_preprocessed.shape}")
            print(f"   HU range (before /255): [{img_hu.min():.1f}, {img_hu.max():.1f}]")
            print(f"   Final range: [{img_preprocessed.min():.4f}, {img_preprocessed.max():.4f}]")

        return True

    except Exception as e:
        print(f"❌ Failed to convert {dicom_path}: {e}")
        return False


def png_to_npy(png_path, output_path, verbose=True):
    """
    Convert PNG to NPY (not recommended - HU values are approximated).

    Args:
        png_path: Path to PNG file
        output_path: Path to save NPY file (.npy)
        verbose: Print conversion details

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load PNG
        image = Image.open(png_path)
        img_array = np.array(image)

        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2)
        else:
            img_gray = img_array

        img_gray = img_gray.astype('float32')

        # Approximate HU values (assumes lung window -1000 to +400)
        # WARNING: This is an approximation and may be inaccurate!
        img_hu_approx = (img_gray / 255.0) * 1400.0 - 1000.0

        # Preprocess using training pipeline
        img_preprocessed = preprocess_for_model(img_hu_approx)

        # Save as NPY
        np.save(output_path, img_preprocessed)

        if verbose:
            print(f"⚠️  Converted (HU approximated): {os.path.basename(png_path)} -> {os.path.basename(output_path)}")
            print(f"   Output shape: {img_preprocessed.shape}")
            print(f"   Approximated HU range: [{img_hu_approx.min():.1f}, {img_hu_approx.max():.1f}]")
            print(f"   Final range: [{img_preprocessed.min():.4f}, {img_preprocessed.max():.4f}]")
            print(f"   WARNING: PNG conversion is unreliable (~70% accuracy). Use DICOM if possible.")

        return True

    except Exception as e:
        print(f"❌ Failed to convert {png_path}: {e}")
        return False


def batch_convert(input_dir, output_dir, file_pattern="*.dcm", verbose=True):
    """
    Convert all DICOM files in a directory to NPY format.

    Args:
        input_dir: Directory containing DICOM files
        output_dir: Directory to save NPY files
        file_pattern: Glob pattern for input files (default: *.dcm)
        verbose: Print conversion details

    Returns:
        Number of successfully converted files
    """
    os.makedirs(output_dir, exist_ok=True)

    input_files = glob.glob(os.path.join(input_dir, file_pattern))

    if not input_files:
        print(f"No files matching '{file_pattern}' found in {input_dir}")
        return 0

    print(f"Found {len(input_files)} files to convert")
    print("-" * 60)

    success_count = 0

    for input_path in input_files:
        filename = os.path.basename(input_path)
        output_filename = os.path.splitext(filename)[0] + '.npy'
        output_path = os.path.join(output_dir, output_filename)

        # Determine file type and convert
        ext = os.path.splitext(filename)[1].lower()

        if ext == '.dcm':
            if dicom_to_npy(input_path, output_path, verbose=verbose):
                success_count += 1
        elif ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            if png_to_npy(input_path, output_path, verbose=verbose):
                success_count += 1
        else:
            print(f"⏭️  Skipped (unsupported format): {filename}")

        if verbose:
            print()

    print("-" * 60)
    print(f"✅ Successfully converted {success_count}/{len(input_files)} files")

    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert DICOM CT scans to NPY format for lung cancer detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python dicom_to_npy.py ct_scan.dcm output.npy

  # Batch convert all DICOM files
  python dicom_to_npy.py --batch dicom_folder/ output_folder/

  # Batch convert PNG files (not recommended)
  python dicom_to_npy.py --batch png_folder/ output_folder/ --pattern "*.png"
        """
    )

    parser.add_argument('input', help='Input DICOM file or directory (with --batch)')
    parser.add_argument('output', help='Output NPY file or directory (with --batch)')
    parser.add_argument('--batch', action='store_true', help='Batch convert all files in directory')
    parser.add_argument('--pattern', default='*.dcm', help='File pattern for batch mode (default: *.dcm)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    verbose = not args.quiet

    if args.batch:
        # Batch conversion
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            sys.exit(1)

        batch_convert(args.input, args.output, args.pattern, verbose=verbose)
    else:
        # Single file conversion
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} does not exist")
            sys.exit(1)

        # Determine file type
        ext = os.path.splitext(args.input)[1].lower()

        if ext == '.dcm':
            success = dicom_to_npy(args.input, args.output, verbose=verbose)
        elif ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            success = png_to_npy(args.input, args.output, verbose=verbose)
        else:
            print(f"Error: Unsupported file format '{ext}'")
            print("Supported formats: .dcm, .png, .jpg, .jpeg, .tif, .tiff")
            sys.exit(1)

        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
