#!/bin/bash

# Demo: Clinical Inference with Sample Images
# This script demonstrates how to use the clinical inference pipeline

echo "=========================================================================="
echo "  CLINICAL INFERENCE DEMO"
echo "=========================================================================="
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "[1/5] Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Error: Virtual environment not found!"
    echo "Run: python3 -m venv venv && source venv/bin/activate"
    exit 1
fi

# Check if model exists
if [ ! -f "final_novel_attention_model.keras" ]; then
    echo "❌ Error: Trained model not found!"
    echo "Please train the model first:"
    echo "  jupyter notebook main.ipynb"
    exit 1
fi

# Check if sample images exist
if [ -d "sample_images" ]; then
    SAMPLE_DIR="sample_images"
    echo "[2/5] Found sample images directory"
else
    echo "[2/5] No sample_images directory, will create test image from dataset"

    # Extract a sample from the dataset
    python3 << 'EOF'
import h5py
import numpy as np
from PIL import Image
import os

os.makedirs('demo_samples', exist_ok=True)

with h5py.File('all_patches.hdf5', 'r') as f:
    images = f['ct_slices'][:]
    labels = f['slice_class'][:]

# Save 3 samples
for i in range(3):
    img = images[i]
    label = labels[i][0]

    # Normalize to 0-255
    img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

    # Save
    Image.fromarray(img_normalized).save(f'demo_samples/sample_{i}_label_{int(label)}.png')

print("✓ Created demo samples in demo_samples/")
EOF

    SAMPLE_DIR="demo_samples"
fi

echo ""
echo "[3/5] Running clinical inference on sample scans..."
echo ""

# Find first image
FIRST_IMAGE=$(find $SAMPLE_DIR -name "*.png" | head -1)

if [ -z "$FIRST_IMAGE" ]; then
    echo "❌ No PNG images found in $SAMPLE_DIR"
    exit 1
fi

echo "Processing: $FIRST_IMAGE"
echo ""

# Run inference
python clinical_inference.py "$FIRST_IMAGE" --output demo_output

echo ""
echo "[4/5] Checking results..."

if [ -f "demo_output/clinical_report.txt" ]; then
    echo "✓ Clinical report generated!"
    echo ""
    echo "=========================================================================="
    echo "  CLINICAL REPORT PREVIEW"
    echo "=========================================================================="
    head -n 30 demo_output/clinical_report.txt
    echo ""
    echo "... (see demo_output/clinical_report.txt for full report)"
else
    echo "❌ Report generation failed"
    exit 1
fi

echo ""
echo "[5/5] Demo complete!"
echo ""
echo "=========================================================================="
echo "  DEMO SUMMARY"
echo "=========================================================================="
echo ""
echo "Files generated:"
echo "  • demo_output/clinical_report.txt    - Text report"
echo "  • demo_output/clinical_report.json   - JSON results"
echo "  • demo_output/clinical_visualization.png - Annotated scan"
echo ""
echo "To view visualization:"
echo "  open demo_output/clinical_visualization.png"
echo ""
echo "To process your own scans:"
echo "  python clinical_inference.py path/to/your/scan.png"
echo ""
echo "Supported formats: DICOM (.dcm), PNG, JPG, TIFF, NumPy (.npy)"
echo ""
echo "=========================================================================="
echo "  ✓ DEMO COMPLETE!"
echo "=========================================================================="
