#!/bin/bash

# Setup script for macOS (Apple Silicon)
# Fixes Python version compatibility for TensorFlow

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     macOS Setup - Python 3.10 + TensorFlow                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python 3.10 is available
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo -e "${GREEN}✓ Found Python 3.10${NC}"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo -e "${GREEN}✓ Found Python 3.11${NC}"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
    echo -e "${GREEN}✓ Found Python 3.9${NC}"
else
    echo -e "${RED}✗ Error: Python 3.9, 3.10, or 3.11 required!${NC}"
    echo "Please install Python 3.10 or 3.11:"
    echo "  brew install python@3.10"
    exit 1
fi

# Show Python version
PYTHON_VERSION=$($PYTHON_CMD --version)
echo -e "${BLUE}→ Using: $PYTHON_VERSION${NC}"
echo ""

# Remove old virtual environment if exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}→ Removing old virtual environment...${NC}"
    rm -rf venv
fi

# Create new virtual environment
echo -e "${BLUE}→ Creating new virtual environment with $PYTHON_CMD...${NC}"
$PYTHON_CMD -m venv venv

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to create virtual environment${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Virtual environment created${NC}"
echo ""

# Activate virtual environment
echo -e "${BLUE}→ Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}→ Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# Install TensorFlow for macOS
echo ""
echo -e "${YELLOW}→ Installing TensorFlow for macOS (this may take 3-5 minutes)...${NC}"
pip install tensorflow-macos==2.15.0 --quiet

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ tensorflow-macos installed${NC}"
else
    echo -e "${RED}✗ Failed to install tensorflow-macos${NC}"
    exit 1
fi

pip install tensorflow-metal==1.1.0 --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ tensorflow-metal installed (GPU acceleration)${NC}"
else
    echo -e "${YELLOW}⚠ tensorflow-metal installation failed (GPU acceleration disabled)${NC}"
fi

# Install other required packages
echo ""
echo -e "${BLUE}→ Installing other required packages...${NC}"

pip install --quiet \
    numpy \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    h5py \
    opencv-python \
    pillow \
    jupyter \
    notebook \
    ipykernel \
    ipywidgets \
    tqdm \
    scipy

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All packages installed successfully${NC}"
else
    echo -e "${RED}✗ Some packages failed to install${NC}"
    exit 1
fi

# Verify TensorFlow installation
echo ""
echo -e "${BLUE}→ Verifying TensorFlow installation...${NC}"
python << EOF
import tensorflow as tf
import sys

print(f"✓ TensorFlow version: {tf.__version__}")
print(f"✓ Python version: {sys.version.split()[0]}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU available: Yes ({len(gpus)} device(s))")
else:
    print("⚠ GPU available: No (CPU only)")

print("\nAll core imports successful:")
try:
    import numpy as np
    import sklearn
    import pandas as pd
    import matplotlib.pyplot as plt
    import cv2
    import h5py
    print("✓ numpy, sklearn, pandas, matplotlib, opencv, h5py")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                  ✓ SETUP COMPLETE! ✓                         ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${GREEN}Everything is installed and ready to use!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Activate environment: ${YELLOW}source venv/bin/activate${NC}"
    echo "  2. Start Jupyter: ${YELLOW}jupyter notebook${NC}"
    echo "  3. Open main.ipynb and run cells"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Verification failed. Please check errors above.${NC}"
    exit 1
fi
