#!/bin/bash

# ============================================
# Lung Nodule Classification - Training Script
# ============================================

echo "╔════════════════════════════════════════╗"
echo "║   Lung Nodule Classification Model    ║"
echo "║          Training Pipeline             ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}✗ Error: Virtual environment not found!${NC}"
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Virtual environment created successfully${NC}"
    else
        echo -e "${RED}✗ Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${BLUE}→ Activating virtual environment...${NC}"
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import tensorflow" 2>/dev/null; then
    echo -e "${YELLOW}→ Installing dependencies (this may take a few minutes)...${NC}"
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install dependencies${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi

# Check if data file exists
echo -e "${BLUE}→ Checking for required data files...${NC}"
if [ ! -f "all_patches.hdf5" ]; then
    echo -e "${RED}✗ Error: all_patches.hdf5 not found!${NC}"
    echo "Please ensure the data file is in the current directory."
    exit 1
else
    DATA_SIZE=$(ls -lh all_patches.hdf5 | awk '{print $5}')
    echo -e "${GREEN}✓ Data file found (${DATA_SIZE})${NC}"
fi

# Display system information
echo ""
echo "╔════════════════════════════════════════╗"
echo "║         System Information             ║"
echo "╚════════════════════════════════════════╝"
python -c "
import tensorflow as tf
import sys
print(f'Python Version: {sys.version.split()[0]}')
print(f'TensorFlow Version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'GPU Available: Yes ({len(gpus)} device(s))')
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu.name}')
else:
    print('GPU Available: No (using CPU)')
print(f'Training Data: 6,691 CT scan patches')
"

echo ""
echo "╔════════════════════════════════════════╗"
echo "║          Starting Training             ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo -e "${YELLOW}The Jupyter notebook will open in your browser.${NC}"
echo -e "${YELLOW}To run training:${NC}"
echo "  1. Click 'Cell → Run All' to execute all cells"
echo "  2. Or press Shift+Enter to run cells one by one"
echo ""
echo -e "${BLUE}Expected training time:${NC}"
echo "  • With GPU: ~10-15 minutes"
echo "  • CPU only: ~30-60 minutes"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop Jupyter when done${NC}"
echo ""

# Run the notebook
jupyter notebook main.ipynb

# Deactivate virtual environment when done
deactivate

echo ""
echo -e "${GREEN}✓ Training session ended${NC}"
