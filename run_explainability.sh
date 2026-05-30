#!/bin/bash

# ================================================
# Lung Nodule Classification - Explainability Script
# Grad-CAM Visualization
# ================================================

echo "╔════════════════════════════════════════╗"
echo "║   Model Explainability (Grad-CAM)      ║"
echo "║      Visual Attention Analysis         ║"
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
    echo "Please run ./run_training.sh first to set up the environment."
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}→ Activating virtual environment...${NC}"
source venv/bin/activate

# Check if required model file exists
echo -e "${BLUE}→ Checking for trained model...${NC}"
if [ ! -f "lung_nodule_classifier_model.h5" ]; then
    echo -e "${YELLOW}⚠ Warning: lung_nodule_classifier_model.h5 not found${NC}"

    # Check if alternative model exists
    if [ -f "final_novel_attention_model.keras" ]; then
        echo -e "${BLUE}→ Found final_novel_attention_model.keras${NC}"
        echo -e "${YELLOW}Converting to H5 format for explainability...${NC}"

        python << EOF
import tensorflow as tf
try:
    model = tf.keras.models.load_model('final_novel_attention_model.keras')
    model.save('lung_nodule_classifier_model.h5')
    print('✓ Model converted successfully')
except Exception as e:
    print(f'✗ Error converting model: {e}')
    exit(1)
EOF

        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Failed to convert model${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ Error: No trained model found!${NC}"
        echo "Please run ./run_training.sh first to train the model."
        exit 1
    fi
else
    echo -e "${GREEN}✓ Model found${NC}"
fi

# Check if data file exists
if [ ! -f "all_patches.hdf5" ]; then
    echo -e "${RED}✗ Error: all_patches.hdf5 not found!${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Data file found${NC}"
fi

# Check OpenCV installation
if ! python -c "import cv2" 2>/dev/null; then
    echo -e "${YELLOW}→ Installing OpenCV for visualization...${NC}"
    pip install opencv-python --quiet
fi

echo ""
echo "╔════════════════════════════════════════╗"
echo "║      Starting Grad-CAM Analysis        ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo -e "${YELLOW}The Jupyter notebook will open in your browser.${NC}"
echo ""
echo -e "${BLUE}What to expect:${NC}"
echo "  • Grad-CAM heatmaps showing model attention"
echo "  • Red/Yellow: High attention areas"
echo "  • Blue/Purple: Low attention areas"
echo "  • 5 sample visualizations from test set"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop Jupyter when done${NC}"
echo ""

# Run the explainability notebook
jupyter notebook Explain.ipynb

# Deactivate virtual environment when done
deactivate

echo ""
echo -e "${GREEN}✓ Explainability session ended${NC}"
