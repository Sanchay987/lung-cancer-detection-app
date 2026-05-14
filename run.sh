#!/bin/bash

# ================================================
# Lung Nodule Classification - Main Launcher
# ================================================

clear

echo "╔════════════════════════════════════════════════════╗"
echo "║                                                    ║"
echo "║    Lung Nodule Classification System              ║"
echo "║    Deep Learning with Explainability              ║"
echo "║                                                    ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Display menu
echo -e "${CYAN}What would you like to do?${NC}"
echo ""
echo "  1) Train the model (main.ipynb)"
echo "  2) Run explainability analysis (Explain.ipynb)"
echo "  3) View project information"
echo "  4) Setup/Install dependencies"
echo "  5) Exit"
echo ""
echo -n "Enter your choice [1-5]: "
read choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}Launching training pipeline...${NC}"
        sleep 1
        ./run_training.sh
        ;;
    2)
        echo ""
        echo -e "${GREEN}Launching explainability analysis...${NC}"
        sleep 1
        ./run_explainability.sh
        ;;
    3)
        echo ""
        echo "╔════════════════════════════════════════════════════╗"
        echo "║              Project Information                   ║"
        echo "╚════════════════════════════════════════════════════╝"
        echo ""
        echo -e "${BLUE}Project:${NC} Lung Nodule Classification"
        echo -e "${BLUE}Model:${NC} MobileNetV2 + Channel Attention"
        echo -e "${BLUE}Task:${NC} Binary Classification (Benign vs Malignant)"
        echo ""
        echo -e "${BLUE}Dataset:${NC}"
        echo "  • Total Samples: 6,691 CT scan patches"
        echo "  • Training: 4,683 | Validation: 1,004 | Test: 1,004"
        echo "  • Data File: all_patches.hdf5 (62 MB)"
        echo ""
        echo -e "${BLUE}Performance:${NC}"
        echo "  • Test Accuracy: 93.23%"
        echo "  • Test AUC-ROC: 0.9728"
        echo "  • Training Time: 10-60 minutes (GPU/CPU)"
        echo ""
        echo -e "${BLUE}Files:${NC}"
        echo "  • main.ipynb - Training pipeline"
        echo "  • Explain.ipynb - Grad-CAM explainability"
        echo "  • README.md - Complete usage guide"
        echo "  • PROJECT.md - Technical documentation"
        echo ""
        echo -e "${YELLOW}For detailed documentation, see:${NC}"
        echo "  • README.md - How to run"
        echo "  • PROJECT.md - Full project details"
        echo ""
        ;;
    4)
        echo ""
        echo -e "${GREEN}Setting up environment...${NC}"
        echo ""

        # Create virtual environment if needed
        if [ ! -d "venv" ]; then
            echo -e "${BLUE}→ Creating virtual environment...${NC}"
            python3 -m venv venv
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ Virtual environment created${NC}"
            else
                echo -e "${RED}✗ Failed to create virtual environment${NC}"
                exit 1
            fi
        else
            echo -e "${GREEN}✓ Virtual environment already exists${NC}"
        fi

        # Activate and install dependencies
        source venv/bin/activate
        echo -e "${BLUE}→ Installing dependencies...${NC}"
        pip install --upgrade pip --quiet
        pip install -r requirements.txt

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
            echo ""
            echo "Installed packages:"
            pip list | grep -E "(tensorflow|numpy|scikit-learn|matplotlib|opencv-python|jupyter)"
        else
            echo -e "${RED}✗ Failed to install dependencies${NC}"
        fi

        deactivate
        echo ""
        echo -e "${GREEN}Setup complete! You can now run options 1 or 2.${NC}"
        echo ""
        ;;
    5)
        echo ""
        echo -e "${CYAN}Thank you for using the Lung Nodule Classification System!${NC}"
        echo ""
        exit 0
        ;;
    *)
        echo ""
        echo -e "${RED}Invalid choice. Please run the script again and select 1-5.${NC}"
        echo ""
        exit 1
        ;;
esac
