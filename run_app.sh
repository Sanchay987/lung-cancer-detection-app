#!/bin/bash

# Lung Cancer CT Scan Analysis - Streamlit App Runner
# This script runs the web interface for CT scan analysis

set -e  # Exit on error

echo "=========================================="
echo "Lung Cancer CT Scan Analysis Web App"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup_macos.sh (macOS) or create venv manually first."
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if model file exists
if [ ! -f "final_novel_attention_model.keras" ]; then
    echo "⚠️  Warning: Model file 'final_novel_attention_model.keras' not found!"
    echo "Please run main.ipynb first to train and save the model."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit>=1.28.0
fi

echo ""
echo "🚀 Starting Streamlit app..."
echo ""
echo "The app will open in your default browser."
echo "If it doesn't open automatically, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Run Streamlit app
streamlit run app.py

echo ""
echo "✅ App stopped."
