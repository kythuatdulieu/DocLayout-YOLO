#!/bin/bash

# DocLayout-YOLO Conda Environment Setup Script
# This script ensures the 'dla' conda environment is properly set up

set -e

echo "Setting up DocLayout-YOLO conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Create dla environment if it doesn't exist
if ! conda env list | grep -q "^dla "; then
    echo "Creating 'dla' conda environment..."
    conda create -n dla python=3.10 -y
fi

# Activate the environment
echo "Activating 'dla' environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dla

# Install required packages
echo "Installing required packages..."
pip install -e .
pip install huggingface_hub
pip install easyocr  # For OCR functionality
pip install scikit-learn  # For feature processing

echo "Environment setup complete!"
echo "To use the environment, run: conda activate dla"