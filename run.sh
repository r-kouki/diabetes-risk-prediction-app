#!/bin/bash
# Quick setup and run script for the Diabetes Risk Prediction Tool

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "diabetes-ml"; then
    echo "Creating conda environment 'diabetes-ml'..."
    conda env create -f environment.yml
else
    echo "Conda environment 'diabetes-ml' already exists."
fi

# Activate environment and run app
echo "Activating environment and starting Streamlit app..."
conda run -n diabetes-ml streamlit run app.py