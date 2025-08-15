#!/bin/bash

# ML Trading Strategy Backtester Setup Script
# This script sets up the environment and launches the Streamlit app

echo "Setting up ML Trading Strategy Backtester..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Installation completed successfully!"
    echo ""
    echo "Ready to launch ML Trading Strategy Backtester"
    echo "Starting Streamlit app..."
    streamlit run app.py
else
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi
