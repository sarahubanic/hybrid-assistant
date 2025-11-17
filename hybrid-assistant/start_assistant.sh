#!/bin/bash

# Change to the script's directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing/updating requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Install CUDA-enabled llama-cpp-python if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA support..."
    pip uninstall -y llama-cpp-python
    CMAKE_ARGS="-DGGML_CUDA=ON" pip install --no-cache-dir llama-cpp-python
else
    echo "No NVIDIA GPU detected, using CPU only mode..."
    pip install llama-cpp-python
fi

# Run the application
echo "Starting the assistant..."
python3 src/detection_gui.py