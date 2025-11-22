#!/bin/bash

# Exit on error
set -e

echo "Setting up environment..."

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created .venv"
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run pipeline
echo "Running pipeline..."
python3 pipeline.py
