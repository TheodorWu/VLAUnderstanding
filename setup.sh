#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
pip install -r requirements.txt

# Run all unit tests in test folder
echo "Running unit tests..."
python -m unittest discover -s tests -p "test_*.py"

echo "Setup complete!"
