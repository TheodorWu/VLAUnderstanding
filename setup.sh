#!/bin/bash

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run all unit tests in test folder
python -m unittest discover -s tests -p "test_*.py"
