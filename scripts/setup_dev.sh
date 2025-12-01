#!/bin/bash
# Development environment setup script
# Run this after cloning the repository to set up pre-commit hooks

set -e

echo "Setting up development environment..."

# Check if we're in the project root
if [ ! -f "requirements.txt" ]; then
    echo "Error: Run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "Pre-commit hooks installed. They will run automatically on 'git commit'."
echo "To run hooks manually: pre-commit run --all-files"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"

