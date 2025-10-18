#!/bin/bash

# KissanDial Virtual Environment Activation Script

echo "Activating KissanDial virtual environment..."

# Change to the project directory
cd /Users/sandeeph/Documents/ACM_kissandial/KissanDial

# Activate the virtual environment
source venv/bin/activate

echo "Virtual environment activated!"
echo "Python version: $(python --version)"
echo "You can now run the application with: python app/agent.py"
echo ""
echo "Don't forget to:"
echo "1. Copy .env.template to .env and fill in your API keys"
echo "2. Ensure you have the necessary CSV files in the data directory"
echo ""
echo "To deactivate the virtual environment later, just type: deactivate"
