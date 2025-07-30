#!/bin/bash

# Medical AI Analyzer - Backoffice Only
echo "🔧 Starting Medical AI Analyzer - Backoffice Panel..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p config

echo "🚀 Starting the backoffice application..."
streamlit run backoffice.py --server.port 8502

echo "📱 Backoffice should be running at http://localhost:8502"
