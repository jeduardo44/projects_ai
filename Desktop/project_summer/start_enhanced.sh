#!/bin/bash

# Medical AI Analyzer - Quick Start Script
echo "🏥 Starting Medical AI Analyzer..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p config
mkdir -p models

# Set default environment variables if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating default .env file..."
    cat > .env << EOL
# OpenAI API Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# App Configuration
DEBUG=False
EOL
    echo "⚠️  Please update .env file with your OpenAI API key if you want to use document analysis features"
fi

echo "🚀 Starting the application..."
streamlit run app.py

echo "📱 App should be running at http://localhost:8501"
echo "🔧 Use the Admin Panel button in the sidebar to manage disease configurations"
