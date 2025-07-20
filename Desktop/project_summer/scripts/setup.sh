#!/bin/bash

# Development setup script for Medical AI Analyzer
echo "🏥 Setting up Medical AI Analyzer development environment..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your OpenAI API key!"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements-dev.txt

# Run configuration validation
echo "✅ Validating configuration..."
python -c "
import sys
sys.path.insert(0, 'src')
from config_manager import validate_environment
result = validate_environment()
print(f'Configuration valid: {result[\"valid\"]}')
if result['issues']:
    print('Issues found:')
    for issue in result['issues']:
        print(f'  - {issue}')
"

echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your OpenAI API key"
echo "2. Run 'source venv/bin/activate' to activate the environment"
echo "3. Run 'streamlit run app.py' to start the application"
