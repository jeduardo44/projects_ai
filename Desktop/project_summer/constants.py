"""
Constants and configuration for the Medical AI Analyzer
"""

# OpenAI Configuration
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.1
OPENAI_MAX_TOKENS = 4000

# App Configuration
APP_TITLE = "Medical AI Analyzer"
EXPORT_VERSION = "1.0"
ALLOWED_FILE_TYPES = ['pdf'] 