"""
Configuration validation and management
"""
import os
import sys
from typing import Dict, Any

def validate_environment() -> Dict[str, Any]:
    """Validate required environment variables and configuration"""
    issues = []
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        issues.append("OPENAI_API_KEY environment variable is not set")
    elif api_key.startswith("sk-proj-") and len(api_key) < 50:
        issues.append("OPENAI_API_KEY appears to be invalid or incomplete")
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "api_key_configured": bool(api_key),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }

def get_disease_defaults() -> Dict[str, Any]:
    """Get default disease configurations"""
    return {
        "Diabetes": {
            "file_types": ["csv", "json", "txt"],
            "image_types": ["png", "jpg", "jpeg", "tiff"],
            "video_types": ["mp4", "avi", "mov"],
            "description": "Diabetes prediction and monitoring using glucose levels, HbA1c, and lifestyle factors"
        },
        "Hypertension": {
            "file_types": ["csv", "json", "txt"],
            "image_types": ["png", "jpg", "jpeg"],
            "video_types": [],
            "description": "Blood pressure monitoring and cardiovascular risk assessment"
        },
        "Heart Disease": {
            "file_types": ["csv", "json", "txt"],
            "image_types": ["png", "jpg", "jpeg", "tiff", "bmp"],
            "video_types": ["mp4", "avi", "mov", "mkv"],
            "description": "Cardiovascular disease prediction using ECG, imaging, and clinical data"
        }
    }
