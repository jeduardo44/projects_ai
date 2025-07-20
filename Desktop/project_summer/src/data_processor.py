"""
Medical data processing utilities
"""
import tempfile
import pypdf
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_file) -> Optional[str]:
    """Extract text from uploaded PDF file using pypdf"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Extract text using pypdf
        text = ""
        with open(tmp_file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

def extract_features_from_data(file_data, images_data, video_data, symptoms_text):
    """Extract features from uploaded file, images, video, and symptoms text"""
    features = []
    
    # Extract features from symptoms text
    if symptoms_text:
        symptoms_lower = symptoms_text.lower()
        if "pain" in symptoms_lower:
            features.append("Pain symptoms")
        if "fever" in symptoms_lower:
            features.append("Fever")
        if "fatigue" in symptoms_lower:
            features.append("Fatigue")
        if "weight" in symptoms_lower:
            features.append("Weight changes")
        if "blood" in symptoms_lower:
            features.append("Blood-related symptoms")
    
    # Extract features from file (simplified)
    if file_data:
        features.append("File data available")
        features.append("Structured medical data")
    
    # Extract features from images
    if images_data:
        features.append(f"Medical images available ({len(images_data)} images)")
        features.append("Visual analysis data")
        features.append("Image-based features")
        
        # Simulate image analysis features based on disease
        if any(img.name.lower().endswith(('.png', '.jpg', '.jpeg')) for img in images_data):
            features.append("Radiographic imaging")
        if any(img.name.lower().endswith('.tiff') for img in images_data):
            features.append("High-resolution imaging")
    
    # Extract features from video
    if video_data:
        features.append("Medical video available")
        features.append("Temporal analysis data")
        features.append("Video-based features")
        features.append("Motion analysis")
    
    # Add some default features
    features.extend([
        "Patient demographics",
        "Medical history",
        "Current symptoms",
        "Risk factors"
    ])
    
    return features
