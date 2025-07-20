"""
Unit tests for data processing utilities
"""
import unittest
from unittest.mock import Mock, patch
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import extract_features_from_data

class TestDataProcessor(unittest.TestCase):
    
    def test_extract_features_from_symptoms(self):
        """Test feature extraction from symptom text"""
        symptoms = "Patient has severe pain and fever"
        features = extract_features_from_data(None, None, None, symptoms)
        
        self.assertIn("Pain symptoms", features)
        self.assertIn("Fever", features)
    
    def test_extract_features_from_images(self):
        """Test feature extraction from image data"""
        # Mock image objects
        mock_images = [Mock(name="test.jpg"), Mock(name="scan.tiff")]
        features = extract_features_from_data(None, mock_images, None, None)
        
        self.assertIn("Medical images available (2 images)", features)
        self.assertIn("Radiographic imaging", features)
        self.assertIn("High-resolution imaging", features)
    
    def test_extract_features_no_data(self):
        """Test feature extraction with no input data"""
        features = extract_features_from_data(None, None, None, None)
        
        # Should still return default features
        self.assertIn("Patient demographics", features)
        self.assertIn("Medical history", features)

if __name__ == '__main__':
    unittest.main()
