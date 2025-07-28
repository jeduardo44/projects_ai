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

def test_extract_features_from_symptoms():
    """Test feature extraction from symptom text"""
    symptoms = "Patient has severe pain and fever"
    features = extract_features_from_data(None, None, None, symptoms)
    
    assert "Pain symptoms" in features, "Pain symptoms should be detected"
    assert "Fever" in features, "Fever should be detected"
    print("✓ test_extract_features_from_symptoms passed")

def test_extract_features_from_images():
    """Test feature extraction from image data"""
    # Mock image objects
    mock_images = [Mock(name="test.jpg"), Mock(name="scan.tiff")]
    features = extract_features_from_data(None, mock_images, None, None)
    
    assert "Medical images available (2 images)" in features, "Image count should be detected"
    assert "Radiographic imaging" in features, "Radiographic imaging should be detected"
    assert "High-resolution imaging" in features, "High-resolution imaging should be detected"
    print("✓ test_extract_features_from_images passed")

def test_extract_features_no_data():
    """Test feature extraction with no input data"""
    features = extract_features_from_data(None, None, None, None)
    
    # Should still return default features
    assert "Patient demographics" in features, "Default features should be present"
    assert "Medical history" in features, "Medical history should be present"
    print("✓ test_extract_features_no_data passed")

def run_all_tests():
    """Run all test functions"""
    print("Running data processor tests...")
    
    try:
        test_extract_features_from_symptoms()
        test_extract_features_from_images()
        test_extract_features_no_data()
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Error running tests: {e}")
        return False
    
    return True

def test_with_pytest():
    """Alternative test runner using pytest-style assertions"""
    def assert_in(item, container, message=""):
        assert item in container, message or f"{item} not found in {container}"
    
    # Test 1: Symptoms
    symptoms = "Patient has severe pain and fever"
    features = extract_features_from_data(None, None, None, symptoms)
    assert_in("Pain symptoms", features, "Pain symptoms should be detected")
    assert_in("Fever", features, "Fever should be detected")
    
    # Test 2: Images
    mock_images = [Mock(name="test.jpg"), Mock(name="scan.tiff")]
    features = extract_features_from_data(None, mock_images, None, None)
    assert_in("Medical images available (2 images)", features)
    assert_in("Radiographic imaging", features)
    assert_in("High-resolution imaging", features)
    
    # Test 3: No data
    features = extract_features_from_data(None, None, None, None)
    assert_in("Patient demographics", features)
    assert_in("Medical history", features)
    
    return True

if __name__ == '__main__':
    # Run function-based tests
    success = run_all_tests()
    
    # Optionally run pytest-style tests
    if success:
        print("\nRunning pytest-style tests...")
        try:
            test_with_pytest()
            print("✅ Pytest-style tests passed!")
        except Exception as e:
            print(f"❌ Pytest-style tests failed: {e}")
    
    sys.exit(0 if success else 1)
