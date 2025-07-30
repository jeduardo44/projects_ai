"""
Disease Configuration Module
Manages disease types, accepted file formats, and parameters for the medical analyzer
"""

import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class DiseaseConfig:
    """Configuration for a specific disease prediction model."""
    name: str
    display_name: str
    description: str
    accepted_formats: List[str]
    parameters: Dict[str, Any]  # Input parameters required for this disease
    model_path: str
    is_active: bool = True

# Default disease configurations
DEFAULT_DISEASES = {
    "diabetes": DiseaseConfig(
        name="diabetes",
        display_name="Diabetes Risk",
        description="Predict diabetes risk based on patient data",
        accepted_formats=["csv", "json"],
        parameters={
            "age": {"type": "number", "min": 18, "max": 120, "default": 45, "label": "Age"},
            "bmi": {"type": "number", "min": 15.0, "max": 50.0, "default": 25.0, "step": 0.1, "label": "BMI"},
            "glucose_level": {"type": "number", "min": 70, "max": 300, "default": 100, "label": "Glucose Level (mg/dL)"},
            "blood_pressure": {"type": "number", "min": 80, "max": 200, "default": 120, "label": "Systolic BP (mmHg)"},
            "insulin_level": {"type": "number", "min": 0, "max": 200, "default": 80, "label": "Insulin Level (μU/mL)"},
            "family_history": {"type": "checkbox", "default": False, "label": "Family History of Diabetes"},
            "physical_activity": {"type": "selectbox", "options": [1, 2, 3, 4, 5], "default": 3, "label": "Physical Activity Level", "help": "1=Very Low, 5=Very High"},
            "diet_score": {"type": "selectbox", "options": [1, 2, 3, 4, 5], "default": 3, "label": "Diet Quality Score", "help": "1=Poor, 5=Excellent"},
            "stress_level": {"type": "selectbox", "options": [1, 2, 3, 4, 5], "default": 3, "label": "Stress Level", "help": "1=Very Low, 5=Very High"}
        },
        model_path="models/diabetes_model.pkl"
    ),
    "heart_disease": DiseaseConfig(
        name="heart_disease",
        display_name="Heart Disease Risk",
        description="Predict heart disease risk (Placeholder - Coming Soon)",
        accepted_formats=["csv", "pdf", "json"],
        parameters={
            "age": {"type": "number", "min": 18, "max": 120, "default": 50, "label": "Age"},
            "cholesterol": {"type": "number", "min": 100, "max": 400, "default": 200, "label": "Cholesterol Level (mg/dL)"},
            "blood_pressure": {"type": "number", "min": 80, "max": 200, "default": 120, "label": "Systolic BP (mmHg)"},
            "smoking": {"type": "checkbox", "default": False, "label": "Smoking History"},
            "exercise_hours": {"type": "number", "min": 0, "max": 20, "default": 3, "label": "Exercise Hours per Week"}
        },
        model_path="models/heart_disease_model.pkl"
    ),
    "lung_disease": DiseaseConfig(
        name="lung_disease",
        display_name="Lung Disease Risk",
        description="Predict lung disease risk from medical images/data (Placeholder - Coming Soon)",
        accepted_formats=["pdf", "jpg", "png", "dicom", "mp4"],
        parameters={
            "age": {"type": "number", "min": 18, "max": 120, "default": 45, "label": "Age"},
            "smoking_years": {"type": "number", "min": 0, "max": 60, "default": 0, "label": "Years of Smoking"},
            "exposure_chemicals": {"type": "checkbox", "default": False, "label": "Chemical Exposure History"},
            "family_history": {"type": "checkbox", "default": False, "label": "Family History of Lung Disease"}
        },
        model_path="models/lung_disease_model.pkl"
    ),
    "cancer_screening": DiseaseConfig(
        name="cancer_screening",
        display_name="Cancer Screening Analysis",
        description="Analyze medical images for cancer screening (Placeholder - Coming Soon)",
        accepted_formats=["jpg", "png", "dicom", "pdf", "mp4"],
        parameters={
            "age": {"type": "number", "min": 18, "max": 120, "default": 55, "label": "Age"},
            "gender": {"type": "selectbox", "options": ["Male", "Female", "Other"], "default": "Female", "label": "Gender"},
            "family_history": {"type": "checkbox", "default": False, "label": "Family History of Cancer"},
            "screening_type": {"type": "selectbox", "options": ["Breast", "Cervical", "Colon", "Lung", "Prostate"], "default": "Breast", "label": "Screening Type"}
        },
        model_path="models/cancer_screening_model.pkl"
    )
}

class DiseaseConfigManager:
    """Manages disease configurations for the medical analyzer."""
    
    def __init__(self, config_file: str = "config/diseases.json"):
        self.config_file = Path(config_file)
        self.diseases = {}
        self.load_configurations()
    
    def load_configurations(self):
        """Load disease configurations from file or use defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.diseases = {
                        name: DiseaseConfig(**config) 
                        for name, config in data.items()
                    }
            except Exception as e:
                print(f"Error loading disease config: {e}. Using defaults.")
                self.diseases = DEFAULT_DISEASES.copy()
        else:
            self.diseases = DEFAULT_DISEASES.copy()
            self.save_configurations()
    
    def save_configurations(self):
        """Save current configurations to file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            data = {name: asdict(config) for name, config in self.diseases.items()}
            json.dump(data, f, indent=2)
    
    def get_active_diseases(self) -> Dict[str, DiseaseConfig]:
        """Get all active disease configurations."""
        return {name: config for name, config in self.diseases.items() if config.is_active}
    
    def get_disease_config(self, disease_name: str) -> DiseaseConfig:
        """Get configuration for a specific disease."""
        return self.diseases.get(disease_name)
    
    def add_disease(self, disease_config: DiseaseConfig):
        """Add a new disease configuration."""
        self.diseases[disease_config.name] = disease_config
        self.save_configurations()
    
    def update_disease(self, disease_name: str, disease_config: DiseaseConfig):
        """Update an existing disease configuration."""
        if disease_name in self.diseases:
            self.diseases[disease_name] = disease_config
            self.save_configurations()
    
    def delete_disease(self, disease_name: str):
        """Delete a disease configuration."""
        if disease_name in self.diseases:
            del self.diseases[disease_name]
            self.save_configurations()
    
    def toggle_disease_status(self, disease_name: str):
        """Toggle active status of a disease."""
        if disease_name in self.diseases:
            self.diseases[disease_name].is_active = not self.diseases[disease_name].is_active
            self.save_configurations()

# Global instance
disease_manager = DiseaseConfigManager()
