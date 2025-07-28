"""
Data models and validation functions for structured output
"""
from typing import List, Dict, Any

def create_medical_analysis_template() -> Dict[str, Any]:
    """Create a template for medical document analysis results"""
    return {
        "summary": "",  # Brief summary of the medical data
        "key_findings": [],  # List of key medical findings
        "risk_factors": [],  # List of identified risk factors
        "recommendations": [],  # List of medical recommendations
        "medical_terms": [],  # List of important medical terminology
        "confidence_score": 0.0,  # Confidence score between 0 and 1
        "critical_alerts": [],  # List of critical medical alerts
        "data_quality": ""  # Assessment of data quality: good/medium/poor
    }

def validate_medical_analysis(data: Dict[str, Any]) -> List[str]:
    """Validate medical analysis data and return list of validation errors"""
    errors = []
    template = create_medical_analysis_template()
    
    # Check required fields
    for field in template.keys():
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate data types
    if "confidence_score" in data:
        if not isinstance(data["confidence_score"], (int, float)) or not (0 <= data["confidence_score"] <= 1):
            errors.append("confidence_score must be a number between 0 and 1")
    
    if "data_quality" in data:
        valid_qualities = ["good", "medium", "poor"]
        if data["data_quality"] not in valid_qualities:
            errors.append(f"data_quality must be one of: {valid_qualities}")
    
    # Validate list fields
    list_fields = ["key_findings", "risk_factors", "recommendations", "medical_terms", "critical_alerts"]
    for field in list_fields:
        if field in data and not isinstance(data[field], list):
            errors.append(f"{field} must be a list")
    
    return errors

def create_disease_analysis_template() -> Dict[str, Any]:
    """Create a template for disease-specific analysis results"""
    return {
        "disease": "",  # The disease being analyzed
        "symptoms_analyzed": [],  # List of symptoms that were analyzed
        "likelihood": "",  # Likelihood assessment: high/medium/low
        "confidence_score": 0.0,  # Confidence score between 0 and 1
        "key_indicators": [],  # Key indicators for the disease
        "differential_diagnosis": [],  # List of differential diagnoses
        "recommendations": [],  # Medical recommendations
        "urgency_level": "",  # Urgency level: high/medium/low
        "next_steps": []  # Recommended next steps
    }

def validate_disease_analysis(data: Dict[str, Any]) -> List[str]:
    """Validate disease analysis data and return list of validation errors"""
    errors = []
    template = create_disease_analysis_template()
    
    # Check required fields
    for field in template.keys():
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate data types
    if "confidence_score" in data:
        if not isinstance(data["confidence_score"], (int, float)) or not (0 <= data["confidence_score"] <= 1):
            errors.append("confidence_score must be a number between 0 and 1")
    
    # Validate enum fields
    if "likelihood" in data:
        valid_likelihoods = ["high", "medium", "low"]
        if data["likelihood"] not in valid_likelihoods:
            errors.append(f"likelihood must be one of: {valid_likelihoods}")
    
    if "urgency_level" in data:
        valid_urgencies = ["high", "medium", "low"]
        if data["urgency_level"] not in valid_urgencies:
            errors.append(f"urgency_level must be one of: {valid_urgencies}")
    
    # Validate list fields
    list_fields = ["symptoms_analyzed", "key_indicators", "differential_diagnosis", "recommendations", "next_steps"]
    for field in list_fields:
        if field in data and not isinstance(data[field], list):
            errors.append(f"{field} must be a list")
    
    return errors

def create_medical_analysis(summary: str = "", key_findings: List[str] = None, 
                          risk_factors: List[str] = None, recommendations: List[str] = None,
                          medical_terms: List[str] = None, confidence_score: float = 0.0,
                          critical_alerts: List[str] = None, data_quality: str = "medium") -> Dict[str, Any]:
    """Create a medical analysis dictionary with the provided data"""
    return {
        "summary": summary,
        "key_findings": key_findings or [],
        "risk_factors": risk_factors or [],
        "recommendations": recommendations or [],
        "medical_terms": medical_terms or [],
        "confidence_score": confidence_score,
        "critical_alerts": critical_alerts or [],
        "data_quality": data_quality
    }

def create_disease_analysis(disease: str = "", symptoms_analyzed: List[str] = None,
                          likelihood: str = "medium", confidence_score: float = 0.0,
                          key_indicators: List[str] = None, differential_diagnosis: List[str] = None,
                          recommendations: List[str] = None, urgency_level: str = "medium",
                          next_steps: List[str] = None) -> Dict[str, Any]:
    """Create a disease analysis dictionary with the provided data"""
    return {
        "disease": disease,
        "symptoms_analyzed": symptoms_analyzed or [],
        "likelihood": likelihood,
        "confidence_score": confidence_score,
        "key_indicators": key_indicators or [],
        "differential_diagnosis": differential_diagnosis or [],
        "recommendations": recommendations or [],
        "urgency_level": urgency_level,
        "next_steps": next_steps or []
    }
