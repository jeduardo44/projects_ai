"""
Pydantic models for structured output
"""
from pydantic import BaseModel, Field
from typing import List

class MedicalAnalysis(BaseModel):
    """Structured output for medical document analysis"""
    summary: str = Field(description="Brief summary of the medical data")
    key_findings: List[str] = Field(description="List of key medical findings")
    risk_factors: List[str] = Field(description="List of identified risk factors")
    recommendations: List[str] = Field(description="List of medical recommendations")
    medical_terms: List[str] = Field(description="List of important medical terminology")
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    critical_alerts: List[str] = Field(description="List of critical medical alerts")
    data_quality: str = Field(description="Assessment of data quality: good/medium/poor")

class DiseaseAnalysis(BaseModel):
    """Structured output for disease-specific analysis"""
    disease: str = Field(description="The disease being analyzed")
    symptoms_analyzed: List[str] = Field(description="List of symptoms that were analyzed")
    likelihood: str = Field(description="Likelihood assessment: high/medium/low")
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    key_indicators: List[str] = Field(description="Key indicators for the disease")
    differential_diagnosis: List[str] = Field(description="List of differential diagnoses")
    recommendations: List[str] = Field(description="Medical recommendations")
    urgency_level: str = Field(description="Urgency level: high/medium/low")
    next_steps: List[str] = Field(description="Recommended next steps")
