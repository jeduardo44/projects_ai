"""
Database Schema and API Design for Future Implementation
Scalable architecture for medical data management
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum

Base = declarative_base()

class UserRole(Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    NURSE = "nurse"
    ADMIN = "admin"
    RESEARCHER = "researcher"

class AnalysisType(Enum):
    CLINICAL_ANALYSIS = "clinical_analysis"
    DISEASE_PREDICTION = "disease_prediction"
    IMAGE_ANALYSIS = "image_analysis"
    LAB_INTERPRETATION = "lab_interpretation"

# Database Models

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    license_number = Column(String(100))  # For medical professionals
    institution = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    analyses = relationship("MedicalAnalysis", back_populates="user")
    patients = relationship("Patient", back_populates="doctor")

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String(100), unique=True, nullable=False)  # Hospital ID
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(DateTime)
    gender = Column(String(20))
    phone = Column(String(20))
    email = Column(String(255))
    address = Column(Text)
    emergency_contact = Column(JSON)
    insurance_info = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    doctor_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    doctor = relationship("User", back_populates="patients")
    medical_records = relationship("MedicalRecord", back_populates="patient")
    analyses = relationship("MedicalAnalysis", back_populates="patient")

class MedicalRecord(Base):
    __tablename__ = "medical_records"
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    record_type = Column(String(100))  # lab_result, imaging, clinical_note, etc.
    title = Column(String(255))
    content = Column(Text)
    file_path = Column(String(500))  # Path to uploaded file
    metadata = Column(JSON)  # Additional structured data
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    patient = relationship("Patient", back_populates="medical_records")
    analyses = relationship("MedicalAnalysis", back_populates="source_record")

class MedicalAnalysis(Base):
    __tablename__ = "medical_analyses"
    
    id = Column(Integer, primary_key=True)
    analysis_type = Column(String(100), nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    source_record_id = Column(Integer, ForeignKey("medical_records.id"))
    
    # Analysis inputs
    input_text = Column(Text)
    input_files = Column(JSON)  # List of file paths
    parameters = Column(JSON)  # Analysis parameters
    
    # Analysis results
    results = Column(JSON, nullable=False)  # Structured analysis results
    confidence_score = Column(Float)
    risk_score = Column(Float)
    
    # Metadata
    model_version = Column(String(100))
    processing_time = Column(Float)  # Seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="completed")  # pending, completed, failed
    error_message = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    patient = relationship("Patient", back_populates="analyses")
    source_record = relationship("MedicalRecord", back_populates="analyses")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(100), nullable=False)  # login, analysis, data_access, etc.
    resource_type = Column(String(100))  # patient, analysis, record
    resource_id = Column(Integer)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(JSON)

# API Design Example

"""
RESTful API Endpoints for Future Implementation

Authentication:
POST /api/auth/login
POST /api/auth/logout
POST /api/auth/refresh
GET  /api/auth/profile

Patient Management:
GET    /api/patients/              # List patients (with pagination)
POST   /api/patients/              # Create new patient
GET    /api/patients/{id}          # Get patient details
PUT    /api/patients/{id}          # Update patient
DELETE /api/patients/{id}          # Delete patient (soft delete)

Medical Records:
GET    /api/patients/{id}/records  # Get patient's medical records
POST   /api/patients/{id}/records  # Upload new medical record
GET    /api/records/{id}           # Get specific record
PUT    /api/records/{id}           # Update record
DELETE /api/records/{id}           # Delete record

Medical Analysis:
POST   /api/analysis/clinical      # Run clinical analysis
POST   /api/analysis/prediction    # Run disease prediction
GET    /api/analysis/{id}          # Get analysis results
GET    /api/analysis/             # List user's analyses
DELETE /api/analysis/{id}          # Delete analysis

File Management:
POST   /api/files/upload           # Upload medical files
GET    /api/files/{id}             # Download file
DELETE /api/files/{id}             # Delete file

Analytics & Reporting:
GET    /api/analytics/usage        # Usage statistics
GET    /api/analytics/performance  # Model performance metrics
GET    /api/reports/patient/{id}   # Generate patient report
GET    /api/reports/analysis/{id}  # Generate analysis report

Administration:
GET    /api/admin/users            # Manage users
POST   /api/admin/users            # Create user
GET    /api/admin/audit            # Audit logs
GET    /api/admin/system           # System health

WebSocket Endpoints (Real-time):
WS     /ws/analysis/{id}           # Real-time analysis progress
WS     /ws/notifications           # Real-time notifications
"""

# Example API Response Schemas

"""
Clinical Analysis Response:
{
    "id": 12345,
    "analysis_type": "clinical_analysis",
    "patient_id": 67890,
    "status": "completed",
    "results": {
        "summary": "Patient shows signs of...",
        "key_findings": ["Finding 1", "Finding 2"],
        "risk_factors": ["Risk 1", "Risk 2"],
        "recommendations": ["Rec 1", "Rec 2"],
        "confidence_score": 0.87
    },
    "metadata": {
        "model_version": "v1.2.3",
        "processing_time": 2.34,
        "created_at": "2025-07-20T10:30:00Z"
    }
}

Disease Prediction Response:
{
    "id": 12346,
    "analysis_type": "disease_prediction",
    "disease": "Diabetes",
    "patient_id": 67890,
    "status": "completed",
    "results": {
        "prediction": "High Risk",
        "confidence_score": 0.92,
        "risk_score": 0.78,
        "key_features": ["BMI", "Glucose", "Family History"],
        "probabilities": {
            "no_diabetes": 0.08,
            "pre_diabetes": 0.14,
            "diabetes": 0.78
        }
    },
    "recommendations": [
        "Consult endocrinologist",
        "Monitor glucose levels",
        "Implement dietary changes"
    ]
}
"""

# Security Considerations for Future Implementation

"""
Security Framework:

1. Authentication & Authorization:
   - JWT tokens with refresh mechanism
   - Role-based access control (RBAC)
   - Multi-factor authentication for sensitive operations
   - API key management for external integrations

2. Data Protection:
   - Encryption at rest (AES-256)
   - Encryption in transit (TLS 1.3)
   - Field-level encryption for sensitive data
   - Secure file storage with access logging

3. Privacy Compliance:
   - HIPAA compliance implementation
   - GDPR data protection measures
   - Data anonymization for research
   - Consent management system

4. Security Monitoring:
   - Comprehensive audit logging
   - Intrusion detection system
   - Rate limiting and DDoS protection
   - Security scanning and vulnerability assessment

5. Access Controls:
   - IP whitelist for admin functions
   - Time-based access restrictions
   - Minimum privilege principle
   - Regular access reviews and cleanup
"""
