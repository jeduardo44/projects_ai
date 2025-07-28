"""
Database Schema and API Design for Future Implementation
Scalable architecture for medical data management
"""

import sqlite3
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

# Enums remain as classes since they're configuration
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

# Database Connection Function
def get_database_connection(db_path: str = "medical_data.db") -> sqlite3.Connection:
    """Get database connection with proper configuration"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn

# Schema Creation Functions
def create_database_tables(conn: sqlite3.Connection) -> None:
    """Create all necessary database tables"""
    
    # Users table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            license_number TEXT,
            institution TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    """)
    
    # Patients table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            date_of_birth TIMESTAMP,
            gender TEXT,
            phone TEXT,
            email TEXT,
            address TEXT,
            emergency_contact TEXT,
            insurance_info TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            doctor_id INTEGER,
            FOREIGN KEY (doctor_id) REFERENCES users (id)
        )
    """)
    
    # Medical records table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS medical_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            record_type TEXT,
            title TEXT,
            content TEXT,
            file_path TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            FOREIGN KEY (patient_id) REFERENCES patients (id),
            FOREIGN KEY (created_by) REFERENCES users (id)
        )
    """)
    
    # Medical analyses table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS medical_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_type TEXT NOT NULL,
            patient_id INTEGER,
            user_id INTEGER NOT NULL,
            source_record_id INTEGER,
            input_text TEXT,
            input_files TEXT,
            parameters TEXT,
            results TEXT NOT NULL,
            confidence_score REAL,
            risk_score REAL,
            model_version TEXT,
            processing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'completed',
            error_message TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients (id),
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (source_record_id) REFERENCES medical_records (id)
        )
    """)
    
    # Audit logs table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            resource_type TEXT,
            resource_id INTEGER,
            ip_address TEXT,
            user_agent TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            details TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()

# User Management Functions
def create_user(conn: sqlite3.Connection, user_data: Dict[str, Any]) -> int:
    """Create a new user and return the user ID"""
    cursor = conn.execute("""
        INSERT INTO users (username, email, password_hash, role, first_name, 
                          last_name, license_number, institution)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_data['username'], user_data['email'], user_data['password_hash'],
        user_data['role'], user_data.get('first_name'), user_data.get('last_name'),
        user_data.get('license_number'), user_data.get('institution')
    ))
    conn.commit()
    return cursor.lastrowid

def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID"""
    cursor = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    return dict(row) if row else None

def get_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[Dict[str, Any]]:
    """Get user by username"""
    cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    return dict(row) if row else None

def update_user_login_time(conn: sqlite3.Connection, user_id: int) -> None:
    """Update user's last login time"""
    conn.execute(
        "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
        (user_id,)
    )
    conn.commit()

def deactivate_user(conn: sqlite3.Connection, user_id: int) -> None:
    """Deactivate a user account"""
    conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
    conn.commit()

# Patient Management Functions
def create_patient(conn: sqlite3.Connection, patient_data: Dict[str, Any]) -> int:
    """Create a new patient and return the patient ID"""
    cursor = conn.execute("""
        INSERT INTO patients (patient_id, first_name, last_name, date_of_birth,
                            gender, phone, email, address, emergency_contact,
                            insurance_info, doctor_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_data['patient_id'], patient_data['first_name'], patient_data['last_name'],
        patient_data.get('date_of_birth'), patient_data.get('gender'), patient_data.get('phone'),
        patient_data.get('email'), patient_data.get('address'),
        json.dumps(patient_data.get('emergency_contact')),
        json.dumps(patient_data.get('insurance_info')),
        patient_data.get('doctor_id')
    ))
    conn.commit()
    return cursor.lastrowid

def get_patient_by_id(conn: sqlite3.Connection, patient_id: int) -> Optional[Dict[str, Any]]:
    """Get patient by ID"""
    cursor = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    row = cursor.fetchone()
    if row:
        patient = dict(row)
        # Parse JSON fields
        if patient['emergency_contact']:
            patient['emergency_contact'] = json.loads(patient['emergency_contact'])
        if patient['insurance_info']:
            patient['insurance_info'] = json.loads(patient['insurance_info'])
        return patient
    return None

def get_patients_by_doctor(conn: sqlite3.Connection, doctor_id: int) -> List[Dict[str, Any]]:
    """Get all patients for a specific doctor"""
    cursor = conn.execute("SELECT * FROM patients WHERE doctor_id = ?", (doctor_id,))
    patients = []
    for row in cursor.fetchall():
        patient = dict(row)
        if patient['emergency_contact']:
            patient['emergency_contact'] = json.loads(patient['emergency_contact'])
        if patient['insurance_info']:
            patient['insurance_info'] = json.loads(patient['insurance_info'])
        patients.append(patient)
    return patients

def update_patient(conn: sqlite3.Connection, patient_id: int, patient_data: Dict[str, Any]) -> None:
    """Update patient information"""
    # Build dynamic update query based on provided fields
    fields = []
    values = []
    for key, value in patient_data.items():
        if key in ['emergency_contact', 'insurance_info'] and value is not None:
            fields.append(f"{key} = ?")
            values.append(json.dumps(value))
        elif value is not None:
            fields.append(f"{key} = ?")
            values.append(value)
    
    if fields:
        values.append(patient_id)
        query = f"UPDATE patients SET {', '.join(fields)} WHERE id = ?"
        conn.execute(query, values)
        conn.commit()

# Medical Records Functions
def create_medical_record(conn: sqlite3.Connection, record_data: Dict[str, Any]) -> int:
    """Create a new medical record and return the record ID"""
    cursor = conn.execute("""
        INSERT INTO medical_records (patient_id, record_type, title, content,
                                   file_path, metadata, created_by)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        record_data['patient_id'], record_data.get('record_type'),
        record_data.get('title'), record_data.get('content'),
        record_data.get('file_path'), json.dumps(record_data.get('metadata')),
        record_data.get('created_by')
    ))
    conn.commit()
    return cursor.lastrowid

def get_medical_records_by_patient(conn: sqlite3.Connection, patient_id: int) -> List[Dict[str, Any]]:
    """Get all medical records for a patient"""
    cursor = conn.execute("""
        SELECT * FROM medical_records WHERE patient_id = ? ORDER BY created_at DESC
    """, (patient_id,))
    records = []
    for row in cursor.fetchall():
        record = dict(row)
        if record['metadata']:
            record['metadata'] = json.loads(record['metadata'])
        records.append(record)
    return records

def get_medical_record_by_id(conn: sqlite3.Connection, record_id: int) -> Optional[Dict[str, Any]]:
    """Get medical record by ID"""
    cursor = conn.execute("SELECT * FROM medical_records WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    if row:
        record = dict(row)
        if record['metadata']:
            record['metadata'] = json.loads(record['metadata'])
        return record
    return None

# Medical Analysis Functions
def create_medical_analysis(conn: sqlite3.Connection, analysis_data: Dict[str, Any]) -> int:
    """Create a new medical analysis and return the analysis ID"""
    cursor = conn.execute("""
        INSERT INTO medical_analyses (analysis_type, patient_id, user_id, source_record_id,
                                    input_text, input_files, parameters, results,
                                    confidence_score, risk_score, model_version,
                                    processing_time, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        analysis_data['analysis_type'], analysis_data.get('patient_id'),
        analysis_data['user_id'], analysis_data.get('source_record_id'),
        analysis_data.get('input_text'), json.dumps(analysis_data.get('input_files')),
        json.dumps(analysis_data.get('parameters')), json.dumps(analysis_data['results']),
        analysis_data.get('confidence_score'), analysis_data.get('risk_score'),
        analysis_data.get('model_version'), analysis_data.get('processing_time'),
        analysis_data.get('status', 'completed')
    ))
    conn.commit()
    return cursor.lastrowid

def get_medical_analysis_by_id(conn: sqlite3.Connection, analysis_id: int) -> Optional[Dict[str, Any]]:
    """Get medical analysis by ID"""
    cursor = conn.execute("SELECT * FROM medical_analyses WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    if row:
        analysis = dict(row)
        # Parse JSON fields
        if analysis['input_files']:
            analysis['input_files'] = json.loads(analysis['input_files'])
        if analysis['parameters']:
            analysis['parameters'] = json.loads(analysis['parameters'])
        if analysis['results']:
            analysis['results'] = json.loads(analysis['results'])
        return analysis
    return None

def get_analyses_by_user(conn: sqlite3.Connection, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Get all analyses by a specific user"""
    cursor = conn.execute("""
        SELECT * FROM medical_analyses WHERE user_id = ? 
        ORDER BY created_at DESC LIMIT ?
    """, (user_id, limit))
    analyses = []
    for row in cursor.fetchall():
        analysis = dict(row)
        if analysis['input_files']:
            analysis['input_files'] = json.loads(analysis['input_files'])
        if analysis['parameters']:
            analysis['parameters'] = json.loads(analysis['parameters'])
        if analysis['results']:
            analysis['results'] = json.loads(analysis['results'])
        analyses.append(analysis)
    return analyses

def get_analyses_by_patient(conn: sqlite3.Connection, patient_id: int) -> List[Dict[str, Any]]:
    """Get all analyses for a specific patient"""
    cursor = conn.execute("""
        SELECT * FROM medical_analyses WHERE patient_id = ? 
        ORDER BY created_at DESC
    """, (patient_id,))
    analyses = []
    for row in cursor.fetchall():
        analysis = dict(row)
        if analysis['input_files']:
            analysis['input_files'] = json.loads(analysis['input_files'])
        if analysis['parameters']:
            analysis['parameters'] = json.loads(analysis['parameters'])
        if analysis['results']:
            analysis['results'] = json.loads(analysis['results'])
        analyses.append(analysis)
    return analyses

def update_analysis_status(conn: sqlite3.Connection, analysis_id: int, status: str, error_message: str = None) -> None:
    """Update analysis status and error message"""
    if error_message:
        conn.execute("""
            UPDATE medical_analyses SET status = ?, error_message = ? WHERE id = ?
        """, (status, error_message, analysis_id))
    else:
        conn.execute("""
            UPDATE medical_analyses SET status = ? WHERE id = ?
        """, (status, analysis_id))
    conn.commit()

def delete_medical_analysis(conn: sqlite3.Connection, analysis_id: int) -> bool:
    """Delete a medical analysis"""
    cursor = conn.execute("DELETE FROM medical_analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    return cursor.rowcount > 0

# Audit Log Functions
def create_audit_log(conn: sqlite3.Connection, log_data: Dict[str, Any]) -> int:
    """Create a new audit log entry"""
    cursor = conn.execute("""
        INSERT INTO audit_logs (user_id, action, resource_type, resource_id,
                               ip_address, user_agent, details)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        log_data.get('user_id'), log_data['action'], log_data.get('resource_type'),
        log_data.get('resource_id'), log_data.get('ip_address'),
        log_data.get('user_agent'), json.dumps(log_data.get('details'))
    ))
    conn.commit()
    return cursor.lastrowid

def get_audit_logs(conn: sqlite3.Connection, user_id: int = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Get audit logs, optionally filtered by user"""
    if user_id:
        cursor = conn.execute("""
            SELECT * FROM audit_logs WHERE user_id = ? 
            ORDER BY timestamp DESC LIMIT ?
        """, (user_id, limit))
    else:
        cursor = conn.execute("""
            SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
    
    logs = []
    for row in cursor.fetchall():
        log = dict(row)
        if log['details']:
            log['details'] = json.loads(log['details'])
        logs.append(log)
    return logs

def get_audit_logs_by_action(conn: sqlite3.Connection, action: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get audit logs filtered by action type"""
    cursor = conn.execute("""
        SELECT * FROM audit_logs WHERE action = ? 
        ORDER BY timestamp DESC LIMIT ?
    """, (action, limit))
    
    logs = []
    for row in cursor.fetchall():
        log = dict(row)
        if log['details']:
            log['details'] = json.loads(log['details'])
        logs.append(log)
    return logs

# Database Utility Functions
def initialize_database(db_path: str = "medical_data.db") -> sqlite3.Connection:
    """Initialize the database with all tables"""
    conn = get_database_connection(db_path)
    create_database_tables(conn)
    return conn

def backup_database(source_db: str, backup_path: str) -> bool:
    """Create a backup of the database"""
    try:
        source = sqlite3.connect(source_db)
        backup = sqlite3.connect(backup_path)
        source.backup(backup)
        source.close()
        backup.close()
        return True
    except Exception as e:
        print(f"Backup failed: {e}")
        return False

def get_database_statistics(conn: sqlite3.Connection) -> Dict[str, int]:
    """Get basic statistics about the database"""
    stats = {}
    
    # Count records in each table
    tables = ['users', 'patients', 'medical_records', 'medical_analyses', 'audit_logs']
    for table in tables:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cursor.fetchone()[0]
    
    return stats

def search_patients(conn: sqlite3.Connection, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search patients by name or patient ID"""
    search_pattern = f"%{search_term}%"
    cursor = conn.execute("""
        SELECT * FROM patients 
        WHERE first_name LIKE ? OR last_name LIKE ? OR patient_id LIKE ?
        ORDER BY last_name, first_name
        LIMIT ?
    """, (search_pattern, search_pattern, search_pattern, limit))
    
    patients = []
    for row in cursor.fetchall():
        patient = dict(row)
        if patient['emergency_contact']:
            patient['emergency_contact'] = json.loads(patient['emergency_contact'])
        if patient['insurance_info']:
            patient['insurance_info'] = json.loads(patient['insurance_info'])
        patients.append(patient)
    return patients

def get_analysis_statistics(conn: sqlite3.Connection, user_id: int = None) -> Dict[str, Any]:
    """Get analysis statistics, optionally filtered by user"""
    stats = {}
    
    # Base query condition
    where_clause = "WHERE user_id = ?" if user_id else ""
    params = (user_id,) if user_id else ()
    
    # Total analyses
    cursor = conn.execute(f"SELECT COUNT(*) FROM medical_analyses {where_clause}", params)
    stats['total_analyses'] = cursor.fetchone()[0]
    
    # Analyses by type
    cursor = conn.execute(f"""
        SELECT analysis_type, COUNT(*) as count 
        FROM medical_analyses {where_clause}
        GROUP BY analysis_type
    """, params)
    stats['by_type'] = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Analyses by status
    cursor = conn.execute(f"""
        SELECT status, COUNT(*) as count 
        FROM medical_analyses {where_clause}
        GROUP BY status
    """, params)
    stats['by_status'] = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Average confidence score
    cursor = conn.execute(f"""
        SELECT AVG(confidence_score) 
        FROM medical_analyses 
        {where_clause} AND confidence_score IS NOT NULL
    """, params)
    avg_confidence = cursor.fetchone()[0]
    stats['avg_confidence'] = round(avg_confidence, 3) if avg_confidence else None
    
    return stats

# Data Validation Functions
def validate_user_data(user_data: Dict[str, Any]) -> List[str]:
    """Validate user data and return list of validation errors"""
    errors = []
    
    required_fields = ['username', 'email', 'password_hash', 'role']
    for field in required_fields:
        if not user_data.get(field):
            errors.append(f"Missing required field: {field}")
    
    # Validate role
    if user_data.get('role') not in [role.value for role in UserRole]:
        errors.append("Invalid user role")
        
    # Validate email format (basic check)
    email = user_data.get('email', '')
    if email and '@' not in email:
        errors.append("Invalid email format")
    
    return errors

def validate_patient_data(patient_data: Dict[str, Any]) -> List[str]:
    """Validate patient data and return list of validation errors"""
    errors = []
    
    required_fields = ['patient_id', 'first_name', 'last_name']
    for field in required_fields:
        if not patient_data.get(field):
            errors.append(f"Missing required field: {field}")
    
    return errors

def validate_analysis_data(analysis_data: Dict[str, Any]) -> List[str]:
    """Validate analysis data and return list of validation errors"""
    errors = []
    
    required_fields = ['analysis_type', 'user_id', 'results']
    for field in required_fields:
        if not analysis_data.get(field):
            errors.append(f"Missing required field: {field}")
    
    # Validate analysis type
    if analysis_data.get('analysis_type') not in [atype.value for atype in AnalysisType]:
        errors.append("Invalid analysis type")
    
    return errors

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
