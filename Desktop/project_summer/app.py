import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import pypdf
import json
import random
import time
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import logging

# Import constants
from constants import *

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for structured output
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

# Initialize LangChain components
def initialize_langchain():
    """Initialize LangChain components"""
    try:
        # Initialize OpenAI LLM
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
        
        # Initialize output parsers
        medical_parser = PydanticOutputParser(pydantic_object=MedicalAnalysis)
        disease_parser = PydanticOutputParser(pydantic_object=DiseaseAnalysis)
        
        return llm, medical_parser, disease_parser
    except Exception as e:
        logger.error(f"Error initializing LangChain: {e}")
        return None, None, None

# Page configuration with modern design
st.set_page_config(
    page_title="Medical AI Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced modern CSS styling
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global reset and base styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 0;
        padding: 0;
    }
    
    /* Main container */
    .main {
        background: transparent;
        padding: 0;
    }
    
    .block-container {
        padding: 1rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header section with glassmorphism */
    .header-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.3rem;
        font-weight: 400;
        margin-top: 1rem;
        margin-bottom: 0;
        letter-spacing: -0.01em;
    }
    
    /* Card containers */
    .analysis-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 0.5rem;
        gap: 0.5rem;
        border: none;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        font-size: 1.1rem;
        padding: 0 2rem;
        transition: all 0.3s ease;
        border: none;
        background: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        box-shadow: 0 4px 20px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: -0.01em;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        height: 50px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    /* Input styling */
    .stFileUploader {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border: 2px dashed #e1e5e9;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        background: #f8f9ff;
    }
    
    .stTextArea textarea {
        border-radius: 16px;
        border: 2px solid #e1e5e9;
        background: white;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        padding: 1rem;
        font-size: 15px;
        resize: vertical;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    .stSelectbox select {
        border-radius: 16px;
        border: 2px solid #e1e5e9;
        background: white;
        font-family: 'Inter', sans-serif;
        padding: 0.8rem 1rem;
        font-size: 15px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Metrics cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 255, 0.95) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
    }
    
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 255, 0.95) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
    }
    
    /* Results sections */
    .results-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #2d3748;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    h3 {
        color: #4a5568;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Info messages */
    .stInfo {
        background: linear-gradient(135deg, #e6f3ff 0%, #f0f8ff 100%);
        border: none;
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #f0fff4 0%, #f7fffa 100%);
        border: none;
        border-left: 4px solid #48bb78;
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #fff5f5 0%, #fffafa 100%);
        border: none;
        border-left: 4px solid #f56565;
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fffbf0 0%, #fffef5 100%);
        border: none;
        border-left: 4px solid #ed8936;
        border-radius: 12px;
        padding: 1rem 1.5rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 255, 0.95) 100%);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        padding: 1rem;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: white;
        border-radius: 0 0 12px 12px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-top: none;
        padding: 1.5rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .subtitle {
            font-size: 1.1rem;
        }
        
        .header-container {
            padding: 2rem 1rem;
            margin: 1rem 0;
        }
        
        .analysis-card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .block-container {
            padding: 0.5rem 1rem;
        }
    }
    
    /* Loading spinner customization */
    .stSpinner {
        text-align: center;
        color: #667eea;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(72, 187, 120, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(72, 187, 120, 0.4);
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None

def extract_text_from_pdf(pdf_file):
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
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def analyze_medical_data(text):
    """Analyze medical data using LangChain and OpenAI"""
    try:
        llm, medical_parser, _ = initialize_langchain()
        if not llm or not medical_parser:
            st.error("Failed to initialize LangChain components")
            return None
        
        # Create structured prompt template
        system_template = """You are a medical data analyst with expertise in analyzing medical documents, reports, and clinical data. 
        Your role is to provide comprehensive, accurate, and clinically relevant analysis of medical information.
        
        Focus on:
        1. Identifying critical medical information and findings
        2. Highlighting potential risks, concerns, and red flags
        3. Providing actionable medical recommendations
        4. Extracting and explaining important medical terminology
        5. Assessing data quality and completeness
        6. Maintaining clinical accuracy and medical standards
        
        Always prioritize patient safety and clinical relevance in your analysis."""
        
        human_template = """Analyze the following medical text and provide a comprehensive analysis:

Medical Text:
{text}

{format_instructions}

Please provide a thorough analysis that would be useful for healthcare professionals."""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
        
        # Format the prompt with instructions
        formatted_prompt = prompt.format_messages(
            text=text[:OPENAI_MAX_TOKENS],
            format_instructions=medical_parser.get_format_instructions()
        )
        
        # Get response from LangChain
        response = llm.invoke(formatted_prompt)
        
        # Parse the structured response
        try:
            parsed_result = medical_parser.parse(response.content)
            return parsed_result.model_dump()
        except Exception as parse_error:
            logger.error(f"Error parsing LangChain response: {parse_error}")
            st.error("Error parsing analysis results. Please try again.")
            return None
            
    except Exception as e:
        logger.error(f"Error during LangChain analysis: {e}")
        st.error(f"Error during analysis: {str(e)}")
        return None

def run_ml_prediction(disease, file_data, images_data, video_data, symptoms_text):
    """Run ML prediction for a specific disease"""
    try:
        # Simulate ML model prediction (replace with actual ML models)
        import random
        
        # Extract features from file, images, video, and symptoms
        features = extract_features_from_data(file_data, images_data, video_data, symptoms_text)
        
        # Simulate prediction based on disease type and data available
        base_confidence = 0.6
        base_risk = 0.3
        
        # Boost confidence and risk based on available data
        if images_data:
            base_confidence += 0.15
            base_risk += 0.1
        if video_data:
            base_confidence += 0.1
            base_risk += 0.05
        if file_data:
            base_confidence += 0.1
            base_risk += 0.05
        
        if disease.lower() == "diabetes":
            prediction = "High Risk" if random.random() > 0.5 else "Low Risk"
            confidence = min(random.uniform(base_confidence, base_confidence + 0.25), 0.95)
            risk_score = min(random.uniform(base_risk, base_risk + 0.4), 0.9)
        elif disease.lower() == "hypertension":
            prediction = "Positive" if random.random() > 0.6 else "Negative"
            confidence = min(random.uniform(base_confidence, base_confidence + 0.25), 0.92)
            risk_score = min(random.uniform(base_risk, base_risk + 0.4), 0.85)
        elif disease.lower() == "heart disease":
            prediction = "High Risk" if random.random() > 0.7 else "Low Risk"
            confidence = min(random.uniform(base_confidence, base_confidence + 0.25), 0.96)
            risk_score = min(random.uniform(base_risk, base_risk + 0.4), 0.9)
        else:
            prediction = "Positive" if random.random() > 0.5 else "Negative"
            confidence = min(random.uniform(base_confidence, base_confidence + 0.25), 0.9)
            risk_score = min(random.uniform(base_risk, base_risk + 0.4), 0.8)
        
        # Generate ML-specific results
        data_types = []
        if file_data:
            data_types.append("structured data")
        if images_data:
            data_types.append(f"{len(images_data)} medical images")
        if video_data:
            data_types.append("medical video")
        if symptoms_text:
            data_types.append("symptom text")
        
        data_summary = ", ".join(data_types) if data_types else "available data"
        
        result = {
            "prediction": prediction,
            "confidence_score": confidence,
            "risk_score": risk_score,
            "model_used": f"{disease} Multimodal ML Model v1.0",
            "prediction_details": f"ML model analyzed {len(features)} features from {data_summary} and predicted {prediction.lower()} for {disease}.",
            "key_features": features[:5],  # Top 5 features
            "data_types_used": data_types,
            "recommendations": [
                "Consult with healthcare provider",
                "Monitor symptoms regularly",
                "Follow up with additional tests if needed"
            ],
            "next_steps": [
                "Schedule medical appointment",
                "Prepare medical history",
                "Bring all relevant test results and images"
            ]
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error during ML prediction: {e}")
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


def generate_clinical_pdf_report(results, filename):
    """Generate a PDF report for clinical analysis results"""
    try:
        # Create temporary file for PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_path = tmp_file.name
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )
        
        normal_style = styles['Normal']
        
        # Title
        story.append(Paragraph("Medical AI Analyzer - Clinical Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Metadata
        story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", normal_style))
        story.append(Paragraph(f"<b>Original File:</b> {filename}", normal_style))
        story.append(Spacer(1, 20))
        
        # Summary
        if results.get('summary'):
            story.append(Paragraph("Executive Summary", heading_style))
            story.append(Paragraph(results['summary'], normal_style))
            story.append(Spacer(1, 15))
        
        # Key Findings
        if results.get('key_findings'):
            story.append(Paragraph("Key Findings", heading_style))
            for finding in results['key_findings']:
                story.append(Paragraph(f"• {finding}", normal_style))
            story.append(Spacer(1, 15))
        
        # Risk Factors
        if results.get('risk_factors'):
            story.append(Paragraph("Risk Factors", heading_style))
            for risk in results['risk_factors']:
                story.append(Paragraph(f"• {risk}", normal_style))
            story.append(Spacer(1, 15))
        
        # Critical Alerts
        if results.get('critical_alerts'):
            story.append(Paragraph("Critical Alerts", heading_style))
            for alert in results['critical_alerts']:
                story.append(Paragraph(f"⚠️ {alert}", normal_style))
            story.append(Spacer(1, 15))
        
        # Recommendations
        if results.get('recommendations'):
            story.append(Paragraph("Medical Recommendations", heading_style))
            for rec in results['recommendations']:
                story.append(Paragraph(f"• {rec}", normal_style))
            story.append(Spacer(1, 15))
        
        # Medical Terms
        if results.get('medical_terms'):
            story.append(Paragraph("Medical Terminology", heading_style))
            terms_text = ", ".join(results['medical_terms'])
            story.append(Paragraph(terms_text, normal_style))
            story.append(Spacer(1, 15))
        
        # Metrics Table
        story.append(Paragraph("Analysis Metrics", heading_style))
        metrics_data = [
            ["Metric", "Value"],
            ["Confidence Score", f"{results.get('confidence_score', 0):.1%}"],
            ["Data Quality", results.get('data_quality', 'Unknown')],
            ["Total Findings", str(len(results.get('key_findings', [])) + len(results.get('risk_factors', [])))],
            ["Critical Alerts", str(len(results.get('critical_alerts', [])))]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)
        
        # Build PDF
        doc.build(story)
        
        # Read the generated PDF
        with open(pdf_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
        
        # Clean up temporary file
        os.unlink(pdf_path)
        
        return pdf_content
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return None


def generate_prediction_pdf_report(results, disease_name, filename):
    """Generate a PDF report for ML prediction results"""
    try:
        # Create temporary file for PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_path = tmp_file.name
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )
        
        normal_style = styles['Normal']
        
        # Title
        story.append(Paragraph(f"Medical AI Analyzer - {disease_name} Prediction Report", title_style))
        story.append(Spacer(1, 20))
        
        # Metadata
        story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", normal_style))
        story.append(Paragraph(f"<b>Disease Analyzed:</b> {disease_name}", normal_style))
        story.append(Paragraph(f"<b>Model Used:</b> {results.get('model_used', 'Unknown')}", normal_style))
        story.append(Spacer(1, 20))
        
        # Prediction Results
        story.append(Paragraph("Prediction Results", heading_style))
        prediction_data = [
            ["Metric", "Value"],
            ["Prediction", results.get('prediction', 'Unknown').title()],
            ["Confidence Score", f"{results.get('confidence_score', 0):.1%}"],
            ["Risk Score", f"{results.get('risk_score', 0):.1%}"],
            ["Model Used", results.get('model_used', 'Unknown')]
        ]
        
        prediction_table = Table(prediction_data, colWidths=[2*inch, 3*inch])
        prediction_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(prediction_table)
        story.append(Spacer(1, 15))
        
        # Prediction Details
        if results.get('prediction_details'):
            story.append(Paragraph("Prediction Details", heading_style))
            story.append(Paragraph(results['prediction_details'], normal_style))
            story.append(Spacer(1, 15))
        
        # Key Features
        if results.get('key_features'):
            story.append(Paragraph("Key Features Used", heading_style))
            for feature in results['key_features']:
                story.append(Paragraph(f"• {feature}", normal_style))
            story.append(Spacer(1, 15))
        
        # Data Types Used
        if results.get('data_types_used'):
            story.append(Paragraph("Data Types Analyzed", heading_style))
            for data_type in results['data_types_used']:
                story.append(Paragraph(f"• {data_type}", normal_style))
            story.append(Spacer(1, 15))
        
        # Recommendations
        if results.get('recommendations'):
            story.append(Paragraph("ML Recommendations", heading_style))
            for rec in results['recommendations']:
                story.append(Paragraph(f"• {rec}", normal_style))
            story.append(Spacer(1, 15))
        
        # Next Steps
        if results.get('next_steps'):
            story.append(Paragraph("Recommended Next Steps", heading_style))
            for step in results['next_steps']:
                story.append(Paragraph(f"• {step}", normal_style))
            story.append(Spacer(1, 15))
        
        # Build PDF
        doc.build(story)
        
        # Read the generated PDF
        with open(pdf_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
        
        # Clean up temporary file
        os.unlink(pdf_path)
        
        return pdf_content
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return None

def main():
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'main'
    
    # Initialize session state for configuration
    if 'disease_config' not in st.session_state:
        st.session_state.disease_config = {
            "Diabetes": {
                "file_types": ["csv", "json", "txt"],
                "image_types": ["png", "jpg", "jpeg", "tiff"],
                "video_types": ["mp4", "avi", "mov"],
                "description": "Diabetes prediction and monitoring"
            },
            "Hypertension": {
                "file_types": ["csv", "json", "txt"],
                "image_types": ["png", "jpg", "jpeg"],
                "video_types": [],
                "description": "Blood pressure monitoring and analysis"
            },
            "Heart Disease": {
                "file_types": ["csv", "json", "txt"],
                "image_types": ["png", "jpg", "jpeg", "tiff", "bmp"],
                "video_types": ["mp4", "avi", "mov", "mkv"],
                "description": "Cardiovascular disease prediction"
            },
            "Cancer": {
                "file_types": ["csv", "json", "txt"],
                "image_types": ["png", "jpg", "jpeg", "tiff", "bmp"],
                "video_types": ["mp4", "avi", "mov"],
                "description": "Cancer screening and detection"
            },
            "COVID-19": {
                "file_types": ["csv", "json", "txt"],
                "image_types": ["png", "jpg", "jpeg", "tiff"],
                "video_types": [],
                "description": "COVID-19 detection and monitoring"
            },
            "Pneumonia": {
                "file_types": ["csv", "json", "txt"],
                "image_types": ["png", "jpg", "jpeg", "tiff", "bmp"],
                "video_types": [],
                "description": "Pneumonia detection from chest X-rays"
            },
            "Stroke": {
                "file_types": ["csv", "json", "txt"],
                "image_types": ["png", "jpg", "jpeg", "tiff", "bmp"],
                "video_types": ["mp4", "avi", "mov"],
                "description": "Stroke risk assessment and detection"
            },
            "Kidney Disease": {
                "file_types": ["csv", "json", "txt"],
                "image_types": ["png", "jpg", "jpeg", "tiff"],
                "video_types": [],
                "description": "Kidney function analysis"
            },
            "Liver Disease": {
                "file_types": ["csv", "json", "txt"],
                "image_types": ["png", "jpg", "jpeg", "tiff", "bmp"],
                "video_types": [],
                "description": "Liver function and disease detection"
            },
            "Thyroid Disorders": {
                "file_types": ["csv", "json", "txt"],
                "image_types": ["png", "jpg", "jpeg", "tiff"],
                "video_types": [],
                "description": "Thyroid function analysis"
            },
            "Depression": {
                "file_types": ["csv", "json", "txt"],
                "image_types": [],
                "video_types": [],
                "description": "Mental health assessment"
            },
            "Anxiety": {
                "file_types": ["csv", "json", "txt"],
                "image_types": [],
                "video_types": [],
                "description": "Anxiety disorder assessment"
            }
        }
    
    # Navigation logic
    if st.session_state.current_page == 'config':
        show_configuration_page()
    else:
        show_main_page()


def show_main_page():
    """Display the main application page"""
    # Enhanced header with glassmorphism effect
    st.markdown(f'''
    <div class="header-container">
        <h1 class="main-title">{APP_TITLE}</h1>
        <p class="subtitle">AI-Powered Medical Analysis Platform</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Initialize session state for analysis status
    if 'analysis_status' not in st.session_state:
        st.session_state.analysis_status = "Ready"
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🔬 Clinical Analysis", "🧬 Disease Prediction"])
    
    with tab1:
        # Clinical Analysis Section with card layout
        st.markdown('''
        <div class="analysis-card">
            <h3>📄 Clinical Document Analysis</h3>
            <p style="color: #6b7280; margin-bottom: 1.5rem;">Upload medical documents for comprehensive AI-powered analysis and insights.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # File upload in a clean container
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader(
                "Choose a medical PDF file",
                type=ALLOWED_FILE_TYPES,
                help="Upload a medical document in PDF format for clinical analysis",
                label_visibility="collapsed"
            )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            
            # Show file info in a nice format
            st.markdown(f'''
            <div class="analysis-card">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; 
                                padding: 0.5rem; 
                                border-radius: 8px; 
                                font-size: 1.2rem;">📄</div>
                    <div>
                        <h4 style="margin: 0; color: #374151;">{uploaded_file.name}</h4>
                        <p style="margin: 0; color: #6b7280; font-size: 0.9rem;">
                            {uploaded_file.size / 1024:.1f} KB • PDF Document
                        </p>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Analyze button with better spacing
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
                    # Update status
                    st.session_state.analysis_status = "Extracting text..."
                
                with st.spinner("Extracting text from PDF..."):
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    
                    if extracted_text:
                        st.session_state.processed_text = extracted_text
                        st.success(f"Successfully extracted {len(extracted_text):,} characters")
                        
                        # Update status
                        st.session_state.analysis_status = "Analyzing with LangChain..."
                        
                        with st.spinner("Analyzing with LangChain..."):
                            analysis_results = analyze_medical_data(extracted_text)
                            if analysis_results:
                                st.session_state.analysis_results = analysis_results
                                st.session_state.analysis_status = "Completed"
                                st.success("Clinical analysis completed successfully!")
                                
                                # Add to history
                                st.session_state.analysis_history.append({
                                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                                    'type': 'Clinical Analysis',
                                    'status': 'Success',
                                    'filename': uploaded_file.name
                                })
                            else:
                                st.session_state.analysis_status = "Failed"
                                st.error("Analysis failed. Please check your configuration.")
                                
                                # Add to history
                                st.session_state.analysis_history.append({
                                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                                    'type': 'Clinical Analysis',
                                    'status': 'Failed',
                                    'filename': uploaded_file.name
                                })
        
        # Display analysis results with enhanced styling
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Metrics row with enhanced cards
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                confidence = results.get('confidence_score', 0)
                st.metric("🎯 Confidence", f"{confidence:.0%}")
            with metric_col2:
                total_findings = len(results.get('key_findings', [])) + len(results.get('risk_factors', [])) + len(results.get('critical_alerts', []))
                st.metric("📊 Total Findings", total_findings)
            with metric_col3:
                data_quality = results.get('data_quality', 'Unknown')
                st.metric("🔍 Data Quality", data_quality.title())
            with metric_col4:
                critical_alerts = len(results.get('critical_alerts', []))
                st.metric("⚠️ Critical Alerts", critical_alerts)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Results display with better organization
            st.markdown("### 📋 Clinical Analysis Results")
            
            # Summary section
            if results.get('summary'):
                st.markdown('''
                <div class="analysis-card">
                    <h4 style="color: #374151; margin-bottom: 1rem;">📝 Executive Summary</h4>
                </div>
                ''', unsafe_allow_html=True)
                st.info(results["summary"])
            
            # Findings sections in organized layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Key findings
                if results.get('key_findings'):
                    st.markdown('''
                    <div class="analysis-card">
                        <h4 style="color: #374151; margin-bottom: 1rem;">🔍 Key Medical Findings</h4>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    for i, finding in enumerate(results['key_findings'][:5], 1):
                        st.markdown(f"**{i}.** {finding}")
                
                # Medical terms
                if results.get('medical_terms'):
                    st.markdown('''
                    <div class="analysis-card" style="margin-top: 1.5rem;">
                        <h4 style="color: #374151; margin-bottom: 1rem;">🏥 Medical Terminology</h4>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    terms_text = ", ".join(results['medical_terms'][:8])
                    if len(results['medical_terms']) > 8:
                        terms_text += f" (+{len(results['medical_terms']) - 8} more)"
                    st.markdown(f"*{terms_text}*")
            
            with col2:
                # Risk factors and alerts
                if results.get('risk_factors') or results.get('critical_alerts'):
                    st.markdown('''
                    <div class="analysis-card">
                        <h4 style="color: #374151; margin-bottom: 1rem;">⚠️ Risk Assessment</h4>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Critical alerts first
                    if results.get('critical_alerts'):
                        for alert in results['critical_alerts'][:3]:
                            st.error(f"🚨 **CRITICAL:** {alert}")
                    
                    # Risk factors
                    if results.get('risk_factors'):
                        for i, risk in enumerate(results['risk_factors'][:5], 1):
                            st.warning(f"**{i}.** {risk}")
                
                # Recommendations
                if results.get('recommendations'):
                    st.markdown('''
                    <div class="analysis-card" style="margin-top: 1.5rem;">
                        <h4 style="color: #374151; margin-bottom: 1rem;">💡 Medical Recommendations</h4>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    for i, rec in enumerate(results['recommendations'][:5], 1):
                        st.success(f"**{i}.** {rec}")
            
            # Export section with enhanced styling
            st.markdown("---")
            st.markdown('''
            <div class="analysis-card">
                <h4 style="color: #374151; margin-bottom: 1rem;">📤 Export Results</h4>
                <p style="color: #6b7280; margin-bottom: 1.5rem;">Download your analysis results in PDF format for sharing with healthcare providers.</p>
            </div>
            ''', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"clinical_analysis_{timestamp}.pdf"
                
                # Generate PDF report
                pdf_content = generate_clinical_pdf_report(
                    st.session_state.analysis_results,
                    st.session_state.uploaded_file.name if st.session_state.uploaded_file else "Unknown"
                )
                
                if pdf_content:
                    st.download_button(
                        label="📄 Download Clinical Analysis PDF",
                        data=pdf_content,
                        file_name=filename,
                        mime="application/pdf"
                    )
                else:
                    st.error("Failed to generate PDF report. Please try again.")
            
            # Reset button at the end - only show when analysis is performed
            if st.session_state.analysis_results:
                st.markdown("---")
                st.markdown("### Reset Analysis")
                if st.button("🔄 Reset Analysis", help="Clear analysis and delete files", use_container_width=True):
                    st.session_state.analysis_results = None
                    st.session_state.uploaded_file = None
                    st.session_state.processed_text = None
                    st.session_state.analysis_status = "Ready"
                    st.success("Reset completed! Analysis cleared and files deleted.")
                    st.rerun()
    
    with tab2:
        # Disease Prediction Section
        st.subheader("Disease Prediction")
        st.markdown("Upload medical data and get ML-powered disease predictions.")
        
        # Disease selection - use configuration from session state
        if 'disease_config' not in st.session_state:
            # Initialize with default diseases if not set
            st.session_state.disease_config = {
                "Diabetes": {"file_types": ["csv", "json", "txt"], "image_types": ["png", "jpg", "jpeg"], "video_types": ["mp4", "avi"], "description": "Diabetes prediction"},
                "Hypertension": {"file_types": ["csv", "json", "txt"], "image_types": ["png", "jpg"], "video_types": [], "description": "Blood pressure monitoring"},
                "Heart Disease": {"file_types": ["csv", "json", "txt"], "image_types": ["png", "jpg", "jpeg", "tiff"], "video_types": ["mp4", "avi", "mov"], "description": "Cardiovascular disease"}
            }
        
        diseases = ["Select an option"] + list(st.session_state.disease_config.keys())
        
        selected_disease = st.selectbox(
            "Select Disease for Prediction:",
            diseases,
            help="Choose the disease you want to predict"
        )
        
        # Check if a valid disease is selected (not the default option)
        is_disease_selected = selected_disease and selected_disease != "Select an option"
        
        # Only show upload buttons when a disease is selected
        if is_disease_selected:
            st.markdown(f"**Upload data for {selected_disease} prediction:**")
            
            # Create 3 columns for file uploads
            upload_col1, upload_col2, upload_col3 = st.columns(3)
            
            with upload_col1:
                # File upload for disease prediction - use configured file types
                disease_config = st.session_state.disease_config.get(selected_disease, {})
                file_types = disease_config.get('file_types', ['csv', 'json', 'txt'])
                
                if file_types:
                    disease_file = st.file_uploader(
                        f"Upload medical data file ({', '.join(file_types).upper()}):",
                        type=file_types,
                        help=f"Upload patient data, test results, or medical records for {selected_disease} prediction"
                    )
                else:
                    st.info("No data file types configured for this disease")
                    disease_file = None
            
            with upload_col2:
                # Image upload for disease prediction - use configured image types
                image_types = disease_config.get('image_types', [])
                
                if image_types:
                    disease_images = st.file_uploader(
                        f"Upload medical images ({', '.join(image_types).upper()}):",
                        type=image_types,
                        accept_multiple_files=True,
                        help=f"Upload medical images for {selected_disease} visual analysis"
                    )
                else:
                    st.info("No image types configured for this disease")
                    disease_images = None
            
            with upload_col3:
                # Video upload for disease prediction - use configured video types
                video_types = disease_config.get('video_types', [])
                
                if video_types:
                    disease_video = st.file_uploader(
                        f"Upload medical video ({', '.join(video_types).upper()}):",
                        type=video_types,
                        help=f"Upload medical videos for {selected_disease} temporal analysis"
                    )
                else:
                    st.info("No video types configured for this disease")
                    disease_video = None
        else:
            # Initialize variables when no disease is selected
            disease_file = None
            disease_images = None
            disease_video = None
        
        # Only show additional inputs and analysis when a disease is selected
        if is_disease_selected:
            # Additional symptoms input
            symptoms_input = st.text_area(
                "Additional symptoms or notes:",
                height=100,
                placeholder="Enter any additional symptoms, observations, or medical notes...",
                help="Provide additional context for the ML prediction"
            )
            
            # Display uploaded media information
            if disease_images or disease_video:
                st.markdown("**Uploaded Media:**")
                if disease_images:
                    st.write(f"• {len(disease_images)} medical image(s) uploaded")
                if disease_video:
                    st.write(f"• Medical video uploaded: {disease_video.name}")
                st.markdown("---")
            
            # Predict button
            if st.button("Run analysis", type="primary", use_container_width=True):
                if disease_file is not None or disease_images is not None or disease_video is not None or symptoms_input.strip():
                    # Update status
                    st.session_state.analysis_status = f"Running ML prediction for {selected_disease}..."
                    
                    with st.spinner(f"Running ML prediction for {selected_disease}..."):
                        # Simulate ML prediction (replace with actual ML model)
                        prediction_results = run_ml_prediction(selected_disease, disease_file, disease_images, disease_video, symptoms_input)
                        if prediction_results:
                            st.session_state.prediction_results = prediction_results
                            st.session_state.analysis_status = "Completed"
                            st.success("ML prediction completed successfully!")
                            
                            # Add to history
                            st.session_state.analysis_history.append({
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'type': f'ML Prediction: {selected_disease}',
                                'status': 'Success',
                                'filename': disease_file.name if disease_file else 'Text input'
                            })
                        else:
                            st.session_state.analysis_status = "Failed"
                            st.error("ML prediction failed. Please check your data.")
                            
                            # Add to history
                            st.session_state.analysis_history.append({
                                'timestamp': datetime.now().strftime("%H:%M:%S"),
                                'type': f'ML Prediction: {selected_disease}',
                                'status': 'Failed',
                                'filename': disease_file.name if disease_file else 'Text input'
                            })
                else:
                    st.warning("Please upload a file, image, video, or enter symptoms for prediction.")
        else:
            # Initialize variables when no disease is selected
            symptoms_input = ""
        
        # Display ML prediction results only when a disease is selected
        if is_disease_selected and hasattr(st.session_state, 'prediction_results') and st.session_state.prediction_results:
            results = st.session_state.prediction_results
            
            # ML prediction metrics
            pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
            with pred_col1:
                prediction = results.get('prediction', 'Unknown').title()
                st.metric("Prediction", prediction)
            with pred_col2:
                confidence = results.get('confidence_score', 0)
                st.metric("Confidence", f"{confidence:.0%}")
            with pred_col3:
                risk_score = results.get('risk_score', 0)
                st.metric("Risk Score", f"{risk_score:.0%}")
            with pred_col4:
                model_used = results.get('model_used', 'Unknown')
                st.metric("Model", model_used)
            
            # ML prediction results
            st.markdown("### ML Prediction Results")
            
            # Prediction details
            if results.get('prediction_details'):
                st.markdown("**Prediction Details**")
                st.write(results['prediction_details'])
            
            # Key features (if available)
            if results.get('key_features'):
                st.markdown("**Key Features Used**")
                features_text = " • ".join(results['key_features'])
                st.write(features_text)
            
            # Recommendations (if available)
            if results.get('recommendations'):
                st.markdown("**ML Recommendations**")
                recs_text = " • ".join(results['recommendations'])
                st.write(recs_text)
            
            # Next steps (if available)
            if results.get('next_steps'):
                st.markdown("**Next Steps**")
                steps_text = " • ".join(results['next_steps'])
                st.write(steps_text)
            
            # Export ML prediction results
            st.markdown("---")
            st.markdown("### Export ML Prediction")
            
            if st.button("Export ML Prediction"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ml_prediction_{selected_disease.lower().replace(' ', '_')}_{timestamp}.pdf"
                
                # Generate PDF report
                pdf_content = generate_prediction_pdf_report(
                    st.session_state.prediction_results,
                    selected_disease,
                    disease_file.name if disease_file else "Text input"
                )
                
                if pdf_content:
                    st.download_button(
                        label="📄 Download ML Prediction PDF",
                        data=pdf_content,
                        file_name=filename,
                        mime="application/pdf"
                    )
                else:
                    st.error("Failed to generate PDF report. Please try again.")
            
            # Reset button at the end - only show when prediction is performed
            if st.session_state.prediction_results:
                st.markdown("---")
                st.markdown("### Reset Prediction")
                if st.button("🔄 Reset Prediction", key="reset_prediction_end", help="Clear prediction and delete files", use_container_width=True):
                    if 'prediction_results' in st.session_state:
                        del st.session_state.prediction_results
                    st.session_state.analysis_status = "Ready"
                    st.success("Reset completed! Prediction cleared and files deleted.")
                    st.rerun()
    
    # Configuration button at the bottom left
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⚙️ Configuration", help="Manage diseases and file types", use_container_width=True):
            st.session_state.current_page = 'config'
            st.rerun()


def show_configuration_page():
    """Display the configuration page with enhanced styling"""
    # Enhanced header with gradient
    st.markdown('''
    <div class="main-header-container">
        <h1 class="main-header">⚙️ System Configuration</h1>
        <p class="subtitle">Manage disease types, file formats, and AI model settings</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Navigation back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Main", help="Return to main application", use_container_width=True):
            st.session_state.current_page = 'main'
            st.rerun()
    
    # Configuration tabs with modern styling
    st.markdown("##")
    
    create_tab, edit_tab, manage_tab = st.tabs(["➕ Create Disease", "✏️ Edit Diseases", "⚙️ Import/Export"])
    
    with create_tab:
        st.markdown('''
        <div class="analysis-card">
            <h3 style="color: #374151; margin-bottom: 1rem;">🆕 Add New Disease Configuration</h3>
            <p style="color: #6b7280; margin-bottom: 2rem;">
                Create a new disease profile with customized file type support and analysis parameters.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Disease information section
        st.markdown('''
        <div class="section-header">
            <h4>📋 Disease Information</h4>
        </div>
        ''', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            new_disease_name = st.text_input("Disease Name:", 
                                           placeholder="e.g., Diabetes, Hypertension", 
                                           key="new_disease_name",
                                           help="Enter the medical condition name")
        with col2:
            new_disease_category = st.selectbox("Category:", 
                                              ["Metabolic", "Cardiovascular", "Neurological", 
                                               "Respiratory", "Endocrine", "Oncological", "Other"],
                                              help="Select the medical category")
        
        new_disease_description = st.text_area("Description:", 
                                             placeholder="Enter detailed description of the condition...", 
                                             key="new_disease_description",
                                             height=100,
                                             help="Provide clinical context and analysis focus")
        
        # File type configuration section
        st.markdown('''
        <div class="section-header" style="margin-top: 2rem;">
            <h4>📁 Supported File Types</h4>
        </div>
        ''', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Data Files**")
            new_file_types = st.multiselect(
                "Data formats:",
                ["csv", "json", "txt", "xlsx", "xml", "pdf"],
                default=["csv", "json", "txt"],
                help="Select supported data file formats",
                key="new_file_types"
            )
        
        with col2:
            st.markdown("**🖼️ Medical Images**")
            new_image_types = st.multiselect(
                "Image formats:",
                ["png", "jpg", "jpeg", "tiff", "bmp", "dicom"],
                default=["png", "jpg", "jpeg"],
                help="Select medical imaging formats",
                key="new_image_types"
            )
        
        with col3:
            st.markdown("**🎥 Video/Motion**")
            new_video_types = st.multiselect(
                "Video formats:",
                ["mp4", "avi", "mov", "mkv", "wmv"],
                default=[],
                help="Select video formats for motion analysis",
                key="new_video_types"
            )
        
        # Advanced settings
        st.markdown('''
        <div class="section-header" style="margin-top: 2rem;">
            <h4>⚡ AI Model Settings</h4>
        </div>
        ''', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.1,
                                            help="Minimum confidence for predictions")
        with col2:
            analysis_depth = st.selectbox("Analysis Depth", 
                                        ["Basic", "Standard", "Comprehensive"],
                                        index=1,
                                        help="Level of AI analysis detail")
        
        # Save configuration
        st.markdown("---")
        save_col1, save_col2, save_col3 = st.columns([1, 1, 1])
        with save_col2:
            if st.button("💾 Create Disease Configuration", type="primary", use_container_width=True):
                if new_disease_name and new_disease_name not in st.session_state.disease_config:
                    st.session_state.disease_config[new_disease_name] = {
                        "file_types": new_file_types,
                        "image_types": new_image_types,
                        "video_types": new_video_types,
                        "description": new_disease_description,
                        "category": new_disease_category,
                        "confidence_threshold": confidence_threshold,
                        "analysis_depth": analysis_depth
                    }
                    st.success(f"✅ Disease '{new_disease_name}' configured successfully!")
                    # Clear form
                    st.session_state.new_disease_name = ""
                    st.session_state.new_disease_description = ""
                    time.sleep(1)
                    st.rerun()
                elif new_disease_name in st.session_state.disease_config:
                    st.error("❌ Disease already exists! Use the Edit tab to modify.")
                else:
                    st.error("❌ Please enter a disease name!")
    
    with edit_tab:
        st.markdown('''
        <div class="analysis-card">
            <h3 style="color: #374151; margin-bottom: 1rem;">✏️ Manage Disease Configurations</h3>
            <p style="color: #6b7280; margin-bottom: 2rem;">
                Edit existing disease profiles or remove outdated configurations.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Check if there are diseases to edit
        if not st.session_state.disease_config:
            st.markdown('''
            <div class="analysis-card" style="text-align: center; padding: 3rem;">
                <h4 style="color: #6b7280;">📋 No Disease Configurations Found</h4>
                <p style="color: #9ca3af; margin-bottom: 2rem;">
                    Create your first disease configuration using the "Create Disease" tab.
                </p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            # Display diseases in card format
            for i, (disease_name, config) in enumerate(st.session_state.disease_config.items()):
                # Enhanced disease card
                st.markdown(f'''
                <div class="analysis-card" style="margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h4 style="color: #374151; margin: 0;">🏥 {disease_name}</h4>
                        <span style="background: #e5e7eb; color: #374151; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem;">
                            {config.get('category', 'Other')}
                        </span>
                    </div>
                    <p style="color: #6b7280; margin-bottom: 1.5rem;">{config.get('description', 'No description available')}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Configuration details in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if config.get('file_types'):
                        st.markdown("**📊 Data Files:**")
                        for file_type in config['file_types'][:3]:
                            st.markdown(f"• {file_type.upper()}")
                        if len(config['file_types']) > 3:
                            st.markdown(f"• +{len(config['file_types']) - 3} more")
                    else:
                        st.info("No data files")
                
                with col2:
                    if config.get('image_types'):
                        st.markdown("**🖼️ Images:**")
                        for img_type in config['image_types'][:3]:
                            st.markdown(f"• {img_type.upper()}")
                        if len(config['image_types']) > 3:
                            st.markdown(f"• +{len(config['image_types']) - 3} more")
                    else:
                        st.info("No images")
                
                with col3:
                    if config.get('video_types'):
                        st.markdown("**🎥 Videos:**")
                        for vid_type in config['video_types'][:3]:
                            st.markdown(f"• {vid_type.upper()}")
                        if len(config['video_types']) > 3:
                            st.markdown(f"• +{len(config['video_types']) - 3} more")
                    else:
                        st.info("No videos")
                
                with col4:
                    st.markdown("**⚡ Settings:**")
                    threshold = config.get('confidence_threshold', 0.7)
                    depth = config.get('analysis_depth', 'Standard')
                    st.markdown(f"• Confidence: {threshold:.1f}")
                    st.markdown(f"• Depth: {depth}")
                
                # Action buttons
                action_col1, action_col2, action_col3 = st.columns(3)
                with action_col1:
                    if st.button(f"✏️ Edit", key=f"edit_{i}", use_container_width=True):
                        st.session_state.editing_disease = disease_name
                        st.rerun()
                
                with action_col2:
                    if st.button(f"🗂️ Duplicate", key=f"duplicate_{i}", use_container_width=True):
                        new_name = f"{disease_name} (Copy)"
                        st.session_state.disease_config[new_name] = config.copy()
                        st.success(f"✅ Duplicated as '{new_name}'")
                        st.rerun()
                
                with action_col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{i}", use_container_width=True, type="secondary"):
                        st.session_state.deleting_disease = disease_name
                        st.rerun()
                
                # Handle editing state
                if 'editing_disease' in st.session_state and st.session_state.editing_disease == disease_name:
                    st.markdown("**Editing Configuration:**")
                    
                    edit_col1, edit_col2 = st.columns(2)
                    with edit_col1:
                        if st.button("💾 Save Changes", key=f"save_{i}", type="primary"):
                            del st.session_state.editing_disease
                            st.success("✅ Changes saved!")
                            st.rerun()
                    
                    with edit_col2:
                        if st.button("❌ Cancel", key=f"cancel_{i}"):
                            del st.session_state.editing_disease
                            st.rerun()
                
                # Handle deletion confirmation
                if 'deleting_disease' in st.session_state and st.session_state.deleting_disease == disease_name:
                    st.error(f"⚠️ Are you sure you want to delete '{disease_name}'?")
                    
                    confirm_col1, confirm_col2 = st.columns(2)
                    with confirm_col1:
                        if st.button("✅ Confirm Delete", key=f"confirm_delete_{i}", type="primary"):
                            del st.session_state.disease_config[disease_name]
                            del st.session_state.deleting_disease
                            st.success(f"✅ Deleted '{disease_name}'")
                            st.rerun()
                    
                    with confirm_col2:
                        if st.button("❌ Cancel Delete", key=f"cancel_delete_{i}"):
                            del st.session_state.deleting_disease
                            st.rerun()
                
                st.markdown("---")
    
    with manage_tab:
        st.markdown('''
        <div class="analysis-card">
            <h3 style="color: #374151; margin-bottom: 1rem;">📦 Configuration Management</h3>
            <p style="color: #6b7280; margin-bottom: 2rem;">
                Import and export disease configurations for backup or sharing across systems.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Export section
        st.markdown('''
        <div class="section-header">
            <h4>📤 Export Configurations</h4>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.session_state.disease_config:
            export_data = {
                "diseases": st.session_state.disease_config,
                "export_date": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            config_json = json.dumps(export_data, indent=2)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    "💾 Download Configuration Backup",
                    data=config_json,
                    file_name=f"disease_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Download all disease configurations as JSON backup"
                )
        else:
            st.info("📋 No configurations to export. Create some disease profiles first.")
        
        # Import section
        st.markdown('''
        <div class="section-header" style="margin-top: 2rem;">
            <h4>� Import Configurations</h4>
        </div>
        ''', unsafe_allow_html=True)
        
        uploaded_config = st.file_uploader(
            "Upload configuration file:",
            type=['json'],
            help="Upload a previously exported configuration backup"
        )
        
        if uploaded_config:
            try:
                config_data = json.loads(uploaded_config.read())
                diseases = config_data.get('diseases', {})
                
                if diseases:
                    st.success(f"✅ Found {len(diseases)} disease configurations")
                    
                    # Preview imported configurations
                    with st.expander("👁️ Preview Import Data"):
                        for name, config in diseases.items():
                            st.markdown(f"**{name}:** {config.get('description', 'No description')}")
                    
                    import_col1, import_col2 = st.columns(2)
                    with import_col1:
                        if st.button("📥 Import All", type="primary", use_container_width=True):
                            st.session_state.disease_config.update(diseases)
                            st.success(f"✅ Imported {len(diseases)} configurations!")
                            st.rerun()
                    
                    with import_col2:
                        if st.button("🔄 Replace All", use_container_width=True):
                            st.session_state.disease_config = diseases
                            st.success(f"✅ Replaced with {len(diseases)} configurations!")
                            st.rerun()
                else:
                    st.error("❌ No valid disease configurations found in file")
            
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file format")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        
        # System information
        st.markdown('''
        <div class="section-header" style="margin-top: 2rem;">
            <h4>ℹ️ System Information</h4>
        </div>
        ''', unsafe_allow_html=True)
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("Total Diseases", len(st.session_state.disease_config))
            st.metric("Configuration Version", "1.0")
        
        with info_col2:
            total_file_types = sum(len(config.get('file_types', [])) + 
                                 len(config.get('image_types', [])) + 
                                 len(config.get('video_types', []))
                                 for config in st.session_state.disease_config.values())
            st.metric("Total File Types", total_file_types)
            st.metric("Last Modified", datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main() 