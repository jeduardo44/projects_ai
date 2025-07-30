"""
Medical AI Analyzer - Streamlined Streamlit Application
A clean, Pythonic medical document analysis and disease prediction app.
"""

import streamlit as st
import os
import tempfile
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Third-party imports
import pypdf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Local imports
from constants import *
from disease_config import disease_manager, DiseaseConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class AnalysisResult:
    """Data class for analysis results."""
    summary: str
    key_findings: List[str]
    risk_factors: List[str]
    recommendations: List[str]
    confidence_score: float
    timestamp: datetime


@dataclass
class PatientData:
    """Data class for patient information."""
    age: int
    bmi: float
    glucose_level: int
    blood_pressure: int
    insulin_level: int
    family_history: bool
    physical_activity: int  # 1-5 scale
    diet_score: int  # 1-5 scale
    stress_level: int  # 1-5 scale


class MedicalAnalyzer:
    """Core medical analysis functionality."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = self._initialize_llm()
        self.ml_models = self._load_ml_models()
    
    def _initialize_llm(self) -> Optional[object]:
        """Initialize OpenAI LLM if API key is available."""
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            logger.warning("OpenAI API key not configured")
            return None
        
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=OPENAI_MODEL,
                temperature=OPENAI_TEMPERATURE,
                api_key=self.api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    def _load_ml_models(self) -> bool:
        """Load ML models for disease prediction."""
        try:
            from ml_models import predict_diabetes, generate_diabetes_recommendations
            self.predict_diabetes = predict_diabetes
            self.generate_recommendations = generate_diabetes_recommendations
            return True
        except ImportError as e:
            logger.error(f"ML models not available: {e}")
            return False
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            pdf_reader = pypdf.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def analyze_medical_document(self, text: str) -> AnalysisResult:
        """Analyze medical document text using AI."""
        if not self.llm:
            return self._create_fallback_analysis(text)
        
        try:
            prompt = f"""
            Analyze the following medical document and provide:
            1. A concise summary (2-3 sentences)
            2. Key medical findings (list)
            3. Risk factors identified (list)
            4. Recommendations (list)
            5. Confidence score (0.0-1.0)
            
            Medical Document:
            {text[:2000]}  # Limit text length
            
            Please respond in JSON format.
            """
            
            response = self.llm.invoke(prompt)
            return self._parse_analysis_response(response.content)
        
        except Exception as e:
            logger.error(f"Error in document analysis: {e}")
            return self._create_fallback_analysis(text)
    
    def _parse_analysis_response(self, response: str) -> AnalysisResult:
        """Parse LLM response into AnalysisResult."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return AnalysisResult(
                    summary=data.get('summary', 'Analysis completed'),
                    key_findings=data.get('key_findings', []),
                    risk_factors=data.get('risk_factors', []),
                    recommendations=data.get('recommendations', []),
                    confidence_score=data.get('confidence_score', 0.8),
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
        
        return self._create_fallback_analysis(response)
    
    def _create_fallback_analysis(self, text: str) -> AnalysisResult:
        """Create fallback analysis when LLM is not available."""
        return AnalysisResult(
            summary=f"Document processed. Content length: {len(text)} characters.",
            key_findings=["Document text extracted successfully"],
            risk_factors=["Manual review recommended"],
            recommendations=["Please review document manually", "Consider consulting medical professional"],
            confidence_score=0.5,
            timestamp=datetime.now()
        )
    
    def predict_disease_risk(self, patient_data: PatientData) -> Dict:
        """Predict disease risk using ML models."""
        if not self.ml_models:
            return self._create_fallback_prediction(patient_data)
        
        try:
            # Convert to format expected by ML model
            data_dict = {
                'age': patient_data.age,
                'bmi': patient_data.bmi,
                'glucose_level': patient_data.glucose_level,
                'blood_pressure': patient_data.blood_pressure,
                'insulin_level': patient_data.insulin_level,
                'family_history': int(patient_data.family_history),
                'physical_activity': patient_data.physical_activity,
                'diet_score': patient_data.diet_score,
                'stress_level': patient_data.stress_level
            }
            
            prediction_result = self.predict_diabetes(data_dict)
            recommendations = self.generate_recommendations(prediction_result, data_dict)
            
            return {
                'prediction': prediction_result.get('prediction', 'Unknown'),
                'risk_score': prediction_result.get('risk_score', 0.0),
                'confidence': prediction_result.get('confidence', 0.0),
                'recommendations': recommendations
            }
        
        except Exception as e:
            logger.error(f"Error in disease prediction: {e}")
            return self._create_fallback_prediction(patient_data)
    
    def _create_fallback_prediction(self, patient_data: PatientData) -> Dict:
        """Create fallback prediction when ML models are not available."""
        # Simple rule-based prediction
        risk_score = 0.0
        if patient_data.age > 45:
            risk_score += 0.2
        if patient_data.bmi > 25:
            risk_score += 0.3
        if patient_data.glucose_level > 126:
            risk_score += 0.4
        if patient_data.family_history:
            risk_score += 0.1
        
        risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
        
        return {
            'prediction': f"{risk_level} Risk",
            'risk_score': min(risk_score, 1.0),
            'confidence': 0.6,
            'recommendations': [
                "Maintain healthy diet",
                "Regular exercise recommended",
                "Monitor blood glucose levels",
                "Consult healthcare provider"
            ]
        }


class ReportGenerator:
    """Generate PDF reports for analysis results."""
    
    @staticmethod
    def create_pdf_report(analysis: AnalysisResult, filename: str = "medical_analysis_report.pdf") -> bytes:
        """Generate PDF report from analysis results."""
        try:
            buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            doc = SimpleDocTemplate(buffer.name, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph("Medical Analysis Report", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Timestamp
            timestamp = Paragraph(f"Generated: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
            story.append(timestamp)
            story.append(Spacer(1, 12))
            
            # Summary
            story.append(Paragraph("Summary", styles['Heading2']))
            story.append(Paragraph(analysis.summary, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Key Findings
            if analysis.key_findings:
                story.append(Paragraph("Key Findings", styles['Heading2']))
                for finding in analysis.key_findings:
                    story.append(Paragraph(f"• {finding}", styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Risk Factors
            if analysis.risk_factors:
                story.append(Paragraph("Risk Factors", styles['Heading2']))
                for risk in analysis.risk_factors:
                    story.append(Paragraph(f"• {risk}", styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Recommendations
            if analysis.recommendations:
                story.append(Paragraph("Recommendations", styles['Heading2']))
                for rec in analysis.recommendations:
                    story.append(Paragraph(f"• {rec}", styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Confidence Score
            story.append(Paragraph(f"Confidence Score: {analysis.confidence_score:.2f}", styles['Normal']))
            
            doc.build(story)
            
            with open(buffer.name, 'rb') as f:
                pdf_content = f.read()
            
            os.unlink(buffer.name)
            return pdf_content
        
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return b""


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Medical AI Analyzer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
        .header-container {
            text-align: center;
            margin: 2rem 0 3rem 0;
            padding: 0;
        }
        
        .main-title {
            color: #1f2937;
            font-size: 3.5rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.025em;
            line-height: 1.1;
        }
        
        .subtitle {
            color: #6b7280;
            font-size: 1.25rem;
            font-weight: 400;
            margin: 0.75rem 0 0 0;
            letter-spacing: 0.025em;
        }
        
        .metric-card {
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the application header."""
    st.markdown(f'''
    <div class="header-container">
        <h1 class="main-title">{APP_TITLE}</h1>
        <p class="subtitle">AI-Powered Medical Analysis Platform</p>
    </div>
    ''', unsafe_allow_html=True)


def render_document_analysis_tab(analyzer: MedicalAnalyzer):
    """Render the document analysis tab."""
    st.markdown("### Clinical Document Analysis")
    st.markdown("Upload medical documents for AI-powered analysis and insights.")
    
    uploaded_file = st.file_uploader(
        "Choose a medical PDF file",
        type=ALLOWED_FILE_TYPES,
        help="Upload a medical document in PDF format"
    )
    
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            text = analyzer.extract_text_from_pdf(uploaded_file)
        
        if text:
            st.success(f"Text extracted successfully ({len(text)} characters)")
            
            if st.button("Analyze Document", type="primary"):
                with st.spinner("Analyzing document..."):
                    analysis = analyzer.analyze_medical_document(text)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Summary")
                    st.write(analysis.summary)
                    
                    if analysis.key_findings:
                        st.markdown("#### Key Findings")
                        for finding in analysis.key_findings:
                            st.write(f"• {finding}")
                
                with col2:
                    if analysis.risk_factors:
                        st.markdown("#### Risk Factors")
                        for risk in analysis.risk_factors:
                            st.write(f"• {risk}")
                    
                    if analysis.recommendations:
                        st.markdown("#### Recommendations")
                        for rec in analysis.recommendations:
                            st.write(f"• {rec}")
                
                # Confidence score
                st.metric("Confidence Score", f"{analysis.confidence_score:.2f}")
                
                # Generate report
                if st.button("Generate PDF Report"):
                    report_generator = ReportGenerator()
                    pdf_content = report_generator.create_pdf_report(analysis)
                    if pdf_content:
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_content,
                            file_name="medical_analysis_report.pdf",
                            mime="application/pdf"
                        )
        else:
            st.error("Could not extract text from the PDF file.")


def render_disease_prediction_tab(analyzer: MedicalAnalyzer):
    """Render the disease prediction tab."""
    st.markdown("### Disease Risk Prediction")
    st.markdown("Select a disease and enter patient information for ML-based risk assessment.")
    
    # Get active diseases
    active_diseases = disease_manager.get_active_diseases()
    
    if not active_diseases:
        st.warning("No disease prediction models are currently active. Please contact administrator.")
        return
    
    # Disease selection dropdown with default option
    disease_options = {"Select a disease": None}  # Default option
    disease_options.update({config.display_name: name for name, config in active_diseases.items()})
    
    selected_display_name = st.selectbox(
        "Select Disease for Prediction",
        options=list(disease_options.keys()),
        help="Choose the disease you want to predict risk for"
    )
    
    # Only show form content if a disease is selected
    if selected_display_name == "Select a disease":
        st.info("👆 Please select a disease from the dropdown above to begin risk assessment.")
        return
    
    selected_disease_name = disease_options[selected_display_name]
    selected_config = active_diseases[selected_disease_name]
    
    # Show disease info
    with st.expander("ℹ️ About this prediction model"):
        st.write(f"**Description:** {selected_config.description}")
        st.write(f"**Accepted file formats:** {', '.join(selected_config.accepted_formats)}")
        if selected_disease_name != "diabetes":
            st.info("This is a placeholder model. Upload functionality will be implemented soon.")
    
    # File upload section (for non-diabetes diseases)
    uploaded_file = None
    if selected_disease_name != "diabetes":
        st.markdown("#### File Upload")
        uploaded_file = st.file_uploader(
            f"Upload file for {selected_display_name} analysis",
            type=selected_config.accepted_formats,
            help=f"Supported formats: {', '.join(selected_config.accepted_formats)}"
        )
        
        if uploaded_file:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            st.info("File processing and prediction will be implemented in the next update.")
    
    # Dynamic parameter form - only show when disease is selected
    with st.form("patient_data_form"):
        st.markdown("#### Patient Information")
        
        # Organize parameters into columns
        params = selected_config.parameters
        param_keys = list(params.keys())
        
        # Create columns based on number of parameters
        num_cols = min(3, len(param_keys))
        if num_cols > 0:
            cols = st.columns(num_cols)
            
            form_data = {}
            
            for i, (param_name, param_config) in enumerate(params.items()):
                col_idx = i % num_cols
                
                with cols[col_idx]:
                    label = param_config.get('label', param_name.replace('_', ' ').title())
                    help_text = param_config.get('help', '')
                    
                    if param_config['type'] == 'number':
                        value = st.number_input(
                            label,
                            min_value=param_config.get('min', 0),
                            max_value=param_config.get('max', 100),
                            value=param_config.get('default', 0),
                            step=param_config.get('step', 1),
                            help=help_text
                        )
                    elif param_config['type'] == 'checkbox':
                        value = st.checkbox(
                            label,
                            value=param_config.get('default', False),
                            help=help_text
                        )
                    elif param_config['type'] == 'selectbox':
                        options = param_config.get('options', [1, 2, 3, 4, 5])
                        default_idx = 0
                        if 'default' in param_config:
                            try:
                                default_idx = options.index(param_config['default'])
                            except ValueError:
                                default_idx = 0
                        value = st.selectbox(
                            label,
                            options=options,
                            index=default_idx,
                            help=help_text
                        )
                    else:
                        value = st.text_input(
                            label,
                            value=str(param_config.get('default', '')),
                            help=help_text
                        )
                    
                    form_data[param_name] = value
        
        submitted = st.form_submit_button("Predict Risk", type="primary")
        
        if submitted:
            # Handle different disease predictions
            if selected_disease_name == "diabetes":
                # Use existing diabetes prediction
                patient_data = PatientData(**form_data)
                
                with st.spinner("Analyzing patient data..."):
                    prediction = analyzer.predict_disease_risk(patient_data)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", prediction['prediction'])
                with col2:
                    st.metric("Risk Score", f"{prediction['risk_score']:.2f}")
                with col3:
                    st.metric("Confidence", f"{prediction['confidence']:.2f}")
                
                # Recommendations
                if prediction.get('recommendations'):
                    st.markdown("#### Recommendations")
                    for rec in prediction['recommendations']:
                        st.write(f"• {rec}")
            else:
                # Placeholder for other diseases
                st.success("Form submitted successfully!")
                st.info(f"Prediction for {selected_display_name} is being processed...")
                
                # Show placeholder results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", "Processing")
                with col2:
                    st.metric("Model", "Placeholder")
                with col3:
                    st.metric("Data Points", len(form_data))
                
                st.markdown("#### Next Steps")
                st.write("• Upload relevant medical files if required")
                st.write("• Wait for model processing (implementation pending)")
                st.write("• Review results and recommendations")
                
                if uploaded_file:
                    st.write(f"• Uploaded file: {uploaded_file.name}")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", prediction['prediction'])
            with col2:
                st.metric("Risk Score", f"{prediction['risk_score']:.2f}")
            with col3:
                st.metric("Confidence", f"{prediction['confidence']:.2f}")
            
            # Recommendations
            if prediction.get('recommendations'):
                st.markdown("#### Recommendations")
                for rec in prediction['recommendations']:
                    st.write(f"• {rec}")


def main():
    """Main application function."""
    setup_page_config()
    apply_custom_css()
    
    # Initialize analyzer
    analyzer = MedicalAnalyzer()
    
    # Render header
    render_header()
    
    # Add backoffice access
    if st.sidebar.button("🔧 Admin Panel"):
        st.session_state.show_backoffice = not st.session_state.get('show_backoffice', False)
    
    if st.session_state.get('show_backoffice', False):
        from backoffice import render_backoffice_page
        render_backoffice_page()
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Document Analysis", "Disease Prediction"])
    
    with tab1:
        render_document_analysis_tab(analyzer)
    
    with tab2:
        render_disease_prediction_tab(analyzer)
    
    # Footer
    st.markdown("---")
    st.markdown("*Medical AI Analyzer v2.0 - For informational purposes only. Always consult healthcare professionals.*")


if __name__ == "__main__":
    main()
