# 🏥 Medical AI Analyzer

A clean, modern Streamlit application that uses LangChain and OpenAI to analyze medical documents and provide intelligent insights, risk assessments, and recommendations.

## ✨ Features

- **PDF Upload & Processing**: Upload medical PDFs and extract text content
- **AI-Powered Analysis**: Uses OpenAI GPT models for intelligent medical data analysis
- **Disease Analysis**: Analyze symptoms for specific medical conditions
- **Comprehensive Insights**: 
  - Key medical findings
  - Risk factor identification
  - Actionable recommendations
  - Critical alerts
  - Medical terminology extraction
- **Export Capabilities**: Export results as JSON
- **Modern UI**: Clean, responsive interface with medical-themed design

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd project_summer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**
   
   **IMPORTANT: Never commit your API key to version control!**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   
   The app will be available at `http://localhost:8501`

## 📖 Usage Guide

### Document Analysis
1. **Upload Medical PDF**
   - Click "Browse files" to select a medical PDF document
   - Supported format: PDF only

2. **Analyze Document**
   - Click "Analyze Document" button
   - The app will extract text and analyze with AI

3. **Review Results**
   - Summary of medical data
   - Key findings and risk factors
   - Medical recommendations
   - Important medical terminology

### Disease Analysis
1. **Select Disease/Exam**
   - Choose from predefined list of conditions
   - Includes common diseases and medical exams

2. **Enter Symptoms**
   - Describe patient symptoms or medical data
   - Provide detailed observations

3. **Get Analysis**
   - Likelihood assessment
   - Key indicators
   - Differential diagnosis
   - Recommendations and next steps

## 🔧 Configuration

### API Configuration
- Enter your OpenAI API key in the sidebar
- The app uses GPT-3.5-turbo by default
- Temperature is set to 0.1 for consistent medical analysis

## 🏗️ Architecture

```
Medical AI Analyzer
├── Streamlit Frontend
│   ├── File Upload Interface
│   ├── Analysis Dashboard
│   └── Results Display
├── LangChain Backend
│   ├── PDF Text Extraction (PyPDF)
│   ├── LLM Chain (OpenAI GPT)
│   └── Structured Output Parsing
└── Data Processing
    ├── JSON Response Parsing
    └── Export Generation
```

## 📊 Sample Output

The app generates structured analysis in JSON format:

```json
{
  "summary": "Patient shows elevated blood pressure and cholesterol levels...",
  "key_findings": [
    "Hypertension stage 2",
    "High LDL cholesterol",
    "Family history of cardiovascular disease"
  ],
  "risk_factors": [
    "Cardiovascular disease risk",
    "Stroke risk",
    "Kidney disease risk"
  ],
  "recommendations": [
    "Lifestyle modifications",
    "Medication review",
    "Regular monitoring"
  ],
  "medical_terms": [
    "Hypertension",
    "Hyperlipidemia",
    "Cardiovascular"
  ],
  "confidence_score": 0.87,
  "critical_alerts": [
    "Immediate blood pressure control needed"
  ],
  "data_quality": "good"
}
```

## 🔒 Privacy & Security

- **Local Processing**: PDF files are processed locally and not stored
- **Temporary Files**: Temporary files are automatically deleted after processing
- **API Security**: OpenAI API key is handled securely
- **No Data Storage**: No medical data is stored on the server

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **LangChain**: AI/LLM orchestration
- **OpenAI**: GPT model access
- **PyPDF**: PDF text extraction
- **Pydantic**: Data validation and serialization

### Key Components
- **MedicalAnalysis**: Pydantic model for document analysis
- **DiseaseAnalysis**: Pydantic model for disease-specific analysis
- **LangChain Integration**: Structured prompts and output parsing
- **Modern UI**: Clean, responsive design with medical theme

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 