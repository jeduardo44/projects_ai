# Medical Data Analyzer 🏥

### � Analyze Medical Documents
- Upload any PDF medical document (lab reports, medical records, clinical notes)
- The AI reads through everything and gives you a summary
- Extracts important medical terms and explains what they mean
- Highlights any critical issues that need attention
- Creates a nice PDF report with all the findings

### 🩺 Predict Diseases 
- Uses machine learning to predict diabetes risk (84% accuracy!)
- Just enter 9 simple parameters like age, BMI, glucose levels, etc.
- Gives you a risk score and confidence level
- Provides personalized health recommendations
- All based on real medical data and validated algorithms

### 🎨 Nice Interface
- Clean, modern design with a purple-blue theme
- Easy to use - no medical degree required!
- Works on desktop and mobile
- Real-time results as you upload or enter data

# Install the required packages
pip install -r requirements.txt

# Add your OpenAI API key (you'll need one for the AI features)
cp .env.example .env
# Edit .env and add your API key

# Start the app
streamlit run app.py
```

Then go to `http://localhost:8501` in your browser and you're good to go!

## The ML Model Stats

The diabetes prediction model is pretty solid:
- **84% accuracy** using RandomForest algorithm
- Trained on 1,000 patient records
- Uses 9 different health parameters
- Properly tested with train/test split

Example prediction result:
```json
{
  "prediction": "Type 2 Diabetes",
  "confidence": 0.84,
  "risk_score": 0.84,
  "recommendations": [
    "See an endocrinologist",
    "Start diabetic meal plan", 
    "Monitor glucose regularly",
    "Begin exercise program"
  ]
}
```

## Who might find this useful?

- **Doctors & Healthcare Workers** - Quick document analysis and decision support
- **Medical Students** - Learning tool for analyzing cases
- **Researchers** - Processing medical data efficiently  
- **Anyone curious about their health** - Get insights into medical documents or diabetes risk

## Cool Technical Stuff

- Built with pure functions (no classes) - keeps the code clean and simple
- Uses real AI (OpenAI + LangChain) plus custom ML models
- Processes everything securely - no data gets stored permanently
- Modern web interface with smooth animations
- Well-tested code with proper error handling

---

That's it! Feel free to contribute or ask questions. The code is MIT licensed so you can do whatever you want with it.
