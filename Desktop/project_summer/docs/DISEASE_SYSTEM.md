# Disease Configuration and Backoffice System

## New Features Added

### 1. Dynamic Disease Selection
- **Dropdown Menu**: Users can now select from multiple disease prediction models
- **Configurable Parameters**: Each disease has its own set of input parameters
- **File Format Support**: Different diseases accept different file formats (CSV, PDF, MP4, Images, etc.)

### 2. Disease Configuration System
- **disease_config.py**: Core configuration management for diseases
- **Flexible Parameters**: Support for various input types (numbers, checkboxes, dropdowns)
- **File Format Validation**: Each disease specifies which file formats it accepts

### 3. Backoffice Administration Panel
- **Admin Interface**: Accessible via sidebar "🔧 Admin Panel" button
- **Disease Management**: Add, edit, delete, and toggle disease configurations
- **Real-time Configuration**: Changes apply immediately without restart
- **Import/Export**: Backup and restore disease configurations

## Default Disease Models

### Active Models
1. **Diabetes Risk** (Fully Functional)
   - Accepts: CSV, JSON
   - Parameters: Age, BMI, glucose, blood pressure, etc.
   - ML Model: Random Forest with synthetic data

### Placeholder Models (Ready for Implementation)
2. **Heart Disease Risk**
   - Accepts: CSV, PDF, JSON
   - Parameters: Age, cholesterol, blood pressure, smoking, exercise

3. **Lung Disease Risk**
   - Accepts: PDF, JPG, PNG, DICOM, MP4
   - Parameters: Age, smoking history, chemical exposure

4. **Cancer Screening Analysis**
   - Accepts: JPG, PNG, DICOM, PDF, MP4
   - Parameters: Age, gender, family history, screening type

## Usage

### For End Users
1. Navigate to "Disease Prediction" tab
2. Select desired disease from dropdown
3. Upload files if required (for non-diabetes models)
4. Fill in patient parameters
5. Click "Predict Risk"

### For Administrators
1. Click "🔧 Admin Panel" in sidebar
2. Manage existing diseases or add new ones
3. Configure accepted file formats and parameters
4. Toggle active/inactive status
5. Export/import configurations

## File Structure
```
├── app.py                  # Main application
├── disease_config.py       # Disease configuration management
├── backoffice.py          # Admin interface
├── config/
│   └── diseases.json      # Disease configurations (auto-generated)
├── models/
│   ├── diabetes_model.pkl # Trained diabetes model
│   └── *.pkl             # Placeholder for other models
```

## Technical Implementation

### Adding New Disease Models
1. Use backoffice to add disease configuration
2. Implement prediction logic in ml_models.py
3. Train and save model to specified path
4. Update file processing logic as needed

### Parameter Types Supported
- **number**: Numeric inputs with min/max validation
- **checkbox**: Boolean true/false inputs
- **selectbox**: Dropdown with predefined options
- **text**: Text input fields

### File Format Processing
Each disease can specify accepted formats:
- **csv**: Tabular data
- **json**: Structured data
- **pdf**: Documents and reports
- **jpg/png**: Medical images
- **mp4**: Video files
- **dicom**: Medical imaging standard
- **xlsx**: Excel spreadsheets

## Next Steps for Full Implementation
1. Implement file processing for each format type
2. Train actual ML models for placeholder diseases
3. Add data validation and preprocessing pipelines
4. Implement batch processing capabilities
5. Add audit logging for admin actions
