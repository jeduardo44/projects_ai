"""
Backoffice Management Module
Administrative interface for managing disease configurations
"""

import streamlit as st
import json
from typing import Dict, List, Any
from disease_config import DiseaseConfig, disease_manager

def render_backoffice_page():
    """Render the backoffice management interface."""
    st.title("🔧 Medical AI Analyzer - Backoffice")
    st.markdown("---")
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["Disease Management", "Add New Disease", "System Settings"])
    
    with tab1:
        render_disease_management()
    
    with tab2:
        render_add_disease_form()
    
    with tab3:
        render_system_settings()

def render_disease_management():
    """Render disease management interface."""
    st.header("Disease Configuration Management")
    
    diseases = disease_manager.diseases
    
    if not diseases:
        st.info("No diseases configured yet.")
        return
    
    # Disease list
    for disease_name, config in diseases.items():
        with st.expander(f"{'✅' if config.is_active else '❌'} {config.display_name}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Name:** {config.name}")
                st.write(f"**Description:** {config.description}")
                st.write(f"**Accepted Formats:** {', '.join(config.accepted_formats)}")
                st.write(f"**Model Path:** {config.model_path}")
                st.write(f"**Parameters:** {len(config.parameters)} fields")
                
                # Show parameters
                if st.checkbox(f"Show Parameters - {disease_name}", key=f"show_params_{disease_name}"):
                    st.json(config.parameters)
            
            with col2:
                if st.button(f"{'Deactivate' if config.is_active else 'Activate'}", 
                           key=f"toggle_{disease_name}"):
                    disease_manager.toggle_disease_status(disease_name)
                    st.rerun()
                
                if st.button("Edit", key=f"edit_{disease_name}"):
                    st.session_state[f"editing_{disease_name}"] = True
                    st.rerun()
                
                if st.button("Delete", key=f"delete_{disease_name}", type="secondary"):
                    if st.session_state.get(f"confirm_delete_{disease_name}", False):
                        disease_manager.delete_disease(disease_name)
                        st.success(f"Deleted {disease_name}")
                        st.rerun()
                    else:
                        st.session_state[f"confirm_delete_{disease_name}"] = True
                        st.warning("Click again to confirm deletion")
            
            # Edit form
            if st.session_state.get(f"editing_{disease_name}", False):
                render_edit_disease_form(disease_name, config)

def render_edit_disease_form(disease_name: str, config: DiseaseConfig):
    """Render form to edit existing disease configuration."""
    st.subheader(f"Edit {config.display_name}")
    
    with st.form(f"edit_disease_{disease_name}"):
        display_name = st.text_input("Display Name", value=config.display_name)
        description = st.text_area("Description", value=config.description)
        
        # File formats
        format_options = ["csv", "json", "pdf", "jpg", "png", "mp4", "dicom", "xlsx"]
        selected_formats = st.multiselect(
            "Accepted File Formats",
            format_options,
            default=config.accepted_formats
        )
        
        model_path = st.text_input("Model Path", value=config.model_path)
        
        # Parameters editing (simplified)
        st.subheader("Parameters")
        parameters_json = st.text_area(
            "Parameters (JSON format)",
            value=json.dumps(config.parameters, indent=2),
            height=200
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Save Changes", type="primary"):
                try:
                    parameters = json.loads(parameters_json)
                    updated_config = DiseaseConfig(
                        name=disease_name,
                        display_name=display_name,
                        description=description,
                        accepted_formats=selected_formats,
                        parameters=parameters,
                        model_path=model_path,
                        is_active=config.is_active
                    )
                    disease_manager.update_disease(disease_name, updated_config)
                    st.success("Disease updated successfully!")
                    st.session_state[f"editing_{disease_name}"] = False
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in parameters")
                except Exception as e:
                    st.error(f"Error updating disease: {e}")
        
        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state[f"editing_{disease_name}"] = False
                st.rerun()

def render_add_disease_form():
    """Render form to add new disease configuration."""
    st.header("Add New Disease")
    
    with st.form("add_new_disease"):
        name = st.text_input("Disease Name (internal)", placeholder="e.g., alzheimer")
        display_name = st.text_input("Display Name", placeholder="e.g., Alzheimer's Disease Risk")
        description = st.text_area("Description", placeholder="Brief description of what this disease prediction does")
        
        # File formats
        format_options = ["csv", "json", "pdf", "jpg", "png", "mp4", "dicom", "xlsx"]
        selected_formats = st.multiselect(
            "Accepted File Formats",
            format_options,
            default=["csv", "json"]
        )
        
        model_path = st.text_input("Model Path", placeholder="models/disease_model.pkl")
        
        # Basic parameters template
        st.subheader("Parameters Template")
        param_template = {
            "age": {"type": "number", "min": 18, "max": 120, "default": 50, "label": "Age"},
            "example_param": {"type": "checkbox", "default": False, "label": "Example Parameter"}
        }
        
        parameters_json = st.text_area(
            "Parameters (JSON format)",
            value=json.dumps(param_template, indent=2),
            height=200,
            help="Define the input parameters for this disease prediction"
        )
        
        if st.form_submit_button("Add Disease", type="primary"):
            if not name or not display_name:
                st.error("Name and Display Name are required")
            else:
                try:
                    parameters = json.loads(parameters_json)
                    new_disease = DiseaseConfig(
                        name=name.lower().replace(" ", "_"),
                        display_name=display_name,
                        description=description,
                        accepted_formats=selected_formats,
                        parameters=parameters,
                        model_path=model_path,
                        is_active=True
                    )
                    disease_manager.add_disease(new_disease)
                    st.success(f"Disease '{display_name}' added successfully!")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in parameters")
                except Exception as e:
                    st.error(f"Error adding disease: {e}")

def render_system_settings():
    """Render system settings interface."""
    st.header("System Settings")
    
    # Export/Import configurations
    st.subheader("Configuration Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export Configuration**")
        if st.button("Download Current Config"):
            config_data = {name: config.__dict__ for name, config in disease_manager.diseases.items()}
            st.download_button(
                label="Download diseases.json",
                data=json.dumps(config_data, indent=2),
                file_name="diseases_config.json",
                mime="application/json"
            )
    
    with col2:
        st.write("**Import Configuration**")
        uploaded_file = st.file_uploader("Upload Configuration File", type=['json'])
        if uploaded_file is not None:
            try:
                config_data = json.load(uploaded_file)
                if st.button("Import Configuration"):
                    for name, config_dict in config_data.items():
                        disease_config = DiseaseConfig(**config_dict)
                        disease_manager.diseases[name] = disease_config
                    disease_manager.save_configurations()
                    st.success("Configuration imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing configuration: {e}")
    
    # Statistics
    st.subheader("System Statistics")
    diseases = disease_manager.diseases
    active_diseases = [d for d in diseases.values() if d.is_active]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Diseases", len(diseases))
    with col2:
        st.metric("Active Diseases", len(active_diseases))
    with col3:
        st.metric("Inactive Diseases", len(diseases) - len(active_diseases))

if __name__ == "__main__":
    # This allows running the backoffice as a standalone app
    # Run with: streamlit run backoffice.py
    st.set_page_config(
        page_title="Medical AI - Backoffice",
        page_icon="🔧",
        layout="wide"
    )
    render_backoffice_page()
