# clustering_pipeline/interface/components/data_upload.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import io

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate uploaded dataframe for clustering requirements.
    
    Returns:
        dict: Validation results with status, message, and suggestions
    """
    validation_result = {
        'is_valid': False,
        'warnings': [],
        'errors': [],
        'info': {},
        'suggestions': []
    }
    
    # Basic shape validation
    rows, cols = df.shape
    validation_result['info']['shape'] = (rows, cols)
    
    if rows < 10:
        validation_result['errors'].append(f"Too few rows ({rows}). Need at least 10 rows for meaningful clustering.")
    
    if cols < 2:
        validation_result['errors'].append(f"Too few columns ({cols}). Need at least 2 numeric columns for clustering.")
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    validation_result['info']['numeric_columns'] = len(numeric_cols)
    validation_result['info']['numeric_column_names'] = numeric_cols
    
    if len(numeric_cols) < 2:
        validation_result['errors'].append(f"Need at least 2 numeric columns. Found only {len(numeric_cols)}.")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    validation_result['info']['missing_values'] = int(total_missing)
    
    if total_missing > 0:
        missing_percentage = (total_missing / (rows * cols)) * 100
        if missing_percentage > 50:
            validation_result['errors'].append(f"Too many missing values ({missing_percentage:.1f}%). Consider data cleaning first.")
        elif missing_percentage > 20:
            validation_result['warnings'].append(f"High percentage of missing values ({missing_percentage:.1f}%). Will use imputation.")
        else:
            validation_result['warnings'].append(f"Some missing values found ({total_missing}). Will handle automatically.")
    
    # Check for constant columns
    constant_cols = []
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        validation_result['warnings'].append(f"Constant columns found: {constant_cols}. These will be excluded from clustering.")
    
    # Check data types
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    if text_cols:
        validation_result['info']['text_columns'] = text_cols
        validation_result['suggestions'].append(f"Text columns found: {text_cols}. Consider encoding if they contain categories.")
    
    # Overall validation
    validation_result['is_valid'] = len(validation_result['errors']) == 0
    
    return validation_result

def show_data_preview(df: pd.DataFrame, max_rows: int = 100):
    """Show interactive data preview with statistics."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 Data Preview")
        
        # Show sample of data
        preview_rows = min(max_rows, len(df))
        st.dataframe(
            df.head(preview_rows),
            use_container_width=True,
            height=300
        )
        
        if len(df) > preview_rows:
            st.info(f"Showing first {preview_rows} of {len(df)} rows")
    
    with col2:
        st.markdown("#### 📊 Quick Stats")
        
        # Basic statistics
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
        
        # Data types breakdown
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        text_cols = len(df.select_dtypes(include=['object']).columns)
        
        st.metric("Numeric Columns", numeric_cols)
        st.metric("Text Columns", text_cols)
        
        # Missing values
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            missing_pct = (missing_total / (len(df) * len(df.columns))) * 100
            st.metric("Missing Values", f"{missing_total} ({missing_pct:.1f}%)")
        else:
            st.metric("Missing Values", "0 ✅")

def show_validation_results(validation_result: Dict[str, Any]):
    """Display validation results with colored indicators."""
    
    if validation_result['is_valid']:
        st.success("✅ **Data validation passed!** Ready for clustering.")
    else:
        st.error("❌ **Data validation failed.** Please fix the issues below.")
    
    # Show errors
    if validation_result['errors']:
        st.markdown("**🚨 Errors (must fix):**")
        for error in validation_result['errors']:
            st.error(f"• {error}")
    
    # Show warnings
    if validation_result['warnings']:
        st.markdown("**⚠️ Warnings (will handle automatically):**")
        for warning in validation_result['warnings']:
            st.warning(f"• {warning}")
    
    # Show suggestions
    if validation_result['suggestions']:
        st.markdown("**💡 Suggestions:**")
        for suggestion in validation_result['suggestions']:
            st.info(f"• {suggestion}")

def create_sample_data_option():
    """Create sample data for testing."""
    st.markdown("#### 🧪 Or Try Sample Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Customer Segments", use_container_width=True):
            # Create customer segmentation sample data
            np.random.seed(42)
            n_customers = 300
            
            # Create 3 customer segments
            segment1 = pd.DataFrame({
                'age': np.random.normal(25, 5, n_customers//3),
                'income': np.random.normal(30000, 5000, n_customers//3),
                'spending_score': np.random.normal(80, 10, n_customers//3),
                'loyalty_years': np.random.normal(1, 0.5, n_customers//3)
            })
            
            segment2 = pd.DataFrame({
                'age': np.random.normal(45, 8, n_customers//3),
                'income': np.random.normal(70000, 10000, n_customers//3),
                'spending_score': np.random.normal(50, 15, n_customers//3),
                'loyalty_years': np.random.normal(5, 2, n_customers//3)
            })
            
            segment3 = pd.DataFrame({
                'age': np.random.normal(35, 6, n_customers//3),
                'income': np.random.normal(50000, 8000, n_customers//3),
                'spending_score': np.random.normal(70, 12, n_customers//3),
                'loyalty_years': np.random.normal(3, 1, n_customers//3)
            })
            
            sample_df = pd.concat([segment1, segment2, segment3], ignore_index=True)
            sample_df = sample_df.round(2)
            
            # Add some categorical data
            sample_df['customer_id'] = [f"CUST_{i:04d}" for i in range(len(sample_df))]
            sample_df['city'] = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], len(sample_df))
            
            st.session_state.df = sample_df
            st.session_state.uploaded_file_name = "customer_segments_sample.csv"
            st.rerun()
    
    with col2:
        if st.button("🌸 Flower Dataset", use_container_width=True):
            # Create iris-like dataset
            np.random.seed(42)
            n_samples = 150
            
            # Create 3 flower types
            type1 = pd.DataFrame({
                'sepal_length': np.random.normal(5.0, 0.4, n_samples//3),
                'sepal_width': np.random.normal(3.5, 0.4, n_samples//3),
                'petal_length': np.random.normal(1.4, 0.2, n_samples//3),
                'petal_width': np.random.normal(0.2, 0.1, n_samples//3)
            })
            
            type2 = pd.DataFrame({
                'sepal_length': np.random.normal(6.0, 0.5, n_samples//3),
                'sepal_width': np.random.normal(2.8, 0.3, n_samples//3),
                'petal_length': np.random.normal(4.3, 0.5, n_samples//3),
                'petal_width': np.random.normal(1.3, 0.2, n_samples//3)
            })
            
            type3 = pd.DataFrame({
                'sepal_length': np.random.normal(6.5, 0.6, n_samples//3),
                'sepal_width': np.random.normal(3.0, 0.3, n_samples//3),
                'petal_length': np.random.normal(5.5, 0.6, n_samples//3),
                'petal_width': np.random.normal(2.0, 0.3, n_samples//3)
            })
            
            sample_df = pd.concat([type1, type2, type3], ignore_index=True)
            sample_df = sample_df.round(2)
            sample_df['flower_id'] = [f"F_{i:03d}" for i in range(len(sample_df))]
            
            st.session_state.df = sample_df
            st.session_state.uploaded_file_name = "flower_measurements_sample.csv"
            st.rerun()
    
    with col3:
        if st.button("⚠️ Problematic Data", use_container_width=True):
            # Create problematic dataset for testing validation
            np.random.seed(42)
            
            # Small dataset with issues
            problem_df = pd.DataFrame({
                'feature1': [1, 2, np.nan, 4, 5],  # Missing values
                'feature2': [10, 10, 10, 10, 10],  # Constant column
                'feature3': ['A', 'B', 'C', 'D', 'E'],  # Text column
                'feature4': [1.1, 2.2, 3.3, np.nan, np.nan]  # More missing values
            })
            
            st.session_state.df = problem_df
            st.session_state.uploaded_file_name = "problematic_data_sample.csv"
            st.rerun()

def render_data_upload_tab():
    """Main function to render the data upload tab."""
    
    st.markdown("### 📁 Upload Your Data")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with your data for clustering analysis"
    )
    
    # Handle file upload
    if uploaded_file is not None:
        try:
            # Try different encodings and separators
            content = uploaded_file.read()
            
            # Try to detect separator and encoding
            sample = content[:1024].decode('utf-8', errors='ignore')
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Detect separator
            comma_count = sample.count(',')
            semicolon_count = sample.count(';')
            separator = ',' if comma_count >= semicolon_count else ';'
            
            # Try reading with detected separator
            try:
                df = pd.read_csv(uploaded_file, sep=separator, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=separator, encoding='latin-1')
            
            # Store in session state
            st.session_state.df = df
            st.session_state.uploaded_file_name = uploaded_file.name
            
            st.success(f"✅ Successfully loaded: **{uploaded_file.name}**")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Make sure your file is a valid CSV with headers in the first row")
    
    # Show sample data options if no file uploaded
    if st.session_state.df is None:
        create_sample_data_option()
    
    # Process loaded data
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("---")
        
        # Validate data
        validation_result = validate_dataframe(df)
        show_validation_results(validation_result)
        
        st.markdown("---")
        
        # Show data preview
        show_data_preview(df)
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reload Data", use_container_width=True):
                st.session_state.df = None
                st.session_state.uploaded_file_name = None
                st.session_state.selected_features = []
                st.session_state.clustering_complete = False
                st.rerun()
        
        with col2:
            if st.button("💾 Download Sample", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"processed_{st.session_state.uploaded_file_name}",
                    mime="text/csv"
                )
        
        with col3:
            if validation_result['is_valid']:
                if st.button("➡️ Next: Select Features", use_container_width=True, type="primary"):
                    st.success("✅ Data is valid! Please go to the 'Feature Selection' tab.")
            else:
                st.button("➡️ Fix Issues First", use_container_width=True, disabled=True)

