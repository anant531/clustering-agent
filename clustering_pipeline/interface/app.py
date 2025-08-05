# clustering_pipeline/interface/app.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Page configuration
st.set_page_config(
    page_title="Clustering Agent",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        color: #155724;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        color: #856404;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'df': None,
        'clustered_df': None,
        'quality_report': None,
        'selected_features': [],
        'clustering_complete': False,
        'processing': False,
        'uploaded_file_name': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_header():
    """Display the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Advanced Clustering Pipeline</h1>
        <p>Upload your data, select features, and discover hidden patterns with machine learning</p>
    </div>
    """, unsafe_allow_html=True)

def show_sidebar_info():
    """Show information in sidebar."""
    st.sidebar.markdown("## 📊 Pipeline Status")
    
    # Status indicators
    if st.session_state.df is not None:
        st.sidebar.success(f"✅ Data loaded: {st.session_state.df.shape[0]} rows")
    else:
        st.sidebar.info("📁 No data loaded")
    
    if st.session_state.selected_features:
        st.sidebar.success(f"✅ Features selected: {len(st.session_state.selected_features)}")
    else:
        st.sidebar.info("🔧 No features selected")
    
    if st.session_state.clustering_complete:
        st.sidebar.success("✅ Clustering complete")
    else:
        st.sidebar.info("⏳ Clustering pending")
    
    # Show sample data info
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📋 Sample Data")
    st.sidebar.markdown("""
    **Available sample datasets:**
    - `customer_segments.csv` - Customer segmentation
    - `iris_like.csv` - Flower measurements
    - `problematic_data.csv` - For testing error handling
    
    Generate sample data:
    ```bash
    python scripts/generate_sample_data.py
    ```
    """)

def main():
    """Main Streamlit application."""
    # Initialize session state
    init_session_state()
    
    # Show header
    show_header()
    
    # Show sidebar info
    show_sidebar_info()
    
    # Main content area - we'll build this in parts
    st.markdown("## 🚀 Getting Started")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Data Upload", 
        "🔧 Feature Selection", 
        "🤖 Clustering", 
        "📊 Results"
    ])
    
    with tab1:
        st.markdown("### Upload Your Data")
        st.info("This section will handle CSV file uploads")
        
        # Placeholder content
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Supported formats:**")
            st.markdown("- CSV files (.csv)")
            st.markdown("- Comma or semicolon separated")
            st.markdown("- Headers in first row")
        
        with col2:
            st.markdown("**Requirements:**")
            st.markdown("- At least 10 rows of data")
            st.markdown("- Numeric features for clustering")
            st.markdown("- Handle missing values properly")
    
    with tab2:
        st.markdown("### Select Features for Clustering")
        st.info("This section will show feature selection tools")
        
        if st.session_state.df is not None:
            st.write("Data preview would go here...")
        else:
            st.warning("Please upload data first")
    
    with tab3:
        st.markdown("### Configure and Run Clustering")
        st.info("This section will have clustering controls")
        
        # Show basic clustering info
        st.markdown("""
        **Available algorithms:**
        - K-Means clustering
        - Agglomerative clustering
        
        **Evaluation metrics:**
        - Silhouette score
        - Davies-Bouldin index  
        - Calinski-Harabasz index
        """)
    
    with tab4:
        st.markdown("### Clustering Results and Visualization")
        st.info("This section will show results and plots")
        
        if st.session_state.clustering_complete:
            st.success("Results would be displayed here")
        else:
            st.warning("Run clustering first to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Made by Anant • Clustering Agent v1.0
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()