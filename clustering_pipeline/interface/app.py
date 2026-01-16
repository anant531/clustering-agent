# clustering_pipeline/interface/app.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path


# # Add project root to path
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import our components
try:
    from components.data_upload import render_data_upload_tab
    from components.feature_selection import render_feature_selection_tab
    from components.clustering import render_clustering_tab
    from components.results_dashboard import render_results_dashboard
except ImportError:
    # Fallback if components not found
    render_data_upload_tab = None
    render_feature_selection_tab = None
    render_clustering_tab = None
    render_results_dashboard = None

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
    /* File uploader styling */
    .stFileUploader > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 2px dashed #ffffff40;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    /* Sample data buttons */
    .sample-data-button {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    .sample-data-button:hover {
        border-color: #1f77b4;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.2);
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
        'uploaded_file_name': None,
        'cluster_labels': None,
        'clustering_metrics': None,
        'clusterer': None,
        'processed_data': None,
        'preprocessing_info': None,
        'clustering_params': None,
        'optimization_results': None
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
    
    # Status indicators with more detail
    if st.session_state.df is not None:
        df_info = f"{st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} cols"
        st.sidebar.success(f"✅ Data loaded: {df_info}")
        
        # Show filename if available
        if st.session_state.uploaded_file_name:
            st.sidebar.caption(f"📄 {st.session_state.uploaded_file_name}")
        
        # Show data quality indicators
        numeric_cols = len(st.session_state.df.select_dtypes(include=[np.number]).columns)
        missing_count = st.session_state.df.isnull().sum().sum()
        
        st.sidebar.metric("Numeric Columns", numeric_cols)
        if missing_count > 0:
            st.sidebar.metric("Missing Values", f"{missing_count:,}", delta="⚠️")
        else:
            st.sidebar.metric("Missing Values", "0", delta="✅")
    else:
        st.sidebar.info("📁 No data loaded")
    
    if st.session_state.selected_features:
        st.sidebar.success(f"✅ Features selected: {len(st.session_state.selected_features)}")
        # Show selected features
        with st.sidebar.expander("View Selected Features"):
            for feature in st.session_state.selected_features:
                st.sidebar.write(f"• {feature}")
    else:
        st.sidebar.info("🔧 No features selected")
    
    if st.session_state.clustering_complete:
        st.sidebar.success("✅ Clustering complete")
    else:
        st.sidebar.info("⏳ Clustering pending")
    
    # Progress indicator
    progress = 0
    if st.session_state.df is not None:
        progress += 25
    if st.session_state.selected_features:
        progress += 25
    if st.session_state.clustering_complete:
        progress += 50
    
    st.sidebar.markdown("### 📈 Progress")
    st.sidebar.progress(progress / 100)
    st.sidebar.caption(f"{progress}% Complete")
    
    # Show sample data info
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📋 Sample Data Available")
    st.sidebar.markdown("""
    **Quick start options:**
    - 📊 Customer Segments (300 rows)
    - 🌸 Flower Dataset (150 rows)
    - ⚠️ Problematic Data (test validation)
    
    Use sample data in the Data Upload tab to get started quickly!
    """)
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ℹ️ System Info")
    st.sidebar.caption(f"Streamlit v{st.__version__}")
    st.sidebar.caption(f"Pandas v{pd.__version__}")
    st.sidebar.caption(f"NumPy v{np.__version__}")

def main():
    """Main Streamlit application."""
    # Initialize session state
    init_session_state()
    
    # Show header
    show_header()
    
    # Show sidebar info
    show_sidebar_info()
    
    # Main content area
    st.markdown("## 🚀 Getting Started")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Data Upload", 
        "🔧 Feature Selection", 
        "🤖 Clustering", 
        "📊 Results"
    ])
    
    with tab1:
        # Use the new data upload component if available
        if render_data_upload_tab:
            render_data_upload_tab()
        else:
            # Fallback to simple upload
            st.markdown("### Upload Your Data")
            st.info("⚠️ Component not found. Please ensure data_upload.py is in the components folder.")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.success(f"✅ Loaded {uploaded_file.name}")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab2:
        # Use the new feature selection component
        if render_feature_selection_tab:
            render_feature_selection_tab()
        else:
            # Fallback to simple selection
            st.markdown("### Select Features for Clustering")
            st.info("⚠️ Component not found. Please ensure feature_selection.py is in the components folder.")
            
            if st.session_state.df is not None:
                df = st.session_state.df
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    selected = st.multiselect("Select features:", numeric_cols)
                    st.session_state.selected_features = selected
                    
                    if selected:
                        st.dataframe(df[selected].head())
                else:
                    st.warning("No numeric columns found for clustering.")
            else:
                st.warning("Please upload data first")
    
    with tab3:
        # Use the new clustering component
        if render_clustering_tab:
            render_clustering_tab()
        else:
            # Fallback to simple clustering
            st.markdown("### Configure and Run Clustering")
            st.info("⚠️ Component not found. Please ensure clustering.py is in the components folder.")
            
            if st.session_state.df is not None and st.session_state.selected_features:
                st.markdown("**Ready for clustering with:**")
                st.write(f"• **Data:** {st.session_state.df.shape[0]} rows")
                st.write(f"• **Features:** {', '.join(st.session_state.selected_features)}")
                
                if st.button("Run Basic Clustering"):
                    st.info("Basic clustering would run here...")
            else:
                st.warning("Please upload data and select features first")
    
    with tab4:
        # Use the new results dashboard component
        if render_results_dashboard:
            render_results_dashboard()
        else:
            # Fallback to simple results
            st.markdown("### Clustering Results and Visualization")
            st.info("⚠️ Component not found. Please ensure results_dashboard.py is in the components folder.")
            
            if st.session_state.clustering_complete:
                st.success("✅ Clustering completed!")
                
                if st.session_state.clustered_df is not None:
                    st.markdown("**Results Preview:**")
                    st.dataframe(st.session_state.clustered_df.head(10))
                    
                    # Show cluster distribution
                    cluster_counts = st.session_state.clustered_df['cluster'].value_counts()
                    st.bar_chart(cluster_counts)
            else:
                st.warning("Run clustering first to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Made by Anant • Clustering Agent v1.0 • Cluster Insights Needs Improvement
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

