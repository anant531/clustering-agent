import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import io
import time
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import your clustering pipeline modules
try:
    from clustering_pipeline.core.pipeline import clustering_pipeline
    from clustering_pipeline.core.data_loader import DataValidationConfig
    from clustering_pipeline.tools.config import load_config, PipelineConfig
    PIPELINE_AVAILABLE = True
except ImportError:
    st.error("⚠️ Clustering pipeline modules not found. Please install the clustering_pipeline package.")
    PIPELINE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Clustering Pipeline Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .section-header {
        background: #f1f3f4;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #4285f4;
        margin: 1rem 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'uploaded_data': None,
        'selected_features': [],
        'clustering_results': None,
        'quality_report': None,
        'pipeline_config': None,
        'processing_stage': 'upload',
        'show_advanced': False,
        'clustering_complete': False,
        'export_ready': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_main_header():
    """Create the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Clustering Pipeline Studio</h1>
        <p>Professional Data Clustering & Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

def setup_sidebar():
    """Configure the sidebar with navigation and settings."""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=ClusterPro", 
                caption="Clustering Pipeline v2.0")
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📋 Navigation")
        stages = {
            'upload': '📁 Data Upload',
            'features': '🎯 Feature Selection', 
            'config': '⚙️ Configuration',
            'clustering': '🔬 Clustering',
            'results': '📊 Results'
        }
        
        for stage_key, stage_name in stages.items():
            if st.button(stage_name, key=f"nav_{stage_key}", use_container_width=True):
                st.session_state.processing_stage = stage_key
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        st.session_state.show_advanced = st.checkbox("Show Advanced Options", 
                                                    value=st.session_state.show_advanced)
        
        # Pipeline Status
        st.markdown("### 📈 Pipeline Status")
        status_icons = {
            'upload': '⏳' if st.session_state.uploaded_data is None else '✅',
            'features': '⏳' if not st.session_state.selected_features else '✅',
            'config': '⏳' if st.session_state.pipeline_config is None else '✅',
            'clustering': '⏳' if not st.session_state.clustering_complete else '✅',
            'results': '⏳' if not st.session_state.export_ready else '✅'
        }
        
        for stage_key, stage_name in stages.items():
            st.write(f"{status_icons[stage_key]} {stage_name.split(' ', 1)[1]}")

@st.cache_data
def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """Analyze data quality and return summary statistics."""
    analysis = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(exclude=[np.number]).columns.tolist(),
        'duplicate_rows': df.duplicated().sum(),
        'unique_values': {col: df[col].nunique() for col in df.columns}
    }
    
    # Statistical summary for numeric columns
    if analysis['numeric_columns']:
        analysis['statistics'] = df[analysis['numeric_columns']].describe().to_dict()
    
    return analysis

def data_upload_section():
    """Handle data upload and initial preview."""
    st.markdown('<div class="section-header">📁 Data Upload & Preview</div>', 
                unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file with your data for clustering analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load data with progress bar
            with st.spinner('Loading data...'):
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = df
            
            # Success message
            st.markdown(f"""
            <div class="success-box">
                ✅ <strong>Data loaded successfully!</strong><br>
                📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns<br>
                💾 Size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
            </div>
            """, unsafe_allow_html=True)
            
            # Data preview tabs
            tab1, tab2, tab3 = st.tabs(["🔍 Preview", "📊 Summary", "🔧 Quality"])
            
            with tab1:
                st.markdown("**Data Preview (First 100 rows)**")
                st.dataframe(df.head(100), use_container_width=True, height=400)
            
            with tab2:
                # Analyze data quality
                analysis = analyze_data_quality(df)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", f"{analysis['shape'][0]:,}")
                with col2:
                    st.metric("Total Columns", analysis['shape'][1])
                with col3:
                    st.metric("Numeric Columns", len(analysis['numeric_columns']))
                with col4:
                    st.metric("Memory Usage", f"{analysis['memory_usage']:.1f} MB")
                
                # Column information
                st.markdown("**Column Information**")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': [str(dtype) for dtype in df.dtypes],
                    'Non-Null Count': [df[col].count() for col in df.columns],
                    'Null Count': [df[col].isnull().sum() for col in df.columns],
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
            
            with tab3:
                # Data quality issues
                issues = []
                if analysis['duplicate_rows'] > 0:
                    issues.append(f"🔄 {analysis['duplicate_rows']} duplicate rows found")
                
                missing_cols = [col for col, count in analysis['missing_values'].items() if count > 0]
                if missing_cols:
                    issues.append(f"❌ Missing values in {len(missing_cols)} columns")
                
                if not analysis['numeric_columns']:
                    issues.append("⚠️ No numeric columns found - clustering requires numeric features")
                
                if issues:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("**⚠️ Data Quality Issues Detected:**")
                    for issue in issues:
                        st.markdown(f"- {issue}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        ✅ <strong>No major data quality issues detected!</strong><br>
                        Your data appears to be ready for clustering analysis.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Missing values heatmap
                if missing_cols:
                    st.markdown("**Missing Values Heatmap**")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=ax)
                    plt.title('Missing Values Pattern')
                    st.pyplot(fig)
        
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                ❌ <strong>Error loading data:</strong><br>
                {str(e)}
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.uploaded_data is not None:
        # Show previously uploaded data
        df = st.session_state.uploaded_data
        st.info(f"📊 Using previously uploaded data: {df.shape[0]} rows × {df.shape[1]} columns")

def feature_selection_section():
    """Handle feature selection and correlation analysis."""
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first in the Data Upload section.")
        return
    
    df = st.session_state.uploaded_data
    analysis = analyze_data_quality(df)
    
    st.markdown('<div class="section-header">🎯 Feature Selection & Analysis</div>', 
                unsafe_allow_html=True)
    
    if not analysis['numeric_columns']:
        st.error("❌ No numeric columns found. Clustering requires numeric features.")
        return
    
    # Feature selection interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Select Features for Clustering**")
        selected_features = st.multiselect(
            "Choose numeric columns:",
            options=analysis['numeric_columns'],
            default=st.session_state.selected_features if st.session_state.selected_features else analysis['numeric_columns'][:min(5, len(analysis['numeric_columns']))],
            help="Select 2 or more numeric features for clustering analysis"
        )
        
        st.session_state.selected_features = selected_features
    
    with col2:
        if selected_features:
            st.markdown("**Selected Features Summary**")
            for feature in selected_features:
                unique_vals = df[feature].nunique()
                missing_vals = df[feature].isnull().sum()
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{feature}</strong><br>
                    Unique: {unique_vals}<br>
                    Missing: {missing_vals}
                </div>
                """, unsafe_allow_html=True)
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features for clustering.")
        return
    
    # Feature analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Correlations", "📈 Scatter Matrix", "⚠️ Multicollinearity"])
    
    with tab1:
        # Feature distributions
        st.markdown("**Feature Distributions**")
        
        n_features = len(selected_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=selected_features,
            vertical_spacing=0.1
        )
        
        for i, feature in enumerate(selected_features):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            fig.add_trace(
                go.Histogram(x=df[feature], name=feature, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=300*n_rows, title="Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Correlation matrix
        st.markdown("**Correlation Matrix**")
        
        corr_matrix = df[selected_features].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Features", y="Features", color="Correlation"),
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Feature Correlation Heatmap"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        high_corr_pairs = []
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:
                    high_corr_pairs.append((selected_features[i], selected_features[j], corr_val))
        
        if high_corr_pairs:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("**⚠️ High Correlation Detected:**")
            for feat1, feat2, corr_val in high_corr_pairs:
                st.markdown(f"- {feat1} ↔ {feat2}: {corr_val:.3f}")
            st.markdown("Consider removing one feature from highly correlated pairs.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Scatter matrix (for up to 6 features)
        if len(selected_features) <= 6:
            st.markdown("**Scatter Plot Matrix**")
            fig = px.scatter_matrix(
                df[selected_features].sample(min(1000, len(df))),  # Sample for performance
                title="Feature Relationships (Sample)"
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Scatter matrix limited to 6 features. Consider reducing selection for visualization.")
    
    with tab4:
        # Multicollinearity analysis
        st.markdown("**Multicollinearity Analysis**")
        
        if len(selected_features) >= 2:
            try:
                from sklearn.preprocessing import StandardScaler
                
                # Calculate condition number
                X = df[selected_features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Simple multicollinearity check using correlation
                condition_number = np.linalg.cond(np.corrcoef(X_scaled.T))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Condition Number", f"{condition_number:.2f}")
                    
                with col2:
                    if condition_number > 30:
                        status = "🔴 High"
                    elif condition_number > 15:
                        status = "🟡 Moderate"  
                    else:
                        status = "🟢 Low"
                    
                    st.metric("Multicollinearity", status)
                
                if condition_number > 15:
                    st.markdown("""
                    <div class="warning-box">
                        ⚠️ <strong>Multicollinearity detected!</strong><br>
                        Consider removing highly correlated features or using dimensionality reduction.
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error calculating multicollinearity: {str(e)}")

def clustering_configuration_section():
    """Handle clustering algorithm configuration."""
    if not st.session_state.selected_features:
        st.warning("⚠️ Please select features first in the Feature Selection section.")
        return
    
    st.markdown('<div class="section-header">⚙️ Clustering Configuration</div>', 
                unsafe_allow_html=True)
    
    # Configuration tabs
    tab1, tab2, tab3 = st.tabs(["🤖 Algorithm", "📊 Parameters", "🔧 Advanced"])
    
    with tab1:
        st.markdown("**Algorithm Selection**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            algorithm = st.selectbox(
                "Clustering Algorithm:",
                options=["kmeans", "agglo"],
                format_func=lambda x: {"kmeans": "K-Means", "agglo": "Agglomerative"}[x],
                help="Choose the clustering algorithm"
            )
            
            # Algorithm descriptions
            descriptions = {
                "kmeans": """
                **K-Means Clustering:**
                - Fast and efficient
                - Works well with spherical clusters
                - Requires pre-specified number of clusters
                - Good for large datasets
                """,
                "agglo": """
                **Agglomerative Clustering:**
                - Hierarchical approach
                - Can find clusters of different shapes
                - More computationally intensive
                - Good for smaller datasets
                """
            }
            
            st.markdown(descriptions[algorithm])
        
        with col2:
            st.markdown("**Algorithm Comparison**")
            
            comparison_data = {
                "Aspect": ["Speed", "Scalability", "Cluster Shapes", "Memory Usage"],
                "K-Means": ["Fast", "High", "Spherical", "Low"],
                "Agglomerative": ["Moderate", "Low", "Any Shape", "High"]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("**Clustering Parameters**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # K range selection
            st.markdown("**Number of Clusters (K)**")
            k_min = st.slider("Minimum K", min_value=2, max_value=10, value=2)
            k_max = st.slider("Maximum K", min_value=k_min, max_value=15, value=min(6, k_min + 4))
            
            if k_max <= k_min:
                st.error("Maximum K must be greater than Minimum K")
        
        with col2:
            # Evaluation metric
            scoring_metric = st.selectbox(
                "Optimization Metric:",
                options=["silhouette", "calinski_harabasz", "davies_bouldin"],
                format_func=lambda x: {
                    "silhouette": "Silhouette Score (Higher is better)",
                    "calinski_harabasz": "Calinski-Harabasz Index (Higher is better)", 
                    "davies_bouldin": "Davies-Bouldin Index (Lower is better)"
                }[x],
                help="Metric used to determine optimal number of clusters"
            )
            
            # Metric descriptions
            metric_descriptions = {
                "silhouette": "Measures how similar points are to their own cluster vs other clusters",
                "calinski_harabasz": "Ratio of between-cluster to within-cluster variance",
                "davies_bouldin": "Average similarity ratio of each cluster with its most similar cluster"
            }
            
            st.info(metric_descriptions[scoring_metric])
    
    with tab3:
        if st.session_state.show_advanced:
            st.markdown("**Advanced Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Data preprocessing options
                st.markdown("**Data Preprocessing**")
                
                scaling_method = st.selectbox(
                    "Scaling Method:",
                    options=["standard", "minmax", "none"],
                    format_func=lambda x: {
                        "standard": "Standard Scaling (Z-score)",
                        "minmax": "Min-Max Scaling (0-1)",
                        "none": "No Scaling"
                    }[x]
                )
                
                handle_missing = st.selectbox(
                    "Missing Values:",
                    options=["drop", "impute_mean", "impute_median"],
                    format_func=lambda x: {
                        "drop": "Drop rows with missing values",
                        "impute_mean": "Impute with mean",
                        "impute_median": "Impute with median"
                    }[x]
                )
            
            with col2:
                # Quality thresholds
                st.markdown("**Quality Thresholds**")
                
                min_samples = st.number_input(
                    "Minimum Samples Required:",
                    min_value=10,
                    max_value=1000,
                    value=50,
                    help="Minimum number of samples after preprocessing"
                )
                
                max_missing_ratio = st.slider(
                    "Max Missing Values Ratio:",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    format="%.2f",
                    help="Maximum allowed missing values per column"
                )
            
            # Create mock data validation config (since we don't have the actual class)
            data_config = {
                'min_samples': min_samples,
                'handle_missing': handle_missing,
                'scaling_method': scaling_method,
                'max_missing_ratio': max_missing_ratio
            }
        else:
            data_config = {
                'min_samples': 50,
                'handle_missing': 'drop',
                'scaling_method': 'standard',
                'max_missing_ratio': 0.1
            }
        
        # Save configuration
        config_data = {
            'algorithm': algorithm,
            'k_range': (k_min, k_max),
            'scoring_metric': scoring_metric,
            'selected_features': st.session_state.selected_features,
            'data_config': data_config
        }
        
        st.session_state.pipeline_config = config_data
        
        # Configuration summary
        st.markdown("**Configuration Summary**")
        st.json({
            'Algorithm': algorithm.upper(),
            'K Range': f"{k_min} - {k_max}",
            'Metric': scoring_metric,
            'Features': len(st.session_state.selected_features),
            'Data Quality': 'Advanced' if st.session_state.show_advanced else 'Default'
        })

def clustering_execution_section():
    """Handle clustering execution with progress tracking."""
    if not st.session_state.pipeline_config:
        st.warning("⚠️ Please configure clustering parameters first.")
        return
    
    st.markdown('<div class="section-header">🔬 Clustering Execution</div>', 
                unsafe_allow_html=True)
    
    config = st.session_state.pipeline_config
    
    # Pre-execution summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dataset Size", f"{st.session_state.uploaded_data.shape[0]:,} rows")
    with col2:
        st.metric("Selected Features", len(config['selected_features']))
    with col3:
        st.metric("K Range", f"{config['k_range'][0]} - {config['k_range'][1]}")
    
    # Execution button
    if st.button("🚀 Run Clustering Analysis", type="primary", use_container_width=True):
        if not PIPELINE_AVAILABLE:
            # Mock clustering for demonstration
            st.warning("⚠️ Using mock clustering since pipeline is not available.")
            
            try:
                # Mock clustering process
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Data Loading
                status_text.text("📂 Loading and validating data...")
                progress_bar.progress(20)
                time.sleep(0.5)
                
                # Step 2: Feature Preparation  
                status_text.text("🎯 Preparing features...")
                progress_bar.progress(40)
                time.sleep(0.5)
                
                # Step 3: Clustering
                status_text.text("🔬 Running clustering analysis...")
                progress_bar.progress(60)
                time.sleep(1)
                
                # Mock clustering with sklearn
                from sklearn.cluster import KMeans, AgglomerativeClustering
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import silhouette_score
                
                df = st.session_state.uploaded_data.copy()
                features = config['selected_features']
                
                # Prepare data
                X = df[features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Mock optimal k selection
                k_optimal = config['k_range'][0] + 1
                
                # Perform clustering
                if config['algorithm'] == 'kmeans':
                    clusterer = KMeans(n_clusters=k_optimal, random_state=42)
                else:
                    clusterer = AgglomerativeClustering(n_clusters=k_optimal)
                
                clusters = clusterer.fit_predict(X_scaled)
                
                # Add cluster labels to dataframe
                clustered_df = df.loc[X.index].copy()
                clustered_df['cluster'] = clusters
                
                # Mock quality report
                mock_quality_report = type('obj', (object,), {
                    'original_shape': df.shape,
                    'final_shape': clustered_df.shape,
                    'quality_score': 0.85,
                    'warnings': [],
                    'actions_taken': ['Applied standard scaling', 'Removed missing values'],
                    'missing_values_count': {col: df[col].isnull().sum() for col in features},
                    'data_types': {col: str(df[col].dtype) for col in features},
                    'columns_validated': features
                })
                
                progress_bar.progress(80)
                status_text.text("📊 Generating results...")
                time.sleep(0.5)
                
                # Store results
                st.session_state.clustering_results = clustered_df
                st.session_state.quality_report = mock_quality_report
                st.session_state.clustering_complete = True
                st.session_state.export_ready = True
                
                progress_bar.progress(100)
                status_text.text("✅ Clustering complete!")
                
                # Success message
                st.balloons()
                st.markdown("""
                <div class="success-box">
                    🎉 <strong>Clustering Analysis Complete!</strong><br>
                    Your data has been successfully clustered. Check the Results section for detailed analysis.
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    ❌ <strong>Clustering failed:</strong><br>
                    {str(e)}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ Clustering pipeline not available. Please install required modules.")
    
    # Show previous results if available
    if st.session_state.clustering_complete:
        st.markdown("### 📈 Quick Results Preview")
        
        df = st.session_state.clustering_results
        quality_report = st.session_state.quality_report
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Data Points", f"{len(df):,}")
        with col2:
            st.metric("Number of Clusters", df['cluster'].nunique())
        with col3:
            st.metric("Data Quality Score", f"{quality_report.quality_score:.3f}")
        with col4:
            # Calculate silhouette score if available
            try:
                from sklearn.metrics import silhouette_score
                from sklearn.preprocessing import StandardScaler
                X = StandardScaler().fit_transform(df[config['selected_features']])
                sil_score = silhouette_score(X, df['cluster'])
                st.metric("Silhouette Score", f"{sil_score:.3f}")
            except:
                st.metric("Algorithm Used", config['algorithm'].upper())
        
        # # Cluster summary table
        # st.markdown("### 📋 Cluster