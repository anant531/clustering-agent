# clustering_pipeline/interface/components/feature_selection.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

def calculate_feature_stats(df: pd.DataFrame, columns: list) -> Dict[str, Any]:
    """Calculate comprehensive statistics for each column."""
    stats = {}
    
    for col in columns:
        series = df[col]
        col_stats = {
            'dtype': str(series.dtype),
            'count': len(series),
            'non_null_count': series.count(),
            'null_count': series.isnull().sum(),
            'unique_count': series.nunique(),
        }
        
        # Handle different data types appropriately
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            # For numeric (non-boolean) columns, calculate all statistics
            try:
                col_stats.update({
                    'mean': series.mean(),
                    'median': series.median(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'q25': series.quantile(0.25),
                    'q75': series.quantile(0.75),
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis(),
                })
            except Exception as e:
                # If quantile calculation fails, provide basic stats only
                col_stats.update({
                    'mean': series.mean(),
                    'median': series.median(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'q25': None,
                    'q75': None,
                    'skewness': None,
                    'kurtosis': None,
                })
        
        elif pd.api.types.is_bool_dtype(series):
            # For boolean columns, provide appropriate statistics
            col_stats.update({
                'mean': float(series.mean()),  # Proportion of True values
                'median': None,  # Not meaningful for boolean
                'std': None,     # Not meaningful for boolean
                'min': bool(series.min()) if not series.empty else None,
                'max': bool(series.max()) if not series.empty else None,
                'q25': None,
                'q75': None,
                'skewness': None,
                'kurtosis': None,
                'true_count': int(series.sum()),
                'false_count': int(len(series) - series.sum()),
                'true_percentage': float((series.sum() / len(series)) * 100) if len(series) > 0 else 0,
            })
        
        elif pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
            # For categorical/object columns
            try:
                mode_result = series.mode()
                most_frequent = mode_result.iloc[0] if len(mode_result) > 0 else None
                most_frequent_count = series.value_counts().iloc[0] if len(series) > 0 else 0
            except:
                most_frequent = None
                most_frequent_count = 0
                
            col_stats.update({
                'mean': None,
                'median': None,
                'std': None,
                'min': None,
                'max': None,
                'q25': None,
                'q75': None,
                'skewness': None,
                'kurtosis': None,
                'mode': most_frequent,
                'most_frequent_count': most_frequent_count,
            })
        
        else:
            # For other data types, provide minimal stats
            col_stats.update({
                'mean': None,
                'median': None,
                'std': None,
                'min': series.min() if len(series) > 0 else None,
                'max': series.max() if len(series) > 0 else None,
                'q25': None,
                'q75': None,
                'skewness': None,
                'kurtosis': None,
            })
        
        # Calculate missing percentage
        if col_stats['count'] > 0:
            col_stats['missing_percentage'] = (col_stats['null_count'] / col_stats['count']) * 100
        else:
            col_stats['missing_percentage'] = 0
        
        # Add consistent keys for display
        col_stats['missing'] = col_stats['null_count']
        col_stats['missing_pct'] = col_stats['missing_percentage']
        col_stats['is_numeric'] = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
        col_stats['unique_values'] = col_stats['unique_count']
        
        # Add additional stats for display - Handle range calculation carefully
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            # Only calculate range for truly numeric columns (not boolean)
            try:
                if col_stats['max'] is not None and col_stats['min'] is not None:
                    col_stats['range'] = float(col_stats['max'] - col_stats['min'])
                else:
                    col_stats['range'] = 0.0
            except:
                col_stats['range'] = 0.0
        else:
            # For non-numeric columns (including boolean), don't calculate range
            col_stats['range'] = None
            
            # For categorical columns, add top categories
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                try:
                    value_counts = df[col].value_counts().head(5)
                    col_stats['top_categories'] = value_counts.to_dict()
                except:
                    col_stats['top_categories'] = {}
            else:
                col_stats['top_categories'] = {}
            
        stats[col] = col_stats
    
    return stats

def create_correlation_matrix(df: pd.DataFrame, features: List[str]):
    """Create an interactive correlation matrix for the selected features."""
    if len(features) < 2:
        st.info("Need at least 2 features for correlation analysis")
        return
    
    try:
        # Calculate correlation matrix
        corr_matrix = df[features].corr()
        
        # Create interactive heatmap using plotly
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Feature Correlation Matrix',
            labels=dict(color="Correlation"),
            text_auto=True
        )
        
        # Update layout for better appearance
        fig.update_layout(
            height=max(400, len(features) * 40),
            width=max(400, len(features) * 40),
            title_x=0.5
        )
        
        # Update text format
        fig.update_traces(texttemplate='%{z:.2f}', textfont_size=10)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation insights
        st.markdown("#### 🔍 Correlation Insights")
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((features[i], features[j], corr_val))
        
        if high_corr_pairs:
            st.warning("⚠️ **High Correlations Found:**")
            for feat1, feat2, corr in high_corr_pairs:
                st.write(f"• **{feat1}** ↔ **{feat2}**: {corr:.3f}")
            st.write("Consider removing one feature from highly correlated pairs to avoid redundancy.")
        else:
            st.success("✅ No highly correlated features detected")
            
    except Exception as e:
        st.error(f"Error creating correlation matrix: {str(e)}")

def create_feature_distributions(df: pd.DataFrame, selected_features: List[str]):
    """Create distribution plots for selected features."""
    if not selected_features:
        return
    
    n_features = len(selected_features)
    
    if n_features == 1:
        # Single histogram
        fig = px.histogram(
            df, 
            x=selected_features[0], 
            title=f"Distribution of {selected_features[0]}",
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif n_features <= 4:
        # Subplot for multiple features
        cols = min(2, n_features)
        rows = (n_features + 1) // 2
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=selected_features,
            vertical_spacing=0.1
        )
        
        for i, feature in enumerate(selected_features):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            fig.add_trace(
                go.Histogram(x=df[feature], name=feature, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=300*rows, title_text="Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Too many features for distribution plots. Showing correlation matrix instead.")

def create_pairwise_plots(df: pd.DataFrame, selected_features: List[str], sample_size: int = 1000):
    """Create pairwise scatter plots for selected features."""
    if len(selected_features) < 2:
        return
    
    # Sample data if too large
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        st.info(f"Showing sample of {sample_size} points for performance")
    else:
        df_sample = df
    
    if len(selected_features) == 2:
        # Simple scatter plot
        fig = px.scatter(
            df_sample, 
            x=selected_features[0], 
            y=selected_features[1],
            title=f"{selected_features[0]} vs {selected_features[1]}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif len(selected_features) <= 5:
        # Pairwise scatter matrix
        fig = px.scatter_matrix(
            df_sample,
            dimensions=selected_features,
            title="Pairwise Feature Relationships"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Too many features for pairwise plots. Consider selecting fewer features or use the correlation matrix.")

def show_feature_recommendations(stats: Dict[str, Dict], selected_features: List[str]):
    """Show intelligent feature selection recommendations."""
    st.markdown("#### 🎯 Feature Selection Recommendations")
    
    recommendations = []
    warnings_list = []
    
    for feature in selected_features:
        if feature in stats:
            feature_stats = stats[feature]
            
            # Check for issues
            if feature_stats.get('unique_count', 0) <= 1:
                warnings_list.append(f"**{feature}**: Constant or single unique value - will not contribute to clustering")
            
            elif feature_stats.get('missing_pct', 0) > 50:
                warnings_list.append(f"**{feature}**: {feature_stats.get('missing_pct', 0):.1f}% missing values")
            
            elif feature_stats.get('unique_count', 0) == 1:
                warnings_list.append(f"**{feature}**: Only one unique value")
            
            elif not feature_stats.get('is_numeric', False):
                warnings_list.append(f"**{feature}**: Non-numeric data - consider encoding")
    
    # Show warnings
    if warnings_list:
        st.warning("⚠️ **Feature Issues:**")
        for warning in warnings_list:
            st.write(f"• {warning}")
    
    # General recommendations
    if len(selected_features) < 2:
        st.error("❌ Need at least 2 features for clustering")
    elif len(selected_features) > 10:
        st.warning("⚠️ Many features selected. Consider dimensionality reduction for better performance.")
    else:
        st.success(f"✅ Good selection: {len(selected_features)} features chosen")
    
    # Feature quality score
    total_features = len(selected_features)
    problematic_features = len(warnings_list)
    quality_score = max(0, (total_features - problematic_features) / total_features * 100) if total_features > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Features Selected", total_features)
    with col2:
        st.metric("Quality Score", f"{quality_score:.0f}%")
    with col3:
        st.metric("Issues Found", problematic_features)

def render_feature_selection_tab():
    """Main function to render the feature selection tab."""
    
    if st.session_state.df is None:
        st.warning("📁 Please upload data first in the Data Upload tab")
        return
    
    df = st.session_state.df
    
    st.markdown("### 🔧 Select Features for Clustering")
    st.markdown("Choose which columns to include in your clustering analysis")
    
    # Initialize selected_features in session state if not exists
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    
    # Get all columns and their types
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Calculate feature statistics
    feature_stats = calculate_feature_stats(df, all_columns)
    
    # Feature selection interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📋 Available Features")
        
        # Split currently selected features into numeric and categorical
        current_numeric_selected = [f for f in st.session_state.selected_features if f in numeric_columns]
        current_categorical_selected = [f for f in st.session_state.selected_features if f in categorical_columns]
        
        # Show feature categories
        if numeric_columns:
            st.markdown("**🔢 Numeric Features:**")
            numeric_selected = st.multiselect(
                "Select numeric features",
                numeric_columns,
                default=current_numeric_selected,
                key="numeric_features_selector",
                help="Numeric features that can be directly used for clustering"
            )
        else:
            numeric_selected = []
            st.info("No numeric columns found")
        
        if categorical_columns:
            st.markdown("**📝 Categorical Features:**")
            st.info("💡 Categorical features need encoding before clustering")
            categorical_selected = st.multiselect(
                "Select categorical features (will need encoding)",
                categorical_columns,
                default=current_categorical_selected,
                key="categorical_features_selector",
                help="Categorical features that will be encoded for clustering"
            )
        else:
            categorical_selected = []
        
        # Update session state with combined selections
        new_selected_features = numeric_selected + categorical_selected
        
        # Only update if there's an actual change to prevent unnecessary reruns
        if set(new_selected_features) != set(st.session_state.selected_features):
            st.session_state.selected_features = new_selected_features
    
    with col2:
        st.markdown("#### 📊 Feature Statistics")
        
        if st.session_state.selected_features:
            # Show statistics for selected features
            for feature in st.session_state.selected_features:
                if feature in feature_stats:
                    stats = feature_stats[feature]
                    
                    with st.expander(f"📈 {feature}", expanded=False):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Count", f"{stats['count']:,}")
                            st.metric("Missing", f"{stats['missing']} ({stats['missing_pct']:.1f}%)")
                            st.metric("Unique Values", f"{stats['unique_values']:,}")
                        
                        with col_b:
                            if stats['is_numeric']:
                                st.metric("Mean", f"{stats['mean']:.2f}" if stats['mean'] is not None else "N/A")
                                st.metric("Std Dev", f"{stats['std']:.2f}" if stats['std'] is not None else "N/A")
                                st.metric("Range", f"{stats['range']:.2f}" if stats['range'] is not None else "N/A")
                            else:
                                st.write("**Top Categories:**")
                                top_cats = stats.get('top_categories', {})
                                if top_cats:
                                    for cat, count in list(top_cats.items())[:3]:
                                        st.write(f"• {cat}: {count}")
                                else:
                                    st.write("No categories available")
        else:
            st.info("Select features to see statistics")
    
    # Show recommendations
    if st.session_state.selected_features:
        st.markdown("---")
        show_feature_recommendations(feature_stats, st.session_state.selected_features)
    
    # Visualization section
    if len(st.session_state.selected_features) >= 2:
        st.markdown("---")
        st.markdown("### 📊 Feature Analysis")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🔗 Correlations", "📈 Distributions", "🔍 Relationships"])
        
        with viz_tab1:
            # Only show correlation for numeric features
            numeric_selected_features = [f for f in st.session_state.selected_features if f in numeric_columns]
            if len(numeric_selected_features) >= 2:
                create_correlation_matrix(df, numeric_selected_features)
            else:
                st.info("Need at least 2 numeric features for correlation analysis")
        
        with viz_tab2:
            numeric_selected_features = [f for f in st.session_state.selected_features if f in numeric_columns]
            if numeric_selected_features:
                create_feature_distributions(df, numeric_selected_features)
            else:
                st.info("No numeric features selected for distribution analysis")
        
        with viz_tab3:
            numeric_selected_features = [f for f in st.session_state.selected_features if f in numeric_columns]
            if len(numeric_selected_features) >= 2:
                create_pairwise_plots(df, numeric_selected_features)
            else:
                st.info("Need at least 2 numeric features for relationship analysis")
    
    # Action buttons
    if st.session_state.selected_features:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reset Selection", use_container_width=True):
                st.session_state.selected_features = []
                st.rerun()
        
        with col2:
            if st.button("📋 Select All Numeric", use_container_width=True):
                st.session_state.selected_features = numeric_columns.copy()
                st.rerun()
        
        with col3:
            # Check if selection is valid for clustering
            numeric_in_selection = [f for f in st.session_state.selected_features if f in numeric_columns]
            if len(numeric_in_selection) >= 2:
                if st.button("➡️ Next: Run Clustering", use_container_width=True, type="primary"):
                    st.success("✅ Features selected! Go to the 'Clustering' tab to configure and run clustering.")
                    
                st.button("➡️ Need 2+ Numeric Features", use_container_width=True, disabled=True)
    
    # Feature selection summary
    if st.session_state.selected_features:
        st.markdown("---")
        st.markdown("### 📝 Selection Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Selected Features:**")
            for i, feature in enumerate(st.session_state.selected_features, 1):
                feature_type = "🔢" if feature in numeric_columns else "📝"
                st.write(f"{i}. {feature_type} **{feature}**")
        
        with col2:
            # Show preview of selected data
            st.markdown("**Data Preview:**")
            preview_df = df[st.session_state.selected_features].head(5)
            st.dataframe(preview_df, use_container_width=True)