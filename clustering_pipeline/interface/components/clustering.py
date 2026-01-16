# # clustering_pipeline/interface/components/clustering.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import time
# from typing import Dict, List, Tuple, Any, Optional
# import warnings
# warnings.filterwarnings('ignore')


# def preprocess_data(df: pd.DataFrame, features: List[str], scaler_type: str = 'standard', handle_missing: str = 'mean') -> Tuple[np.ndarray, Dict]:
#     """Preprocess data for clustering."""
#     preprocessing_info = {
#         'original_shape': df.shape,
#         'features_used': features,
#         'scaler': scaler_type,
#         'missing_strategy': handle_missing
#     }
    
#     # Extract and clean data
#     feature_data = df[features].copy()
    
#     if handle_missing == 'mean':
#         feature_data = feature_data.fillna(feature_data.mean())
#     elif handle_missing == 'median':
#         feature_data = feature_data.fillna(feature_data.median())
#     elif handle_missing == 'drop':
#         feature_data = feature_data.dropna()
    
#     preprocessing_info['processed_shape'] = feature_data.shape
#     preprocessing_info['missing_removed'] = df.shape[0] - feature_data.shape[0]
    
#     # Scale data
#     scalers = {
#         'standard': StandardScaler(),
#         'minmax': MinMaxScaler(),
#         'robust': RobustScaler(),
#         'none': None
#     }
    
#     scaler = scalers[scaler_type]
#     if scaler is not None:
#         scaled_data = scaler.fit_transform(feature_data)
#         preprocessing_info['scaler_fitted'] = scaler
#     else:
#         scaled_data = feature_data.values
    
#     return scaled_data, preprocessing_info


# def find_optimal_clusters(data: np.ndarray, max_clusters: int = 10, algorithm: str = 'kmeans') -> Dict:
#     """Find optimal number of clusters using multiple metrics."""
    
#     if data is None or len(data) < 4:
#         st.error("❌ Need at least 4 data points for optimization")
#         return {}
    
#     results = {'k_values': [], 'silhouette_scores': [], 'davies_bouldin_scores': [], 'inertias': []}
    
#     max_k = min(max_clusters, len(data) - 1)
#     k_range = range(2, max_k + 1)
    
#     progress_bar = st.progress(0)
    
#     for i, k in enumerate(k_range):
#         try:
#             if algorithm == 'kmeans':
#                 clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
#             elif algorithm == 'agglomerative':
#                 clusterer = AgglomerativeClustering(n_clusters=k)
#             else:
#                 continue
                
#             labels = clusterer.fit_predict(data)
            
#             if len(np.unique(labels)) >= 2:
#                 results['k_values'].append(k)
#                 results['silhouette_scores'].append(silhouette_score(data, labels))
#                 results['davies_bouldin_scores'].append(davies_bouldin_score(data, labels))
                
#                 if hasattr(clusterer, 'inertia_'):
#                     results['inertias'].append(clusterer.inertia_)
                    
#         except Exception as e:
#             st.warning(f"⚠️ Error with {k} clusters: {str(e)}")
        
#         progress_bar.progress((i + 1) / len(k_range))
    
#     progress_bar.empty()
    
#     # Find optimal k
#     if results['silhouette_scores']:
#         results['optimal_k'] = results['k_values'][np.argmax(results['silhouette_scores'])]
#         st.success(f"✅ Recommended clusters: {results['optimal_k']}")
    
#     return results


# def perform_clustering(data: np.ndarray, algorithm: str, params: Dict) -> Tuple[np.ndarray, Dict, Any]:
#     """Perform clustering with specified algorithm."""
    
#     if data is None or len(data) == 0:
#         raise ValueError("No data provided for clustering")
    
#     start_time = time.time()
    
#     # Initialize clusterer
#     if algorithm == 'kmeans':
#         clusterer = KMeans(
#             n_clusters=params.get('n_clusters', 3),
#             random_state=42,
#             n_init=10,
#             max_iter=params.get('max_iter', 300)
#         )
#     elif algorithm == 'agglomerative':
#         clusterer = AgglomerativeClustering(
#             n_clusters=params.get('n_clusters', 3),
#             linkage=params.get('linkage', 'ward')
#         )
#     elif algorithm == 'dbscan':
#         clusterer = DBSCAN(
#             eps=params.get('eps', 0.5),
#             min_samples=params.get('min_samples', 5)
#         )
#     else:
#         raise ValueError(f"Unknown algorithm: {algorithm}")
    
#     # Fit and calculate metrics
#     labels = clusterer.fit_predict(data)
    
#     metrics = {
#         'execution_time': time.time() - start_time,
#         'n_clusters_found': len(np.unique(labels)),
#         'algorithm': algorithm
#     }
    
#     # Calculate clustering quality metrics
#     if len(np.unique(labels)) > 1:
#         try:
#             metrics['silhouette_score'] = silhouette_score(data, labels)
#             metrics['davies_bouldin_score'] = davies_bouldin_score(data, labels)
#         except Exception:
#             metrics['silhouette_score'] = None
#             metrics['davies_bouldin_score'] = None
    
#     # Algorithm-specific metrics
#     if hasattr(clusterer, 'inertia_'):
#         metrics['inertia'] = clusterer.inertia_
    
#     if algorithm == 'dbscan':
#         metrics['n_noise_points'] = np.sum(labels == -1)
#         metrics['noise_percentage'] = (metrics['n_noise_points'] / len(labels)) * 100
    
#     return labels, metrics, clusterer


# def create_cluster_visualization_3d(data: np.ndarray, labels: np.ndarray, features: List[str], method: str = 'pca') -> go.Figure:
#     """Create 3D visualization of clusters using PCA or original features."""
    
#     # Determine visualization method and prepare data
#     if method == 'pca' and data.shape[1] >= 3:
#         pca = PCA(n_components=3, random_state=42)
#         data_3d = pca.fit_transform(data)
#         x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
#         y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
#         z_label = f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
#         title = "3D Cluster Visualization (PCA)"
        
#     elif len(features) >= 3:
#         data_3d = data[:, :3]
#         x_label, y_label, z_label = features[:3]
#         title = f"3D Cluster Visualization ({', '.join(features[:3])})"
        
#     else:
#         # Fallback: use PCA even with fewer dimensions
#         if data.shape[1] == 2:
#             # Add a dummy third dimension
#             data_3d = np.column_stack([data, np.zeros(len(data))])
#             x_label = features[0] if features else "Feature 1"
#             y_label = features[1] if len(features) > 1 else "Feature 2"
#             z_label = "Dummy Dimension"
#         elif data.shape[1] == 1:
#             # Add two dummy dimensions
#             data_3d = np.column_stack([data, np.zeros(len(data)), np.zeros(len(data))])
#             x_label = features[0] if features else "Feature 1"
#             y_label = "Dummy Dimension 1"
#             z_label = "Dummy Dimension 2"
#         else:
#             # Use PCA to reduce to 3D
#             pca = PCA(n_components=min(3, data.shape[1]), random_state=42)
#             pca_data = pca.fit_transform(data)
#             if pca_data.shape[1] < 3:
#                 # Pad with zeros if needed
#                 padding = np.zeros((len(pca_data), 3 - pca_data.shape[1]))
#                 data_3d = np.column_stack([pca_data, padding])
#             else:
#                 data_3d = pca_data
#             x_label = "PC1"
#             y_label = "PC2" 
#             z_label = "PC3"
            
#         title = "3D Cluster Visualization"
    
#     # Create 3D scatter plot
#     fig = go.Figure()
    
#     # Get unique clusters and assign colors
#     unique_clusters = np.unique(labels)
#     colors = px.colors.qualitative.Set3
    
#     for i, cluster in enumerate(unique_clusters):
#         cluster_mask = labels == cluster
#         cluster_name = f"Noise" if cluster == -1 else f"Cluster {cluster}"
        
#         fig.add_trace(go.Scatter3d(
#             x=data_3d[cluster_mask, 0],
#             y=data_3d[cluster_mask, 1],
#             z=data_3d[cluster_mask, 2],
#             mode='markers',
#             name=cluster_name,
#             marker=dict(
#                 size=6,
#                 color=colors[i % len(colors)],
#                 opacity=0.6 if cluster == -1 else 0.8,
#                 line=dict(width=0.5, color='black')
#             ),
#             text=[f"{cluster_name}<br>Index: {idx}" for idx in np.where(cluster_mask)[0]],
#             hovertemplate="<b>%{text}</b><br>" +
#                          f"{x_label}: %{{x:.3f}}<br>" +
#                          f"{y_label}: %{{y:.3f}}<br>" +
#                          f"{z_label}: %{{z:.3f}}<extra></extra>"
#         ))
    
#     # Update layout
#     fig.update_layout(
#         title=title,
#         scene=dict(
#             xaxis_title=x_label,
#             yaxis_title=y_label,
#             zaxis_title=z_label,
#             camera=dict(
#                 eye=dict(x=1.5, y=1.5, z=1.5)
#             )
#         ),
#         width=800,
#         height=600,
#         showlegend=True
#     )
    
#     return fig


# def show_clustering_metrics(metrics: Dict):
#     """Display clustering metrics in a clean format."""
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Clusters Found", metrics.get('n_clusters_found', 'N/A'))
    
#     with col2:
#         score = metrics.get('silhouette_score')
#         if score is not None:
#             delta = "Good" if score > 0.5 else "Fair" if score > 0.25 else "Poor"
#             st.metric("Silhouette Score", f"{score:.3f}", delta=delta)
#         else:
#             st.metric("Silhouette Score", "N/A")
    
#     with col3:
#         score = metrics.get('davies_bouldin_score')
#         if score is not None:
#             delta = "Good" if score < 1.0 else "Fair" if score < 2.0 else "Poor"
#             st.metric("Davies-Bouldin Score", f"{score:.3f}", delta=delta)
#         else:
#             st.metric("Davies-Bouldin Score", "N/A")
    
#     with col4:
#         st.metric("Execution Time", f"{metrics.get('execution_time', 0):.2f}s")
    
#     # Additional info
#     if metrics.get('n_noise_points', 0) > 0:
#         st.warning(f"⚠️ DBSCAN found {metrics['n_noise_points']} noise points ({metrics['noise_percentage']:.1f}%)")
    
#     if metrics.get('inertia') is not None:
#         st.info(f"ℹ️ K-Means inertia: {metrics['inertia']:.2f}")


# def show_optimization_results(results: Dict):
#     """Show optimization results with plots."""
    
#     if not results.get('k_values'):
#         st.warning("No optimization results to display")
#         return
    
#     # Create optimization plots
#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=('Silhouette Score vs Clusters', 'Elbow Method (Inertia)'),
#         specs=[[{"secondary_y": False}, {"secondary_y": False}]]
#     )
    
#     k_values = results['k_values']
    
#     # Silhouette score
#     if results.get('silhouette_scores'):
#         fig.add_trace(
#             go.Scatter(
#                 x=k_values, 
#                 y=results['silhouette_scores'], 
#                 mode='lines+markers', 
#                 name='Silhouette',
#                 line=dict(color='green', width=3),
#                 marker=dict(size=8)
#             ),
#             row=1, col=1
#         )
    
#     # Elbow method
#     if results.get('inertias') and any(x is not None for x in results['inertias']):
#         inertias = [x for x in results['inertias'] if x is not None]
#         fig.add_trace(
#             go.Scatter(
#                 x=k_values[:len(inertias)], 
#                 y=inertias, 
#                 mode='lines+markers', 
#                 name='Inertia',
#                 line=dict(color='red', width=3),
#                 marker=dict(size=8)
#             ),
#             row=1, col=2
#         )
    
#     fig.update_layout(height=400, showlegend=False, title_text="Cluster Optimization Analysis")
#     fig.update_xaxes(title_text="Number of Clusters")
    
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Show recommendation
#     if results.get('optimal_k'):
#         st.success(f"🎯 **Recommended number of clusters:** {results['optimal_k']}")


# def render_clustering_tab():
#     """Main clustering interface."""
    
#     # Check prerequisites
#     if st.session_state.df is None:
#         st.warning("📁 Please upload data first")
#         return
    
#     if not st.session_state.selected_features:
#         st.warning("🔧 Please select features first")
#         return
    
#     df = st.session_state.df
#     features = st.session_state.selected_features
    
#     # Filter numeric features
#     numeric_features = [f for f in features if f in df.select_dtypes(include=[np.number]).columns]
    
#     if len(numeric_features) < 2:
#         st.error("❌ Need at least 2 numeric features for clustering")
#         return
    
#     st.markdown("### 🤖 Clustering Analysis")
#     st.markdown(f"**Features:** {', '.join(numeric_features)}")
    
#     # Configuration
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("**Data Preprocessing**")
#         scaler_type = st.selectbox("Scaling", ["standard", "minmax", "robust", "none"])
#         missing_strategy = st.selectbox("Missing Values", ["mean", "median", "drop"])
    
#     with col2:
#         st.markdown("**Algorithm Settings**")
#         algorithm = st.selectbox("Algorithm", ["kmeans", "agglomerative", "dbscan"])
        
#         # Algorithm parameters
#         if algorithm in ["kmeans", "agglomerative"]:
#             n_clusters = st.slider("Clusters", 2, 10, 3)
#             params = {'n_clusters': n_clusters}
            
#             if algorithm == "agglomerative":
#                 linkage = st.selectbox("Linkage", ["ward", "complete", "average"])
#                 params['linkage'] = linkage
                
#         else:  # DBSCAN
#             eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
#             min_samples = st.slider("Min Samples", 2, 20, 5)
#             params = {'eps': eps, 'min_samples': min_samples}
    
#     # Optimization section
#     if algorithm in ["kmeans", "agglomerative"]:
#         st.markdown("---")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             max_clusters = st.slider("Max Clusters to Test", 3, 12, 8)
            
#             if st.button("🔍 Find Optimal Clusters"):
#                 with st.spinner("Optimizing..."):
#                     processed_data, _ = preprocess_data(df, numeric_features, scaler_type, missing_strategy)
#                     results = find_optimal_clusters(processed_data, max_clusters, algorithm)
#                     st.session_state.optimization_results = results
        
#         with col2:
#             if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
#                 results = st.session_state.optimization_results
#                 if results.get('optimal_k'):
#                     st.success(f"🎯 Recommended: {results['optimal_k']} clusters")
#                     if st.button("Use Recommendation"):
#                         params['n_clusters'] = results['optimal_k']
#                         st.rerun()
        
#         # Show optimization plots
#         if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
#             show_optimization_results(st.session_state.optimization_results)
    
#     # Clustering execution
#     st.markdown("---")
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.button("🚀 Run Clustering", type="primary"):
#             with st.spinner("Clustering..."):
#                 try:
#                     # Process and cluster
#                     processed_data, preprocessing_info = preprocess_data(df, numeric_features, scaler_type, missing_strategy)
#                     labels, metrics, clusterer = perform_clustering(processed_data, algorithm, params)
                    
#                     # Store results
#                     result_df = df.copy()
#                     result_df['Cluster'] = labels
                    
#                     st.session_state.update({
#                         'clustered_df': result_df,
#                         'cluster_labels': labels,
#                         'clustering_metrics': metrics,
#                         'clusterer': clusterer,
#                         'processed_data': processed_data,
#                         'preprocessing_info': preprocessing_info,
#                         'clustering_params': {
#                             'algorithm': algorithm,
#                             'params': params,
#                             'features': numeric_features,
#                             'scaler': scaler_type,
#                             'missing_strategy': missing_strategy
#                         },
#                         'clustering_complete': True
#                     })
                    
#                     st.success("✅ Clustering completed!")
#                     st.balloons()
                    
#                 except Exception as e:
#                     st.error(f"❌ Error: {str(e)}")
    
#     with col2:
#         if getattr(st.session_state, 'clustering_complete', False):
#             if st.button("🔄 Reset"):
#                 st.session_state.clustering_complete = False
#                 st.rerun()
    
#     with col3:
#         if getattr(st.session_state, 'clustering_complete', False):
#             if st.button("➡️ View Results"):
#                 st.success("✅ Check the Results tab!")
    
#     # Show results
#     if getattr(st.session_state, 'clustering_complete', False) and hasattr(st.session_state, 'clustering_metrics'):
#         st.markdown("---")
#         st.markdown("### 📊 Results")
        
#         # Metrics
#         show_clustering_metrics(st.session_state.clustering_metrics)
        
#         # 3D Visualization
#         if hasattr(st.session_state, 'processed_data') and st.session_state.processed_data is not None:
#             st.markdown("#### 🎨 3D Cluster Visualization")
#             fig_3d = create_cluster_visualization_3d(
#                 st.session_state.processed_data,
#                 st.session_state.cluster_labels,
#                 numeric_features,
#                 method='pca'
#             )
#             st.plotly_chart(fig_3d, use_container_width=True)
        
#         # Cluster summary
#         if st.session_state.clustered_df is not None:
#             st.markdown("#### 📋 Cluster Distribution")
#             cluster_counts = st.session_state.clustered_df['Cluster'].value_counts().sort_index()
            
#             summary_df = pd.DataFrame({
#                 'Cluster': cluster_counts.index,
#                 'Count': cluster_counts.values,
#                 'Percentage': (cluster_counts.values / len(st.session_state.clustered_df) * 100).round(1)
#             })
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.dataframe(summary_df, use_container_width=True)
#             with col2:
#                 # Simple bar chart
#                 fig_bar = px.bar(summary_df, x='Cluster', y='Count', title='Points per Cluster')
#                 st.plotly_chart(fig_bar, use_container_width=True)

# clustering_pipeline/interface/components/clustering.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
from typing import Dict, List, Tuple, Any, Optional
import warnings
from .import ai_agent
warnings.filterwarnings('ignore')


def preprocess_data(df: pd.DataFrame, features: List[str], scaler_type: str = 'standard', handle_missing: str = 'mean') -> Tuple[np.ndarray, Dict]:
    """Preprocess data for clustering."""
    preprocessing_info = {
        'original_shape': df.shape,
        'features_used': features,
        'scaler': scaler_type,
        'missing_strategy': handle_missing
    }
    
    # Extract and clean data
    feature_data = df[features].copy()
    
    if handle_missing == 'mean':
        feature_data = feature_data.fillna(feature_data.mean())
    elif handle_missing == 'median':
        feature_data = feature_data.fillna(feature_data.median())
    elif handle_missing == 'drop':
        feature_data = feature_data.dropna()
    
    preprocessing_info['processed_shape'] = feature_data.shape
    preprocessing_info['missing_removed'] = df.shape[0] - feature_data.shape[0]
    
    # Scale data
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'none': None
    }
    
    scaler = scalers[scaler_type]
    if scaler is not None:
        scaled_data = scaler.fit_transform(feature_data)
        preprocessing_info['scaler_fitted'] = scaler
    else:
        scaled_data = feature_data.values
    
    return scaled_data, preprocessing_info


def find_optimal_clusters(data: np.ndarray, max_clusters: int = 10, algorithm: str = 'kmeans') -> Dict:
    """Find optimal number of clusters using multiple metrics."""
    
    if data is None or len(data) < 4:
        st.error("❌ Need at least 4 data points for optimization")
        return {}
    
    results = {'k_values': [], 'silhouette_scores': [], 'davies_bouldin_scores': [], 'inertias': []}
    
    max_k = min(max_clusters, len(data) - 1)
    k_range = range(2, max_k + 1)
    
    progress_bar = st.progress(0)
    
    for i, k in enumerate(k_range):
        try:
            if algorithm == 'kmeans':
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            elif algorithm == 'agglomerative':
                clusterer = AgglomerativeClustering(n_clusters=k)
            else:
                continue
                
            labels = clusterer.fit_predict(data)
            
            if len(np.unique(labels)) >= 2:
                results['k_values'].append(k)
                results['silhouette_scores'].append(silhouette_score(data, labels))
                results['davies_bouldin_scores'].append(davies_bouldin_score(data, labels))
                
                if hasattr(clusterer, 'inertia_'):
                    results['inertias'].append(clusterer.inertia_)
                    
        except Exception as e:
            st.warning(f"⚠️ Error with {k} clusters: {str(e)}")
        
        progress_bar.progress((i + 1) / len(k_range))
    
    progress_bar.empty()
    
    # Find optimal k
    if results['silhouette_scores']:
        results['optimal_k'] = results['k_values'][np.argmax(results['silhouette_scores'])]
        st.success(f"✅ Recommended clusters: {results['optimal_k']}")
    
    return results


def perform_clustering(data: np.ndarray, algorithm: str, params: Dict) -> Tuple[np.ndarray, Dict, Any]:
    """Perform clustering with specified algorithm."""
    
    if data is None or len(data) == 0:
        raise ValueError("No data provided for clustering")
    
    start_time = time.time()
    
    # Initialize clusterer
    if algorithm == 'kmeans':
        clusterer = KMeans(
            n_clusters=params.get('n_clusters', 3),
            random_state=42,
            n_init=10,
            max_iter=params.get('max_iter', 300)
        )
    elif algorithm == 'agglomerative':
        clusterer = AgglomerativeClustering(
            n_clusters=params.get('n_clusters', 3),
            linkage=params.get('linkage', 'ward')
        )
    elif algorithm == 'dbscan':
        clusterer = DBSCAN(
            eps=params.get('eps', 0.5),
            min_samples=params.get('min_samples', 5)
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Fit and calculate metrics
    labels = clusterer.fit_predict(data)
    
    metrics = {
        'execution_time': time.time() - start_time,
        'n_clusters_found': len(np.unique(labels)),
        'algorithm': algorithm
    }
    
    # Calculate clustering quality metrics
    if len(np.unique(labels)) > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(data, labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(data, labels)
        except Exception:
            metrics['silhouette_score'] = None
            metrics['davies_bouldin_score'] = None
    
    # Algorithm-specific metrics
    if hasattr(clusterer, 'inertia_'):
        metrics['inertia'] = clusterer.inertia_
    
    if algorithm == 'dbscan':
        metrics['n_noise_points'] = np.sum(labels == -1)
        metrics['noise_percentage'] = (metrics['n_noise_points'] / len(labels)) * 100
    
    return labels, metrics, clusterer


def create_cluster_visualization_3d(data: np.ndarray, labels: np.ndarray, features: List[str], method: str = 'pca') -> go.Figure:
    """Create 3D visualization of clusters using PCA or original features."""
    
    # Determine visualization method and prepare data
    if method == 'pca' and data.shape[1] >= 3:
        pca = PCA(n_components=3, random_state=42)
        data_3d = pca.fit_transform(data)
        x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
        y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
        z_label = f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
        title = "3D Cluster Visualization (PCA)"
        
    elif len(features) >= 3:
        data_3d = data[:, :3]
        x_label, y_label, z_label = features[:3]
        title = f"3D Cluster Visualization ({', '.join(features[:3])})"
        
    else:
        # Fallback: use PCA even with fewer dimensions
        if data.shape[1] == 2:
            # Add a dummy third dimension
            data_3d = np.column_stack([data, np.zeros(len(data))])
            x_label = features[0] if features else "Feature 1"
            y_label = features[1] if len(features) > 1 else "Feature 2"
            z_label = "Dummy Dimension"
        elif data.shape[1] == 1:
            # Add two dummy dimensions
            data_3d = np.column_stack([data, np.zeros(len(data)), np.zeros(len(data))])
            x_label = features[0] if features else "Feature 1"
            y_label = "Dummy Dimension 1"
            z_label = "Dummy Dimension 2"
        else:
            # Use PCA to reduce to 3D
            pca = PCA(n_components=min(3, data.shape[1]), random_state=42)
            pca_data = pca.fit_transform(data)
            if pca_data.shape[1] < 3:
                # Pad with zeros if needed
                padding = np.zeros((len(pca_data), 3 - pca_data.shape[1]))
                data_3d = np.column_stack([pca_data, padding])
            else:
                data_3d = pca_data
            x_label = "PC1"
            y_label = "PC2" 
            z_label = "PC3"
            
        title = "3D Cluster Visualization"
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Get unique clusters and assign colors
    unique_clusters = np.unique(labels)
    colors = px.colors.qualitative.Set3
    
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = labels == cluster
        cluster_name = f"Noise" if cluster == -1 else f"Cluster {cluster}"
        
        fig.add_trace(go.Scatter3d(
            x=data_3d[cluster_mask, 0],
            y=data_3d[cluster_mask, 1],
            z=data_3d[cluster_mask, 2],
            mode='markers',
            name=cluster_name,
            marker=dict(
                size=6,
                color=colors[i % len(colors)],
                opacity=0.6 if cluster == -1 else 0.8,
                line=dict(width=0.5, color='black')
            ),
            text=[f"{cluster_name}<br>Index: {idx}" for idx in np.where(cluster_mask)[0]],
            hovertemplate="<b>%{text}</b><br>" +
                         f"{x_label}: %{{x:.3f}}<br>" +
                         f"{y_label}: %{{y:.3f}}<br>" +
                         f"{z_label}: %{{z:.3f}}<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600,
        showlegend=True
    )
    
    return fig


def show_clustering_metrics(metrics: Dict):
    """Display clustering metrics in a clean format."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Clusters Found", metrics.get('n_clusters_found', 'N/A'))
    
    with col2:
        score = metrics.get('silhouette_score')
        if score is not None:
            delta = "Good" if score > 0.5 else "Fair" if score > 0.25 else "Poor"
            st.metric("Silhouette Score", f"{score:.3f}", delta=delta)
        else:
            st.metric("Silhouette Score", "N/A")
    
    with col3:
        score = metrics.get('davies_bouldin_score')
        if score is not None:
            delta = "Good" if score < 1.0 else "Fair" if score < 2.0 else "Poor"
            st.metric("Davies-Bouldin Score", f"{score:.3f}", delta=delta)
        else:
            st.metric("Davies-Bouldin Score", "N/A")
    
    with col4:
        st.metric("Execution Time", f"{metrics.get('execution_time', 0):.2f}s")
    
    # Additional info
    if metrics.get('n_noise_points', 0) > 0:
        st.warning(f"⚠️ DBSCAN found {metrics['n_noise_points']} noise points ({metrics['noise_percentage']:.1f}%)")
    
    if metrics.get('inertia') is not None:
        st.info(f"ℹ️ K-Means inertia: {metrics['inertia']:.2f}")


def show_optimization_results(results: Dict):
    """Show optimization results with plots."""
    
    if not results.get('k_values'):
        st.warning("No optimization results to display")
        return
    
    # Create optimization plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Silhouette Score vs Clusters', 'Elbow Method (Inertia)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    k_values = results['k_values']
    
    # Silhouette score
    if results.get('silhouette_scores'):
        fig.add_trace(
            go.Scatter(
                x=k_values, 
                y=results['silhouette_scores'], 
                mode='lines+markers', 
                name='Silhouette',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    # Elbow method
    if results.get('inertias') and any(x is not None for x in results['inertias']):
        inertias = [x for x in results['inertias'] if x is not None]
        fig.add_trace(
            go.Scatter(
                x=k_values[:len(inertias)], 
                y=inertias, 
                mode='lines+markers', 
                name='Inertia',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
    
    fig.update_layout(height=400, showlegend=False, title_text="Cluster Optimization Analysis")
    fig.update_xaxes(title_text="Number of Clusters")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show recommendation
    if results.get('optimal_k'):
        st.success(f"🎯 **Recommended number of clusters:** {results['optimal_k']}")


def render_clustering_tab():
    """Main clustering interface with AI Recommendation."""

    # --- Check prerequisites ---
    if st.session_state.df is None:
        st.warning("📁 Please upload data first")
        return
    
    if not st.session_state.selected_features:
        st.warning("🔧 Please select features first")
        return
    
    df = st.session_state.df
    features = st.session_state.selected_features

    # Numeric feature check
    numeric_features = [f for f in features if f in df.select_dtypes(include=[np.number]).columns]
    if len(numeric_features) < 2:
        st.error("❌ Need at least 2 numeric features for clustering")
        return

    st.markdown("### 🤖 Clustering Analysis")
    st.markdown(f"**Features:** {', '.join(numeric_features)}")

    # --- AI Recommendation Section ---
    if "clustering_recommendation" not in st.session_state:
        st.session_state.clustering_recommendation = ai_agent.get_clustering_recommendation(df, numeric_features)

    rec = st.session_state.clustering_recommendation

    st.markdown("### 🤖 AI Recommendation")
    st.write(f"**Recommended:** {rec['algorithm']} with k={rec['k']}")
    st.write(f"**Why:** {rec['reason']}")

    colA, colB = st.columns(2)
    with colA:
        st.write("✅ Pros:")
        for p in rec["pros"]:
            st.write(f"- {p}")
    with colB:
        st.write("⚠️ Cons:")
        for c in rec["cons"]:
            st.write(f"- {c}")

    col1, col2 = st.columns(2)
    with col1:
        accept = st.button("✅ Accept & Run")
    with col2:
        modify = st.button("⚙️ Modify Settings")

    # --- Accept & Run (skip manual config) ---
    if accept:
        with st.spinner("Clustering with AI recommendation..."):
            try:
                processed_data, preprocessing_info = preprocess_data(df, numeric_features, "standard", "mean")
                labels, metrics, clusterer = perform_clustering(
                    processed_data,
                    rec["algorithm"].lower(),
                    {"n_clusters": rec["k"]}
                )

                result_df = df.copy()
                result_df['Cluster'] = labels

                st.session_state.update({
                    'clustered_df': result_df,
                    'cluster_labels': labels,
                    'clustering_metrics': metrics,
                    'clusterer': clusterer,
                    'processed_data': processed_data,
                    'preprocessing_info': preprocessing_info,
                    'clustering_params': {
                        'algorithm': rec['algorithm'],
                        'params': {"n_clusters": rec["k"]},
                        'features': numeric_features,
                        'scaler': "standard",
                        'missing_strategy': "mean"
                    },
                    'clustering_complete': True
                })

                st.success("✅ Clustering completed with AI recommendation!")
                st.balloons()
                return  # stop here, no manual config shown

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return

    # --- Manual Config Section (only if Modify clicked) ---
    if not modify:
        st.info("👉 Modify settings to explore other algorithms, or accept the AI recommendation above.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Data Preprocessing**")
        scaler_type = st.selectbox("Scaling", ["standard", "minmax", "robust", "none"])
        missing_strategy = st.selectbox("Missing Values", ["mean", "median", "drop"])
    
    with col2:
        st.markdown("**Algorithm Settings**")
        algorithm = st.selectbox("Algorithm", ["kmeans", "agglomerative", "dbscan"])
        
        if algorithm in ["kmeans", "agglomerative"]:
            n_clusters = st.slider("Clusters", 2, 10, 3)
            params = {'n_clusters': n_clusters}
            
            if algorithm == "agglomerative":
                linkage = st.selectbox("Linkage", ["ward", "complete", "average"])
                params['linkage'] = linkage
                
        else:  # DBSCAN
            eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
            min_samples = st.slider("Min Samples", 2, 20, 5)
            params = {'eps': eps, 'min_samples': min_samples}
    
    # Optimization
    if algorithm in ["kmeans", "agglomerative"]:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            max_clusters = st.slider("Max Clusters to Test", 3, 12, 8)
            
            if st.button("🔍 Find Optimal Clusters"):
                with st.spinner("Optimizing..."):
                    processed_data, _ = preprocess_data(df, numeric_features, scaler_type, missing_strategy)
                    results = find_optimal_clusters(processed_data, max_clusters, algorithm)
                    st.session_state.optimization_results = results
        
        with col2:
            if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
                results = st.session_state.optimization_results
                if results.get('optimal_k'):
                    st.success(f"🎯 Recommended: {results['optimal_k']} clusters")
                    if st.button("Use Recommendation"):
                        params['n_clusters'] = results['optimal_k']
                        st.rerun()
        
        if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
            show_optimization_results(st.session_state.optimization_results)
    
    # Run Clustering
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Run Clustering", type="primary"):
            with st.spinner("Clustering..."):
                try:
                    processed_data, preprocessing_info = preprocess_data(df, numeric_features, scaler_type, missing_strategy)
                    labels, metrics, clusterer = perform_clustering(processed_data, algorithm, params)
                    
                    result_df = df.copy()
                    result_df['Cluster'] = labels
                    
                    st.session_state.update({
                        'clustered_df': result_df,
                        'cluster_labels': labels,
                        'clustering_metrics': metrics,
                        'clusterer': clusterer,
                        'processed_data': processed_data,
                        'preprocessing_info': preprocessing_info,
                        'clustering_params': {
                            'algorithm': algorithm,
                            'params': params,
                            'features': numeric_features,
                            'scaler': scaler_type,
                            'missing_strategy': missing_strategy
                        },
                        'clustering_complete': True
                    })
                    
                    st.success("✅ Clustering completed!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if getattr(st.session_state, 'clustering_complete', False):
            if st.button("🔄 Reset"):
                st.session_state.clustering_complete = False
                st.rerun()
    
    with col3:
        if getattr(st.session_state, 'clustering_complete', False):
            if st.button("➡️ View Results"):
                st.success("✅ Check the Results tab!")
    
    # Show results
    if getattr(st.session_state, 'clustering_complete', False) and hasattr(st.session_state, 'clustering_metrics'):
        st.markdown("---")
        st.markdown("### 📊 Results")
        
        show_clustering_metrics(st.session_state.clustering_metrics)
        
        if hasattr(st.session_state, 'processed_data') and st.session_state.processed_data is not None:
            st.markdown("#### 🎨 3D Cluster Visualization")
            fig_3d = create_cluster_visualization_3d(
                st.session_state.processed_data,
                st.session_state.cluster_labels,
                numeric_features,
                method='pca'
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        if st.session_state.clustered_df is not None:
            st.markdown("#### 📋 Cluster Distribution")
            cluster_counts = st.session_state.clustered_df['Cluster'].value_counts().sort_index()
            
            summary_df = pd.DataFrame({
                'Cluster': cluster_counts.index,
                'Count': cluster_counts.values,
                'Percentage': (cluster_counts.values / len(st.session_state.clustered_df) * 100).round(1)
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(summary_df, use_container_width=True)
            with col2:
                fig_bar = px.bar(summary_df, x='Cluster', y='Count', title='Points per Cluster')
                st.plotly_chart(fig_bar, use_container_width=True)
