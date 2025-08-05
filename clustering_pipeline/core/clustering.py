# clustering_pipeline/core/clustering.py
from abc import ABC, abstractmethod
from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

from clustering_pipeline.tools.logger import get_logger
from clustering_pipeline.exceptions.errors import ClusteringError

logger = get_logger(__name__)


@dataclass
class ClusterResult:
    labels: np.ndarray
    model: object
    metrics: Dict[str, float]


class BaseClusterer(ABC):
    """Base class for clustering algorithms."""
    
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> ClusterResult:
        """Fit the model and predict cluster labels."""
        pass

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality using centralized evaluator."""
        # Import here to avoid circular imports
        from clustering_pipeline.core.evaluator import compute_metrics
        try:
            return compute_metrics(X, labels)
        except Exception as e:
            logger.warning(f"Clustering evaluation failed: {e}")
            return {"silhouette": -1, "davies_bouldin": 999, "calinski_harabasz": 0}


class KMeansClusterer(BaseClusterer):
    """K-Means clustering implementation."""
    
    def fit_predict(self, X: np.ndarray) -> ClusterResult:
        try:
            # X should already be scaled by the pipeline
            model = KMeans(
                n_clusters=self.n_clusters, 
                random_state=self.random_state, 
                n_init="auto"
            )
            labels = model.fit_predict(X)
            metrics = self.evaluate(X, labels)
            
            logger.info(f"KMeans clustering complete. k={self.n_clusters} | "
                       f"silhouette={metrics.get('silhouette', 0):.3f}")
            
            return ClusterResult(labels, model, metrics)
        except Exception as e:
            logger.error(f"KMeans clustering failed: {e}")
            raise ClusteringError(f"KMeans clustering failed: {e}") from e


class AgglomerativeClusterer(BaseClusterer):
    """Agglomerative clustering implementation."""
    
    def fit_predict(self, X: np.ndarray) -> ClusterResult:
        try:
            # X should already be scaled by the pipeline
            model = AgglomerativeClustering(n_clusters=self.n_clusters)
            labels = model.fit_predict(X)
            metrics = self.evaluate(X, labels)
            
            logger.info(f"Agglomerative clustering complete. k={self.n_clusters} | "
                       f"silhouette={metrics.get('silhouette', 0):.3f}")
            
            return ClusterResult(labels, model, metrics)
        except Exception as e:
            logger.error(f"Agglomerative clustering failed: {e}")
            raise ClusteringError(f"Agglomerative clustering failed: {e}") from e


# BACKWARD COMPATIBILITY FUNCTION
def run_clustering(X: np.ndarray, k: int, model_type: str) -> Tuple[np.ndarray, object]:
    """
    Backward compatibility function for existing pipeline code.
    
    Args:
        X: Pre-scaled feature matrix
        k: Number of clusters
        model_type: "kmeans" or "agglo"
    
    Returns:
        Tuple of (labels, model) for compatibility
    """
    if model_type == "kmeans":
        clusterer = KMeansClusterer(n_clusters=k)
    elif model_type in ["agglo", "agglomerative"]:
        clusterer = AgglomerativeClusterer(n_clusters=k)
    else:
        available = ["kmeans", "agglo", "agglomerative"]
        raise ClusteringError(f"Unsupported model type: {model_type}. Available: {available}")
    
    result = clusterer.fit_predict(X)
    return result.labels, result.model