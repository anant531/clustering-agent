# clustering_pipeline/core/estimator.py
from typing import Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from clustering_pipeline.tools.logger import get_logger
from clustering_pipeline.exceptions.errors import EvaluationError

logger = get_logger(__name__)


def find_best_k(
    X: np.ndarray,
    k_range: Tuple[int, int],
    scoring_metric: str = "silhouette",
) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    Find the best k value based on clustering evaluation metrics.
    
    Args:
        X: Raw feature matrix (will be scaled internally)
        k_range: Range of k values to try (min_k, max_k)
        scoring_metric: Metric to optimize ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    
    Returns:
        Tuple of (best_k, all_scores_dict)
    """
    # Validate scoring metric
    valid_metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
    if scoring_metric not in valid_metrics:
        raise EvaluationError(f"Unsupported scoring metric: {scoring_metric}. "
                             f"Valid options: {valid_metrics}")
    
    # Scale data once for all k values
    logger.info("Scaling data for k optimization...")
    X_scaled = StandardScaler().fit_transform(X)
    logger.info(f"Data scaled. Shape: {X_scaled.shape}")
    
    scores_by_k: Dict[int, Dict[str, float]] = {}
    
    # Try each k value
    for k in range(k_range[0], k_range[1] + 1):
        try:
            logger.info(f"Testing k={k}...")
            
            # Fit KMeans
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(X_scaled)
            
            # Evaluate using centralized function
            from clustering_pipeline.core.evaluator import compute_metrics
            scores = compute_metrics(X_scaled, labels)
            scores_by_k[k] = scores
            
            logger.info(f"[K={k}] {scoring_metric}={scores[scoring_metric]:.4f} | "
                       f"all_metrics={scores}")
            
        except Exception as e:
            logger.warning(f"K={k} failed: {e}")
            continue
    
    # Check if any k values worked
    if not scores_by_k:
        raise EvaluationError(f"No valid clustering results found in range {k_range}")
    
    # Select best k based on metric direction
    if scoring_metric == "davies_bouldin":
        # Lower is better for Davies-Bouldin
        best_k = min(scores_by_k, key=lambda k: scores_by_k[k][scoring_metric])
        best_score = scores_by_k[best_k][scoring_metric]
        direction = "minimize"
    else:
        # Higher is better for silhouette and Calinski-Harabasz
        best_k = max(scores_by_k, key=lambda k: scores_by_k[k][scoring_metric])
        best_score = scores_by_k[best_k][scoring_metric]
        direction = "maximize"
    
    logger.info(f"✅ Best k={best_k} selected ({direction} {scoring_metric}={best_score:.4f})")
    return best_k, scores_by_k