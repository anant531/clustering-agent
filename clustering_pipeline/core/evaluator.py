# clustering_pipeline/core/evaluator.py
from typing import Dict
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from clustering_pipeline.tools.logger import get_logger
from clustering_pipeline.exceptions.errors import EvaluationError

logger = get_logger(__name__)

# === METRIC COMPUTATION ===

def compute_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute clustering validation metrics.
    """
    try:
        metrics = {
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels)
        }
        logger.info(f"[METRICS] Raw metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"[METRICS] Failed to compute clustering metrics: {e}")
        raise EvaluationError(str(e)) from e


# === METRIC NORMALIZATION ===

class NormalizationConfig:
    def __init__(self, db_max: float = 2.0, ch_clip_max: float = 500.0):
        self.db_max = db_max
        self.ch_clip_max = ch_clip_max

def normalize_metrics(scores: Dict[str, float], config: NormalizationConfig = NormalizationConfig()) -> Dict[str, float]:
    """
    Normalize clustering metrics to comparable [0,1] range.
    """
    try:
        norm = {
            "silhouette": (scores["silhouette"] + 1) / 2,
            "davies_bouldin": max(0.0, min(1.0, 1 - scores["davies_bouldin"] / config.db_max)),
            "calinski_harabasz": min(1.0, scores["calinski_harabasz"] / config.ch_clip_max)
        }
        logger.info(f"[NORMALIZED] Metrics: {norm}")
        return norm
    except Exception as e:
        logger.error(f"[NORMALIZED] Failed to normalize: {e}")
        raise EvaluationError(f"Failed to normalize metrics: {e}") from e


# === SCORE AGGREGATION ===

def combined_score(norm_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Compute weighted sum of normalized metrics.
    """
    try:
        if not np.isclose(sum(weights.values()), 1.0):
            raise EvaluationError("Weights must sum to 1.0")

        total = sum(weights[metric] * norm_scores[metric] for metric in weights)
        logger.info(f"[SCORE] Combined weighted score: {total:.4f}")
        return total
    except KeyError as e:
        raise EvaluationError(f"Metric not found: {e}")
    except Exception as e:
        raise EvaluationError(f"Failed to compute combined score: {e}") from e
