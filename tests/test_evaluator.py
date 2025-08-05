import numpy as np
import pytest
from clustering_pipeline.core.evaluator import (
    compute_metrics,
    normalize_metrics,
    combined_score,
    NormalizationConfig
)
from clustering_pipeline.exceptions.errors import EvaluationError


def make_simple_clusters():
    # Two well-separated clusters in 2D
    a = np.random.normal(0, 0.1, (20, 2))
    b = np.random.normal(5, 0.1, (20, 2))
    X = np.vstack([a, b])
    labels = np.array([0] * 20 + [1] * 20)
    return X, labels


def test_compute_metrics():
    """Test raw metric calculation with valid clusters."""
    X, labels = make_simple_clusters()
    scores = compute_metrics(X, labels)

    assert isinstance(scores, dict)
    assert scores["silhouette"] > 0.5
    assert scores["davies_bouldin"] < 1.0
    assert scores["calinski_harabasz"] > 0


def test_normalize_metrics():
    """Test normalization of raw metrics."""
    raw = {"silhouette": 0.5, "davies_bouldin": 0.5, "calinski_harabasz": 100}
    norm = normalize_metrics(raw, config=NormalizationConfig(db_max=2.0, ch_clip_max=200))
    
    assert 0 <= norm["silhouette"] <= 1
    assert 0 <= norm["davies_bouldin"] <= 1
    assert 0 <= norm["calinski_harabasz"] <= 1


def test_combined_score_valid():
    """Test weighted score aggregation with valid weights."""
    norm = {"silhouette": 0.8, "davies_bouldin": 0.2, "calinski_harabasz": 0.5}
    weights = {"silhouette": 0.5, "davies_bouldin": 0.2, "calinski_harabasz": 0.3}
    score = combined_score(norm, weights)
    
    expected = 0.5 * 0.8 + 0.2 * 0.2 + 0.3 * 0.5
    assert pytest.approx(expected, rel=1e-6) == score


def test_combined_score_invalid_weights():
    """Should raise if weights do not sum to 1.0."""
    norm = {"silhouette": 0.8, "davies_bouldin": 0.2}
    weights = {"silhouette": 0.7, "davies_bouldin": 0.2}  # sum != 1.0
    with pytest.raises(EvaluationError):
        combined_score(norm, weights)
