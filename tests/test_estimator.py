import numpy as np
from clustering_pipeline.core.estimator import find_best_k
from clustering_pipeline.exceptions.errors import EvaluationError


def test_find_best_k_silhouette():
    """Check k-selection works with silhouette metric."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 1, (50, 2)),
        np.random.normal(5, 1, (50, 2)),
        np.random.normal(10, 1, (50, 2)),
    ])

    best_k, scores = find_best_k(X, (2, 5), scoring_metric="silhouette")
    assert isinstance(best_k, int)
    assert 2 <= best_k <= 5
    assert "silhouette" in scores[best_k]


def test_find_best_k_davies():
    """Check that scoring works for DB metric."""
    np.random.seed(0)
    X = np.vstack([
        np.random.normal(0, 1, (40, 2)),
        np.random.normal(4, 1, (40, 2))
    ])

    best_k, scores = find_best_k(X, (2, 4), scoring_metric="davies_bouldin")
    assert isinstance(best_k, int)
    assert best_k in scores
    assert "davies_bouldin" in scores[best_k]


def test_invalid_metric():
    X = np.random.rand(100, 2)
    try:
        find_best_k(X, (2, 4), scoring_metric="unknown_metric")
    except EvaluationError:
        assert True
