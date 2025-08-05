# tests/test_clustering.py
import pytest
import numpy as np
from clustering_pipeline.core.clustering import (
    KMeansClusterer,
    AgglomerativeClusterer,
    ClusterResult
)
from clustering_pipeline.exceptions.errors import ClusteringError


@pytest.fixture
def dummy_data():
    """Generates 2D dummy data."""
    return np.random.rand(100, 2)


@pytest.mark.parametrize("clusterer_class, n_clusters", [
    (KMeansClusterer, 3),
    (AgglomerativeClusterer, 2),
])
def test_cluster_output_shape_and_metrics(clusterer_class, n_clusters, dummy_data):
    clusterer = clusterer_class(n_clusters=n_clusters)
    result = clusterer.fit_predict(dummy_data)

    assert isinstance(result, ClusterResult)
    assert result.labels.shape[0] == dummy_data.shape[0]
    assert isinstance(result.metrics, dict)
    assert all(metric in result.metrics for metric in ["silhouette", "davies_bouldin", "calinski_harabasz"])


def test_kmeans_fails_on_invalid_k(dummy_data):
    with pytest.raises(ClusteringError):
        clusterer = KMeansClusterer(n_clusters=-1)
        clusterer.fit_predict(dummy_data)


def test_agglo_fails_on_invalid_k(dummy_data):
    with pytest.raises(ClusteringError):
        clusterer = AgglomerativeClusterer(n_clusters=0)
        clusterer.fit_predict(dummy_data)
