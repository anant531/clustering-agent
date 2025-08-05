# tests/test_pipeline.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from clustering_pipeline.core.pipeline import clustering_pipeline


@pytest.fixture(scope="module")
def create_sample_csv(tmp_path_factory) -> Path:
    """Create a small synthetic dataset and save as CSV."""
    tmp_dir = tmp_path_factory.mktemp("data")
    file_path = tmp_dir / "sample.csv"
    
    np.random.seed(42)
    df = pd.DataFrame({
        "feature_1": np.random.normal(0, 1, 100),
        "feature_2": np.random.normal(5, 1, 100),
        "feature_3": np.random.normal(-2, 1, 100),  # Extra feature for testing
        "id": range(100)  # Non-numeric column
    })
    df.to_csv(file_path, index=False)
    return file_path


def test_pipeline_end_to_end(create_sample_csv):
    """Run full clustering pipeline and check results."""
    # UPDATED: Expect tuple return
    df, quality_report = clustering_pipeline(
        file_path=create_sample_csv,
        feature_cols=["feature_1", "feature_2"],
        k_range=(2, 4),
        scoring_metric="silhouette",
        model_type="kmeans",
        save_output=False,
    )

    # Test DataFrame results
    assert "cluster" in df.columns
    assert df["cluster"].nunique() >= 2
    assert df["cluster"].nunique() <= 4  # Within k_range
    assert not df.isnull().any().any()
    assert df.shape[0] == 100
    
    # Test quality report
    assert quality_report.quality_score > 0.0
    assert quality_report.quality_score <= 1.0
    assert quality_report.final_shape[0] == 100
    assert "feature_1" in quality_report.columns_validated
    assert "feature_2" in quality_report.columns_validated


def test_pipeline_with_agglo(create_sample_csv):
    """Test pipeline with agglomerative clustering."""
    df, quality_report = clustering_pipeline(
        file_path=create_sample_csv,
        feature_cols=["feature_1", "feature_2"],
        k_range=(2, 3),
        scoring_metric="silhouette",
        model_type="agglo",  # Test agglomerative
        save_output=False,
    )
    
    assert "cluster" in df.columns
    assert df["cluster"].nunique() >= 2
    assert quality_report.quality_score > 0.0


def test_pipeline_different_metrics(create_sample_csv):
    """Test pipeline with different scoring metrics."""
    metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
    
    for metric in metrics:
        df, quality_report = clustering_pipeline(
            file_path=create_sample_csv,
            feature_cols=["feature_1", "feature_2"],
            k_range=(2, 3),
            scoring_metric=metric,
            model_type="kmeans",
            save_output=False,
        )
        
        assert "cluster" in df.columns
        assert df["cluster"].nunique() >= 2
        print(f"✅ {metric} metric test passed")