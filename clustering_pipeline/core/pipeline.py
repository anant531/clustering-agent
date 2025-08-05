#clustering_pipeline/core/pipeline.py
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from clustering_pipeline.tools.logger import get_logger
from clustering_pipeline.core.data_loader import load_data, DataValidationConfig, DataQualityReport
from clustering_pipeline.core.estimator import find_best_k
from clustering_pipeline.core.clustering import KMeansClusterer, AgglomerativeClusterer

from clustering_pipeline.core.evaluator import (
    compute_metrics,
    normalize_metrics,
    combined_score,
)
from clustering_pipeline.exceptions.errors import PipelineError, EvaluationError

logger = get_logger(__name__)


def clustering_pipeline(
    file_path: Path,
    feature_cols: List[str],
    k_range: Tuple[int, int] = (2, 6),
    scoring_metric: str = "silhouette",
    model_type: str = "kmeans",
    weights: Optional[dict] = None,
    save_output: bool = False,
    output_path: Path = Path("outputs/clustered_output.csv"),
    data_config: Optional[DataValidationConfig] = None,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Run the full clustering pipeline with enhanced data validation.

    Args:
        file_path: Path to CSV file
        feature_cols: Features to use for clustering
        k_range: k min and max values to try
        scoring_metric: Metric to optimize ("silhouette", "calinski_harabasz", "davies_bouldin")
        model_type: "kmeans" or "agglo"
        weights: Weights for combined score (optional)
        save_output: Save clustered CSV
        output_path: Where to save output
        data_config: Data validation configuration

    Returns:
        Tuple of (clustered DataFrame with 'cluster' column, data quality report)
    """
    try:
        # STEP 1: Load and validate data
        logger.info(f"🚀 Starting clustering pipeline...")
        logger.info(f"📁 Input: {file_path}")
        logger.info(f"📊 Features: {feature_cols}")
        logger.info(f"🎯 K range: {k_range}, Metric: {scoring_metric}, Model: {model_type}")
        
        df, quality_report = load_data(file_path, feature_cols, data_config)
        logger.info(f"✅ Data loaded. Quality score: {quality_report.quality_score:.3f}")
        
        # STEP 2: Prepare feature matrix
        X = df[feature_cols].values
        logger.info(f"📈 Feature matrix shape: {X.shape}")
        
        # STEP 3: Find optimal k
        logger.info(f"🔍 Finding optimal k in range {k_range}...")
        best_k, all_scores = find_best_k(X, k_range, scoring_metric)
        logger.info(f"🎯 Optimal k selected: {best_k}")
        
        # STEP 4: Perform final clustering with best k
        logger.info(f"🤖 Running final clustering with k={best_k}...")
        
        # Scale data for final clustering
        X_scaled = StandardScaler().fit_transform(X)
        
        # Create appropriate clusterer
        if model_type == "kmeans":
            clusterer = KMeansClusterer(n_clusters=best_k)
        elif model_type in ["agglo", "agglomerative"]:
            clusterer = AgglomerativeClusterer(n_clusters=best_k)
        else:
            available = ["kmeans", "agglo", "agglomerative"]
            raise PipelineError(f"Unsupported model type: {model_type}. Available: {available}")
        
        # Get final results
        result = clusterer.fit_predict(X_scaled)
        df["cluster"] = result.labels
        
        # STEP 5: Compute final metrics and combined score
        final_metrics = result.metrics
        norm_scores = normalize_metrics(final_metrics)
        
        if weights is None:
            weights = {
                "silhouette": 0.5,
                "davies_bouldin": 0.2,
                "calinski_harabasz": 0.3,
            }
        
        total_score = combined_score(norm_scores, weights)
        
        # STEP 6: Log final results
        logger.info(f"📊 Final Metrics: {final_metrics}")
        logger.info(f"🏆 Combined Score: {total_score:.4f}")
        
        cluster_counts = df["cluster"].value_counts().sort_index()
        logger.info(f"📈 Cluster Distribution: {dict(cluster_counts)}")
        
        # STEP 7: Save output if requested
        if save_output:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"💾 Output saved to: {output_path.resolve()}")
        
        logger.info(f"✅ Pipeline complete! Best K={best_k}, Quality={quality_report.quality_score:.3f}")
        return df, quality_report

    except (PipelineError, EvaluationError) as pe:
        logger.error(f"❌ Pipeline error: {pe}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected pipeline failure: {e}")
        raise PipelineError(f"Unexpected pipeline failure: {e}") from e