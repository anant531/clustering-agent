# scripts/run_pipeline.py
import argparse
from pathlib import Path
from clustering_pipeline.core.pipeline import clustering_pipeline
from clustering_pipeline.tools.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run Clustering Pipeline")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--features", type=str, nargs="+", required=True, help="Feature columns to use")
    parser.add_argument("--min_k", type=int, help="Minimum k (overrides YAML)")
    parser.add_argument("--max_k", type=int, help="Maximum k (overrides YAML)")
    parser.add_argument("--metric", type=str, 
                       choices=["silhouette", "calinski_harabasz", "davies_bouldin"], 
                       help="Scoring metric")
    parser.add_argument("--model", type=str, choices=["kmeans", "agglo"], help="Model type")
    parser.add_argument("--config", type=str, help="Optional YAML config path")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--save", action="store_true", help="Save output to file")

    args = parser.parse_args()
    
    # Load configuration
    config = load_config(Path(args.config)) if args.config else load_config()

    # Merge CLI args with config (CLI takes precedence)
    k_range = (
        args.min_k or config.clustering.k_range[0],
        args.max_k or config.clustering.k_range[1]
    )
    scoring_metric = args.metric or config.evaluation.metrics[0]
    model_type = args.model or config.clustering.algorithms[0]
    
    # Handle output path
    output_path = Path(args.output) if args.output else Path("outputs/clustered_output.csv")

    try:
        print(f"🚀 Starting clustering pipeline...")
        print(f"📁 Input file: {args.file}")
        print(f"📊 Features: {args.features}")
        print(f"🎯 K range: {k_range}")
        print(f"📈 Metric: {scoring_metric}")
        print(f"🤖 Model: {model_type}")
        print("-" * 50)
        
        # UPDATED: Handle new tuple return
        df, quality_report = clustering_pipeline(
            file_path=Path(args.file),
            feature_cols=args.features,
            k_range=k_range,
            scoring_metric=scoring_metric,
            model_type=model_type,
            save_output=args.save,
            output_path=output_path
        )
        
        # Print results
        print("\n" + "="*50)
        print("🎉 CLUSTERING COMPLETE!")
        print("="*50)
        print(f"📊 Final dataset shape: {df.shape}")
        print(f"🏆 Data quality score: {quality_report.quality_score:.3f}")
        print(f"🎯 Number of clusters: {df['cluster'].nunique()}")
        
        cluster_counts = df['cluster'].value_counts().sort_index()
        print(f"📈 Cluster distribution:")
        for cluster_id, count in cluster_counts.items():
            print(f"   Cluster {cluster_id}: {count} samples ({count/len(df)*100:.1f}%)")
        
        if quality_report.warnings:
            print(f"⚠️  Data quality warnings: {len(quality_report.warnings)}")
            for warning in quality_report.warnings[:3]:  # Show first 3
                print(f"   • {warning}")
        
        if args.save:
            print(f"💾 Results saved to: {output_path.resolve()}")
        
        print("✅ Pipeline execution successful!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
